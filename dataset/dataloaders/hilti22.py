# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
# 2024 Yue Pan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os
import sys
# sys.path.insert(0, '/home/wangzhong/pin-lio/')
import struct
import numpy as np
from pathlib import Path
import natsort
from utils.point_cloud2 import read_point_cloud
from utils.transformations import quaternion_matrix

from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
import cv2
from typing import cast
import math
import yaml

class Hilti2022Dataset:
    def __init__(self, data_dir: str, sequence: str, lidar_topics: dict, imu_topic: str, camera_topic: str, calibration: dict, *_, **__):
        self.bag_filename = Path(os.path.join(data_dir, sequence+'.bag'))
        self.sequence_id = sequence
        if self.bag_filename.is_file():
            self.bag = AnyReader([self.bag_filename])
            self.bag.open()
            print('Open rosbag: {}'.format(self.bag_filename))
        else:
            raise FileNotFoundError('Open rosbag: {} failed!'.format(self.bag_filename))

        # self.topic = self.check_topic(topic)
        self.lidar_topic = lidar_topics['master_lidar']
        self.imu_topic = imu_topic
        self.camera_topic = camera_topic

        # print(self.lidar_topic, self.imu_topic)

        self.parse_calibratin(calibration)
        
        self.n_scans = self.bag.topics[self.lidar_topic].msgcount
        self.n_images = self.bag.topics[self.camera_topic].msgcount

        connections = [x for x in self.bag.connections if x.topic in [self.lidar_topic, self.camera_topic, self.imu_topic]]
        self.msgs = self.bag.messages(connections=connections)
        self.pointcloud_timestamps = []
        self.image_timestamps = []

    def parse_calibratin(self, calibration_dict):
        calib_dir = calibration_dict['calibration_dir']
        lidar_yaml_path =os.path.join(calib_dir, 'lidar_calibration.yaml')
        cam0_yaml_path =os.path.join(calib_dir, 'calib_3_cam0-1-camchain-imucam.yaml')

        lidar_cfgs = self.parse_config_yaml(os.path.abspath(lidar_yaml_path))
        if lidar_cfgs is not None:
            sensors = lidar_cfgs.get('sensors')
            imu = sensors.get('imu')
            imu_intrinsics = imu['intrinsics']['parameters']
            imu_bias_a = imu_intrinsics['bias_a']
            imu_bias_g = imu_intrinsics['bias_g']
            imu_gravity = imu_intrinsics['gravity']

            hesai_lidar = sensors.get('PandarXT-32')
            hesai_lidar_extinsics = hesai_lidar['extrinsics']
            hesai_lidar_Q = hesai_lidar_extinsics['quaternion']
            hesai_lidar_P = hesai_lidar_extinsics['translation']

            # lidar parameters
            q_w_first = np.array([hesai_lidar_Q[3], hesai_lidar_Q[0], hesai_lidar_Q[1], hesai_lidar_Q[2]])
            T = quaternion_matrix(q_w_first)
            T[:3,3] = hesai_lidar_P
            self.T_IL = np.array(T)

            self.imu_bias_a = imu_bias_a
            self.imu_bias_g = imu_bias_g
            # print(hesai_lidar_P)
            # print(self.T_IL)
        
        cam_cfgs = self.parse_config_yaml(os.path.abspath(cam0_yaml_path))
        if cam_cfgs is not None:
            cam0 = cam_cfgs.get('cam0')
            cam0_extinsics = cam0['T_cam_imu']
            
            cam0_intrinsics = cam0['intrinsics']
            cam0_cx = cam0_intrinsics[0]#TODO(): verify the order
            cam0_cy = cam0_intrinsics[1]
            cam0_fx = cam0_intrinsics[2]
            cam0_fy = cam0_intrinsics[3]
            cam0_image_size = cam0['resolution']
            cam0_distortion_model = cam0['distortion_coeffs']
            cam0_k1 = cam0_distortion_model[0]
            cam0_k2 = cam0_distortion_model[1]
            cam0_k3 = cam0_distortion_model[2]
            cam0_k4 = cam0_distortion_model[3]
            cam0_model = cam0['camera_model'] #pinhole
            cam0_distortion_model = cam0['distortion_model'] #equidistant
            
            #####################################################
            # IMU noises are not provided in the calibration file！！！
            self.gravity = calibration_dict['gravity']
            self.accel_std = calibration_dict['accel_std']
            self.accel_rw = calibration_dict['accel_rw']
            self.gyro_std = calibration_dict['gyro_std']
            self.gyro_rw = calibration_dict['gyro_rw']


            # camera parameters
            self.T_IC = np.array(cam0_extinsics).reshape((4,4))
            self.fx = cam0_fx
            self.fy = cam0_fy
            self.cx = cam0_cx
            self.cy = cam0_cy
            self.width = cam0_image_size[0]
            self.height = cam0_image_size[1]
            self.fovx = self.focal2fov(self.fx, self.width)
            self.fovy = self.focal2fov(self.fy, self.height)
            self.K = np.array(
                [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
            )
            # distortion parameters
            self.disorted = True
            self.dist_coeffs = np.array(
                [
                    cam0_k1,
                    cam0_k2,
                    cam0_k3,
                    cam0_k4,
                    0,
                ]
            )
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K,
                self.dist_coeffs,
                np.eye(3),
                self.K,
                (self.width, self.height),
                cv2.CV_32FC1,
            )
    
    def parse_config_yaml(self, yaml_path, args=None):
        with open(yaml_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        if configs is not None:
            base_config = configs.get('base_config')
            if base_config is not None:
                base_config = self.parse_config_yaml(configs["base_config"])
                if base_config is not None:
                    configs = self.update_recursive(base_config, configs)
                else:
                    raise FileNotFoundError("base_config specified but not found!")

        return configs

    def convert_to_namespace(self, dict_in, args=None):
        if args is None:
            args = argparse.Namespace()
        for ckey, cvalue in dict_in.items():
            if ckey not in args.__dict__.keys():
                args.__dict__[ckey] = cvalue

        return args

    def update_recursive(self, dict1, dict2):
        for k, v in dict2.items():
            if k not in dict1:
                dict1[k] = dict()
            if isinstance(v, dict):
                self.update_recursive(dict1[k], v)
            else:
                dict1[k] = v
        return dict1
    
    def focal2fov(self, focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))

    def __del__(self):
        if hasattr(self, "bag"):
            self.bag.close()
            print('Close rosbag: {}'.format(self.bag_filename))

    def __len__(self):
        return self.n_scans

    def __next__(self):
        connection, timestamp, rawdata = next(self.msgs)
        
        msg = self.bag.deserialize(rawdata, connection.msgtype)
        # Hesai lidar记录的是每个点的采样时间戳（s）而不是相对于第一个点的时间戳
        # Hesai lidar的点云timestamp是第一个点的时间戳
        if connection.msgtype=='sensor_msgs/msg/PointCloud2':
            points, point_ts, min_ts, max_ts = read_point_cloud(msg)
            # print(min_ts, max_ts, timestamp) 
            # min_ts=0.
            # max_ts~=0.1s
            # timestamp is the first point?
            # print(self.to_sec(timestamp)-max_ts, self.to_sec(timestamp)-min_ts) 
            point_ts = (point_ts - min_ts) / (max_ts - min_ts) # normalized to 0-1
            # print(timestamp, point_ts[0], point_ts[-1])
            frame_data = {"points": points, "point_ts": point_ts, "frame_ts": max_ts} 
            self.pointcloud_timestamps.append(max_ts)
            # print(points.shape)
        elif connection.msgtype=='sensor_msgs/msg/Image':
            # https://gitlab.com/ternaris/rosbags-image/-/blob/master/src/rosbags/image/image.py?ref_type=heads
            image = message_to_cvimage(msg, 'mono8')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # if self.disorted:
            #     image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            frame_data = {"image": image, "image_ts": self.to_sec(timestamp)}
            self.image_timestamps.append(timestamp)
            # print(image.shape)
        elif connection.msgtype=='sensor_msgs/msg/Imu':
            # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html
            # orientation_x = msg.orientation.x
            # orientation_y = msg.orientation.y
            # orientation_z = msg.orientation.z
            # orientation_w = msg.orientation.w
            linear_acc_x = msg.linear_acceleration.x
            linear_acc_y = msg.linear_acceleration.y
            linear_acc_z = msg.linear_acceleration.z
            ang_vel_x = msg.angular_velocity.x
            ang_vel_y = msg.angular_velocity.y
            ang_vel_z = msg.angular_velocity.z
            reading = [ang_vel_x, ang_vel_y, ang_vel_z, linear_acc_x, linear_acc_y, linear_acc_z]
            frame_data = {"imu": reading, "imu_ts": self.to_sec(timestamp)}
            # print(frame_data)
            
        return frame_data
 

    @staticmethod
    def to_sec(nsec: int):
        return float(nsec) / 1e9

    def get_frames_timestamps(self) -> list:
        return self.pointcloud_timestamps

    def check_topic(self, topic: str) -> str:
        # Extract all PointCloud2 msg topics from the bagfile
        point_cloud_topics = [
            topic[0]
            for topic in self.bag.topics.items()
            if topic[1].msgtype == "sensor_msgs/msg/PointCloud2"
        ]

        def print_available_topics_and_exit():
            print("Select from the following topics:")
            print(50 * "-")
            for t in point_cloud_topics:
                print(f"{t}")
            print(50 * "-")
            sys.exit(1)

        if topic and topic in point_cloud_topics:
            return topic
        # when user specified the topic check that exists
        if topic and topic not in point_cloud_topics:
            print(
                f'[ERROR] Dataset does not containg any msg with the topic name "{topic}". '
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml rosbag your/topic ... ..."
            )
            print_available_topics_and_exit()
        if len(point_cloud_topics) > 1:
            print(
                "Multiple sensor_msgs/msg/PointCloud2 topics available."
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml rosbag your/topic ... ..."
            )
            print_available_topics_and_exit()

        if len(point_cloud_topics) == 0:
            print("[ERROR] Your dataset does not contain any sensor_msgs/msg/PointCloud2 topic")
        if len(point_cloud_topics) == 1:
            return point_cloud_topics[0]


if __name__ == '__main__':
    dataset = HiltiDataset(data_dir='/home/wangzhong/mydata/HILTI2021/', sequence='Basement_1',lidar_topics={'/os_cloud_node/points','/livox/lidar'}, imu_topic='/alphasense/imu', camera_topic='/alphasense/cam0/image_raw', calibration={})
    while 1:
        try:
            frame = dataset.__next__()
        except:
            break