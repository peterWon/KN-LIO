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
# sys.path.insert(0, '/home/wz/GS_ws/PIN_SLAM/')
import struct
import numpy as np
from pathlib import Path
import natsort
from utils.point_cloud2 import read_point_cloud

from rosbags.highlevel import AnyReader
from rosbags.image import compressed_image_to_cvimage, message_to_cvimage
import cv2
from typing import cast
import math

class NewerCollegeDataset:
    def __init__(self, data_dir: str, sequence: str, lidar_topics: dict, imu_topic: str, camera_topic: str, calibration: dict, *_, **__):
        self.sequence_id = sequence
        self.bag_dir = os.path.join(data_dir, sequence, 'rosbag')
        bagfiles = [Path(path) for path in glob.glob(os.path.join(self.bag_dir, "*.bag"))]
        if len(bagfiles) > 0:
            self.bag = AnyReader(bagfiles)
            self.bag.open()
            print('Open rosbag: {}'.format(bagfiles[0]))
        else:
            raise FileNotFoundError('Open rosbag: {} failed!'.format(bagfiles[0]))
        
        self.lidar_topic = lidar_topics['master_lidar']
        self.imu_topic = imu_topic
        self.camera_topic = camera_topic

        
        
        connections = [x for x in self.bag.connections if x.topic in [self.lidar_topic, self.camera_topic, self.imu_topic]]
        
        # using the sub-sequence as ncd_example. timestamps are from the ground_truth of 02_long_experiment
        self.start_timestamp = 1583840260*1e9+539731968 #490-th lidar frame, several more frames for initialization.
        self.stop_timestamp  = 1583840391*1e9+537008896 #1800-th lidar frame
        
        
        self.msgs = self.bag.messages(connections=connections, start=self.start_timestamp,stop=self.stop_timestamp)

        lidar_connection = [x for x in self.bag.connections if x.topic==self.lidar_topic]
        self.lidar_msgs = list(self.bag.messages(connections=lidar_connection, start=self.start_timestamp, stop=self.stop_timestamp))
        self.n_scans = len(self.lidar_msgs)
        # self.n_images = self.bag.topics[self.camera_topic].msgcount
        self.pointcloud_timestamps = []
        self.image_timestamps = []
    
        # imu parameters
        self.gravity = calibration['gravity']
        self.accel_std = calibration['accel_std']
        self.accel_rw = calibration['accel_rw']
        self.gyro_std = calibration['gyro_std']
        self.gyro_rw = calibration['gyro_rw']

        # lidar parameters
        self.T_IL = np.array(calibration['T_imu_lidar']).reshape(4,4)

        # camera parameters
        self.T_IC = np.array(calibration['T_imu_camera']).reshape(4,4)
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = self.focal2fov(self.fx, self.width)
        self.fovy = self.focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
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
    
    def focal2fov(self, focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))

    def __del__(self):
        if hasattr(self, "bag"):
            self.bag.close()
            print('Closed rosbag.')

    def __len__(self):
        return self.n_scans

    def __next__(self):
        connection, timestamp, rawdata = next(self.msgs)
        
        msg = self.bag.deserialize(rawdata, connection.msgtype)

        if connection.msgtype=='sensor_msgs/msg/PointCloud2':
            points, point_ts, min_ts, max_ts = read_point_cloud(msg) #point_ts is normalized to 0~1 in read_point_cloud()
            point_ts = (point_ts - min_ts) / (max_ts - min_ts) # normalized to 0-1
            frame_data = {"points": points, "point_ts": point_ts, "frame_ts": self.to_sec(timestamp)} 
            self.pointcloud_timestamps.append(timestamp)
        elif connection.msgtype=='sensor_msgs/msg/CompressedImage':
            # https://gitlab.com/ternaris/rosbags-image/-/blob/master/src/rosbags/image/image.py?ref_type=heads
            image = compressed_image_to_cvimage(msg,'rgb8')
            if self.disorted:
                image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            frame_data = {"image": image, "image_ts": self.to_sec(timestamp)}
            self.image_timestamps.append(timestamp)
        elif connection.msgtype=='sensor_msgs/msg/Image':
            # https://gitlab.com/ternaris/rosbags-image/-/blob/master/src/rosbags/image/image.py?ref_type=heads
            image = message_to_cvimage(msg, 'mono8')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.disorted:
                image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            frame_data = {"image": image, "image_ts": self.to_sec(timestamp)}
            self.image_timestamps.append(timestamp)
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
    viral = ViralDataset(data_dir=Path('/data0/dataset/VIRAL/eee_03/eee_03.bag'), lidar_topic='/os1_cloud_node1/points',imu_topic='/imu/imu', camera_topic='/left/image_raw')
    while 1:
        connection, timestamp, rawdata = next(viral.msgs)
        
        msg = viral.bag.deserialize(rawdata, connection.msgtype)
        if connection.msgtype=='sensor_msgs/msg/PointCloud2':
            points, point_ts = read_point_cloud(msg)
            frame_data = {"points": points, "pointcload_ts": point_ts}
        elif connection.msgtype=='sensor_msgs/msg/Image':
            img = message_to_cvimage(msg, 'mono8') #https://gitlab.com/ternaris/rosbags-image/-/blob/master/src/rosbags/image/image.py?ref_type=heads
            frame_data = {"image": img, "image_ts": timestamp}
            # print(img.shape)
        elif connection.msgtype=='sensor_msgs/msg/Imu':
            # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Imu.html
            orientation_x = msg.orientation.x
            orientation_y = msg.orientation.y
            orientation_z = msg.orientation.z
            orientation_w = msg.orientation.w
            linear_acc_x = msg.linear_acceleration.x
            linear_acc_y = msg.linear_acceleration.y
            linear_acc_z = msg.linear_acceleration.z
            ang_vel_x = msg.angular_velocity.x
            ang_vel_y = msg.angular_velocity.y
            ang_vel_z = msg.angular_velocity.z
            reading = [linear_acc_x, linear_acc_y, linear_acc_z, ang_vel_x, ang_vel_y, ang_vel_z]
            frame_data = {"imu": reading, "imu_ts": timestamp}
            print(frame_data)