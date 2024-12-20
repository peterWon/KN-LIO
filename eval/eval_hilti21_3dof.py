# Import some stuff

import os
import pathlib
import sys
import glob
import numpy as np
import pandas as pd
import math
# from pathlib import Path

from evo.core import metrics
from evo.core.trajectory import PoseTrajectory3D
# Conversion between quat and rotm
from evo.core.transformations import quaternion_matrix as quat2rotm_

from warnings import warn
from evo.core.trajectory import PoseTrajectory3D
from evo.core.transformations import quaternion_slerp as qslerp

import evo
from matplotlib import pyplot as plt
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS

import yaml

def parse_calibratin(yaml_path):
    with open(yaml_path, 'r') as f:
        calibs = yaml.load(f, Loader=yaml.FullLoader)

    if calibs is not None:
        sensors = calibs.get('sensors')
        imu = sensors.get('imu')

        os_lidar = sensors.get('os_lidar')
        os_lidar_extinsics = os_lidar['extrinsics']
        os_lidar_Q = os_lidar_extinsics['quaternion']
        os_lidar_P = os_lidar_extinsics['translation']
        
        # points are in os_sensor
        # os_sensor->os_lidar->imu
        os_sensor = sensors.get('os_sensor')
        os_sensor_extinsics = os_sensor['extrinsics']
        os_sensor_Q = os_sensor_extinsics['quaternion']
        os_sensor_P = os_sensor_extinsics['translation']

        # lidar parameters
        q_w_first_oslidar = np.array([os_lidar_Q[3], os_lidar_Q[0], os_lidar_Q[1], os_lidar_Q[2]])
        q_w_first_ossensor = np.array([os_sensor_Q[3], os_sensor_Q[0], os_sensor_Q[1], os_sensor_Q[2]])
        T_I_oslidar = quat2rotm_(q_w_first_oslidar)
        T_I_oslidar[:3,3] = os_lidar_P
        T_oslidar_ossensor = quat2rotm_(q_w_first_ossensor)
        T_oslidar_ossensor[:3,3] = os_sensor_P
        T_IL = T_I_oslidar @ T_oslidar_ossensor

        cam0 = sensors.get('cam0')
        cam0_extinsics = cam0['extrinsics']
        cam0_Q = cam0_extinsics['quaternion']
        cam0_P = cam0_extinsics['translation']
        q_w_first_cam0 = np.array([cam0_Q[3], cam0_Q[0], cam0_Q[1], cam0_Q[2]])
        T_I_cam0 = quat2rotm_(q_w_first_cam0)
        T_I_cam0[:3,3] = cam0_P
        
        # prism=optitrack_marker
        optitrack_marker = sensors.get('optitrack_marker')
        optitrack_marker_extinsics = optitrack_marker['extrinsics']
        optitrack_marker_Q = optitrack_marker_extinsics['quaternion']
        optitrack_marker_P = optitrack_marker_extinsics['translation']
        q_w_first_om = np.array([optitrack_marker_Q[3], optitrack_marker_Q[0], optitrack_marker_Q[1], optitrack_marker_Q[2]])
        T_cam0_om = quat2rotm_(q_w_first_om)
        T_cam0_om[:3,3] = optitrack_marker_P
        T_I_om = T_I_cam0 @ T_cam0_om
        T_I_prism = T_I_om

        pole_tip = sensors.get('pole_tip')
        pole_tip_extinsics = pole_tip['extrinsics']
        pole_tip_Q = pole_tip_extinsics['quaternion']
        pole_tip_P = pole_tip_extinsics['translation']
        q_w_first_pole = np.array([pole_tip_Q[3], pole_tip_Q[0], pole_tip_Q[1], pole_tip_Q[2]])
        T_om_pole = quat2rotm_(q_w_first_pole)
        T_om_pole[:3,3] = pole_tip_P
        T_I_pole = T_I_om @ T_om_pole

        return T_IL, T_I_prism, T_I_pole

def quat2rotm(q):
    return quat2rotm_(q)[0:3, 0:3]


# Set the ground truth path here
gtgen_res_path = '/media/wz/2C96A0A60155E8F8/Dataset/HILTI2021/ground_truth/ori/'
slam_est_path = '/media/wz/2C96A0A60155E8F8/Dataset/HILTI2021/LOG/fast_lio/'

# Offset (TODO)
T_IL, T_I_prism, T_I_pole = parse_calibratin('/media/wz/2C96A0A60155E8F8/Dataset/HILTI2021/rosbag/calibration_02.yaml')

# Minimum completeness to judge ate
min_completeness = 5.0


# Search for the sequence name
latest_log = 'result_Basement_4'

sequence_name = latest_log.split('sult_')[1]
gndtr = glob.glob(gtgen_res_path + '/'+sequence_name+'*.txt', recursive=False)[0]
est = os.path.join(slam_est_path, latest_log, 'opt_odom.csv')

basename = gndtr.split('/')[-1][:-4]
reference = basename.split('_')[-1]


# Check for the sequence
def decode_gndtr_sequence_name(x):
    sequence = x.split('/')[-2]
    return sequence

def load_csv(log):
    # Open the log
    data = np.loadtxt(log, delimiter=' ', skiprows=0)

    # Pose stamp
    t    = data[:, 0] #in seconds
    xyz  = data[:, 1:4]
    quat = data[:, 4:]

    pose_stamp = np.empty((data.shape[0], 8))
    pose_stamp[:, 0]   = t
    pose_stamp[:, 1:4] = xyz
    pose_stamp[:, 4:8] = quat
    
    # Create the spline
    return pose_stamp

def decode_est_sequence_name(x):
    dirname = os.path.dirname(x)
    seqname = dirname.split('/')[-1].replace('result_', '')
    return seqname

def getGTMinTime(gndtr_pose_stp):
    return gndtr_pose_stp[0, 0]

def getGTMaxTime(gndtr_pose_stp):
    return gndtr_pose_stp[-1, 0]

def extract_est_data(data, t_min, t_max, path = None):
    # print(t_min, 'to', t_max)
    t = data[:, 0]/1.0e9
    P = data[:, 3:6]
    Q = data[:, [9, 6, 7, 8]]
    idx_intime = [ idx for idx in range(0, len(t)) if t[idx] >= t_min and t[idx] <= t_max ]

    t = t[idx_intime]
    P = P[idx_intime, :]
    Q = Q[idx_intime, :]
    
    # TODO
    # Add the pose offset
    # for n in range(0, len(P)):
    #     P[n, :] = P[n, :] + (quat2rotm(Q[n, :]).dot(t_B_prism)).transpose()

    return (t, P, Q)

def lerp(a, b, t):
    return (1.0-t)*a + t*b

# We need to do linear interpolation, ourselves...
def cv_interp(traj, ref_timestamps, *, extrapolate_past_end=False):
    "Compute points along trajectory *est* at *timestamps* or trajectory *ref*."
    # Accept trajectories 
    if hasattr(ref_timestamps, 'timestamps'):
        ref_timestamps = ref_timestamps.timestamps

    ref_timestamps = np.array(ref_timestamps, copy=True)
    
    est_tran = traj.positions_xyz
    est_quat = traj.orientations_quat_wxyz
    
    # Index of closest next estimated timestamp for each pose in *traj*.
    # Must be >= 1 since we look at the previous pose.
    i = 1

    for j, t_ref in enumerate(ref_timestamps):
        while i < traj.num_poses and traj.timestamps[i] <= t_ref:
            i += 1
        if not extrapolate_past_end and traj.num_poses <= i:
            warn('reference trajectory ends after estimated, cut short')
            break
        t_prev = traj.timestamps[i-1]
        t_next = traj.timestamps[i]
        td     = (t_ref - t_prev) / (t_next - t_prev)
        quat_j = qslerp(est_quat[i-1], est_quat[i], td)
        tran_j = lerp(est_tran[i-1], est_tran[i], td)
        yield np.r_[t_ref, tran_j, quat_j]

def trajectory_interpolation(traj, timestamps):
    poses = np.array(list(cv_interp(traj, timestamps)))
    times = poses[:, 0]
    trans = poses[:, 1:4]
    quats = poses[:, 4:8]
    return PoseTrajectory3D(trans, quats, times, meta=traj.meta)

def sample_poses(traj, timestamps):
    poses = np.array(list(cv_interp(traj, timestamps)))
    return poses

# Make the associable trajectory
def make_traj(t_gt, t_est, P_est, Q_est):
    # t_gt = gndtr_df[gndtr_df['sequence'] == x['sequence']]['pose_stamped'].iloc[0][:, 0]    
    valid = []
    for n in range(0, len(t_est)):
        sample_time = t_est[n]
        min_diff = np.min(np.abs(t_gt - sample_time))
        if min_diff < 0.05:
            valid.append(n)
    
    t_valid = t_est[valid]
    P_valid = P_est[valid, :]
    Q_valid = Q_est[valid, :]
    
    return PoseTrajectory3D(P_valid, Q_valid, t_valid)

def makeGTTraj(gt_pose_stamped, traj_est):
    # pose_stamped = gndtr_df[gndtr_df['sequence'] == x['sequence']]['pose_stamped'].iloc[0]
    gtr_traj = PoseTrajectory3D(gt_pose_stamped[:, 1:4], gt_pose_stamped[:, 4:8], gt_pose_stamped[:, 0])
    return trajectory_interpolation(gtr_traj, traj_est.timestamps)

def sample_trajectory_at_gttimestamps(t_gt, est_t, est_P, est_Q):
    est_traj = PoseTrajectory3D(est_P, est_Q, est_t)
    return sample_poses(est_traj, t_gt)


def calculate_metric(traj_gtr, traj_est, fullpath):
    metric = metrics.APE(pose_relation=metrics.PoseRelation.translation_part)
    metric.process_data((traj_gtr, traj_est))
    
    # print(fullpath)
    save_dir = os.path.dirname(pathlib.Path(fullpath))
    plot_mode = evo.tools.plot.PlotMode.xz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE")
    evo.tools.plot.traj(ax, plot_mode, traj_gtr, "--", "gray", "gt")
    evo.tools.plot.traj(ax, plot_mode, traj_est, "--", "red", "est")
    # fig.savefig(os.path.join(save_dir, 'aligned_traj.png'))
    fig.savefig( 'aligned_traj.png')

    return float(metric.get_result(ref_name='reference', est_name='estimate').stats['rmse'])


def make_ate_txt(odom):
    return f'{round(odom, 3):.03f}'

gndtr_pose_stamped = load_csv(gndtr)
t_min = getGTMinTime(gndtr_pose_stamped)
t_max = getGTMaxTime(gndtr_pose_stamped)

est_pose_data = np.loadtxt(est, delimiter=',', skiprows=1)
est_t, est_P, est_Q = extract_est_data(est_pose_data, t_min, t_max)

traj_est = make_traj(gndtr_pose_stamped[:,0], est_t, est_P, est_Q)
poses_est_at_GTtimestamp = sample_trajectory_at_gttimestamps(gndtr_pose_stamped[:,0], est_t, est_P, est_Q)


assert(poses_est_at_GTtimestamp.shape[0] <= gndtr_pose_stamped.shape[0])
index_start = np.argmin(np.abs(poses_est_at_GTtimestamp[0,0]-gndtr_pose_stamped[:,0]))
index_end = np.argmin(np.abs(poses_est_at_GTtimestamp[-1,0]-gndtr_pose_stamped[:,0]))

poses_est = []
poses_gt = []
for i in range(poses_est_at_GTtimestamp.shape[0]):
    T = np.eye(4)
    xyz = poses_est_at_GTtimestamp[i, 1:4]
    quat = poses_est_at_GTtimestamp[i, 4:]
    T[:3,:3]=quat2rotm(quat)
    T[:3, 3]=xyz
    poses_est.append(T)

    T = np.eye(4)
    xyz = gndtr_pose_stamped[i, 1:4]
    quat = gndtr_pose_stamped[i, 4:]
    T[:3,:3]=quat2rotm(quat)
    T[:3, 3]=xyz
    poses_gt.append(T)

# \delta{T}_L=T_B^L @ \delta{T}_B @ T_L^B
if reference == 'pole':
    T_li = np.linalg.inv(T_IL)
    T_lpole = T_li @ T_I_pole
    poses_est = [T_wl @  T_lpole for T_wl in poses_est]
elif reference == 'prism':
    T_li = np.linalg.inv(T_IL)
    T_lprism = T_li @ T_I_prism
    poses_est = [T_wl @  T_lprism for T_wl in poses_est]
elif reference == 'imu':
    T_li = np.linalg.inv(T_IL)
    poses_est = [T_wl @  T_li for T_wl in poses_est]
else:
    sys.exit('Error ground truth type!')

# evaluate relative transform between the last and the first pose
delta_T_est = np.linalg.inv(poses_est[0]) @ poses_est[-1]
delta_T_gt = np.linalg.inv(poses_gt[index_start]) @ poses_gt[index_end]
err_T = np.linalg.inv(delta_T_gt) @ delta_T_est
err_trans = np.linalg.norm(err_T[:3,3])
completeness_est = round((est_t[-1] -  t_min) / (t_max - t_min)*100)
print(completeness_est, err_trans)