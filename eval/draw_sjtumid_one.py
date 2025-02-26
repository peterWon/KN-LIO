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
from evo.core import lie_algebra as lie

def quat2rotm(q):
    return quat2rotm_(q)[0:3, 0:3]


# Check for the sequence
def decode_gndtr_sequence_name(x):
    sequence = x.split('/')[-2]
    return sequence

def load_csv_modify(log):
    # Open the log
    data = np.loadtxt(log, delimiter=',', skiprows=1)
    
    # Pose stamp
    t    = data[:, 2]
    xyz  = data[:, 3:6]
    quat = data[:, [9, 6, 7, 8]]

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
    t = data[:, 0]
    if t[0]>1e12 or t[0]==0:
        t = data[:, 0]/1e9
    # t = data[:, 0]
    P = data[:, 3:6]
    Q = data[:, [9, 6, 7, 8]]
    idx_intime = [ idx for idx in range(0, len(t)) if t[idx] >= t_min and t[idx] <= t_max ]

    t = t[idx_intime]
    P = P[idx_intime, :]
    Q = Q[idx_intime, :]

    # Add the pose offset
    for n in range(0, len(P)):
        P[n, :] = P[n, :] + (quat2rotm(Q[n, :]).dot(t_B_prism)).transpose()

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
        while (i < traj.num_poses and traj.timestamps[i] <= t_ref):
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
    if poses.shape[0] == len(timestamps)-1:
        poses = np.insert(poses, 0, values=poses[0,:], axis=0)
    times = poses[:, 0]
    trans = poses[:, 1:4]
    quats = poses[:, 4:8]
    return PoseTrajectory3D(trans, quats, times, meta=traj.meta)

# Make the associable trajectory
def make_traj(t_gt, t_est, P_est, Q_est):
    # t_gt = gndtr_df[gndtr_df['sequence'] == x['sequence']]['pose_stamped'].iloc[0][:, 0]    
    valid = []
    for n in range(0, len(t_est)):
        sample_time = t_est[n]
        min_diff = np.min(np.abs(t_gt - sample_time))
        # print(min_diff)
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


def calculate_metric(traj_gtr, traj_est):
    metric = metrics.APE(pose_relation=metrics.PoseRelation.translation_part)
    metric.process_data((traj_gtr, traj_est))
    return float(metric.get_result(ref_name='reference', est_name='estimate').stats['rmse'])


def make_ate_txt(odom):
    return f'{round(odom, 3):.03f}'


def get_sub_traj(traj, num_poses):
    n = traj.num_poses
    if num_poses >= n:
        raise RuntimeError
    sub_timestamps=[]
    sub_xyz=[]
    sub_quat=[]
    for i in range(num_poses):
        sub_timestamps.append(traj.timestamps[i])
        sub_xyz.append(traj.positions_xyz[i])
        sub_quat.append(traj.orientations_quat_wxyz[i])
    sub_timestamps=np.array(sub_timestamps)   
    sub_xyz=np.array(sub_xyz)   
    sub_quat=np.array(sub_quat)   
    return PoseTrajectory3D(sub_xyz, sub_quat, sub_timestamps)


gtgen_res_path = '/media/wz/2C96A0A60155E8F8/Dataset/SJTU-MID/LOG/knlio/'
result_path = '/media/wz/2C96A0A60155E8F8/Dataset/SJTU-MID/LOG/'
t_B_prism = np.array([-0.293656, -0.012288, -0.273095]).reshape((3,1))

# Search for the sequence name
sequence_name = 'forest'
gndtr = os.path.join(gtgen_res_path, sequence_name, 'odometry.viral')

gndtr_pose_stamped = load_csv_modify(gndtr)
t_min = getGTMinTime(gndtr_pose_stamped)
t_max = getGTMaxTime(gndtr_pose_stamped)

# print(t_min, t_max)

algo_compare = ['knlio','fast_lio','lio_sam','D-LIOM','pinslam']#, 'semi-knlio'
color = {'knlio':'#F0988C', 'fast_lio': '#A1A9D0', 'pinslam': '#CFEAF1', 'D-LIOM': '#F6CAE5', 'lio_sam':'#9E9E9E'}
name = {'knlio':'KN-LIO', 'fast_lio': 'Fast-LIO2', 'pinslam': 'PIN-SLAM', 'D-LIOM': 'D-LIOM', 'lio_sam':'LIO-SAM'}
# plot_mode = evo.tools.plot.PlotMode.yz #statir
plot_mode = evo.tools.plot.PlotMode.xy

fig = plt.figure()
ax = evo.tools.plot.prepare_axis(fig, plot_mode)
# ax.set_title(f"RMSE")
num_poses_to_align = 500

for algo in algo_compare:
    est = None
    if algo in ['knlio', 'semi-knlio', 'pinslam']:
        est = os.path.join(result_path, algo, sequence_name, 'odometry.viral')
    elif algo in ['fast_lio', 'lio_sam', 'D-LIOM']:
        est = os.path.join(result_path, algo, sequence_name, 'opt_odom.csv')
    else:
        raise FileNotFoundError

    est_pose_data = np.loadtxt(est, delimiter=',', skiprows=1)
    est_t, est_P, est_Q = extract_est_data(est_pose_data, t_min, t_max)

    print(algo)
    
    # traj_est = make_traj(gndtr_pose_stamped[0][:num_poses_to_align], est_t[:num_poses_to_align], est_P[:num_poses_to_align], est_Q[:num_poses_to_align])
    # traj_gt = make_traj(gndtr_pose_stamped[0][:num_poses_to_align], gndtr_pose_stamped[0][:num_poses_to_align], gndtr_pose_stamped[1][:num_poses_to_align], gndtr_pose_stamped[2][:num_poses_to_align])
    traj_est = make_traj(gndtr_pose_stamped[:,0], est_t, est_P, est_Q)
    traj_gt = makeGTTraj(gndtr_pose_stamped, traj_est)

    print(traj_est)
    print(traj_gt)
    
    sub_traj_est = get_sub_traj(traj_est, num_poses_to_align)
    sub_traj_gt = get_sub_traj(traj_gt, num_poses_to_align)

    r, t, s = sub_traj_est.align(sub_traj_gt)
    # traj_est.align(traj_gt)
    traj_est.transform(lie.se3(r, t))
    
    if algo=='knlio':
        # evo.tools.plot.traj(ax, plot_mode, traj_gt, "-", "#B883D4", "KN-LIO")
        start = sub_traj_gt.positions_xyz[0]
        if plot_mode == evo.tools.plot.PlotMode.xy:
            plt.plot(start[0], start[1], marker='^', color='black', label='start')
            plt.plot(start[0], start[1], marker='x', color='black', label='end')
        elif plot_mode == evo.tools.plot.PlotMode.yz:
            plt.plot(start[1], start[2], marker='^', color='black',label='start')
            plt.plot(start[1], start[2], marker='x', color='black', label='end')
    
    
    # rmse = calculate_metric(traj_gt, traj_est)
    # completeness_est = round((est_t[-1] -  t_min) / (t_max - t_min)*100)
    # if rmse < 3.0:
    evo.tools.plot.traj(ax, plot_mode, traj_est, "--", color[algo], name[algo])
    
plt.tight_layout()
plt.autoscale()
fig.savefig('/media/wz/2C96A0A60155E8F8/Dataset/SJTU-MID/traj/' + sequence_name+'.pdf')