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

def quat2rotm(q):
    return quat2rotm_(q)[0:3, 0:3]

# Download NTUVIRAL Groundtruth,
# !rm -rf ntuviral_gt
# !git clone https://github.com/ntu-aris/ntuviral_gt

# Set the ground truth path here
gtgen_res_path = '/home/wangzhong/mydata/VIRAL/ntuviral_gt/'

# Downdload sample fast-lio2 estimate
# !rm -rf fastlio2_sample
# !git clone https://github.com/ntu-aris/fastlio2_sample

# Set the path to the logs of your slam estimate
slam_est_path = '/home/wangzhong/mydata/VIRAL/experiments/'

# Set the directory where results are exported
output_dir = slam_est_path + '/analysis'
os.makedirs(output_dir, exist_ok=True)

# region Customizable for each method ---------------------

# Offset from body center to the prism
t_B_prism = np.array([-0.293656, -0.012288, -0.273095]).reshape((3,1))

# endregion Customizable for each method -----------------
algo_name = 'fastlio2'

# Minimum completeness to judge ate
min_completeness = 5.0


# Search for the sequence name
gndtr_logs = glob.glob(gtgen_res_path + '/**/ground_truth.csv', recursive=True)
gndtr_logs = sorted(gndtr_logs)
gndtr_df   = pd.DataFrame([str(x) for x in gndtr_logs], columns=['fullpath'])

# Search for the estimates
slam_est_logs = glob.glob(slam_est_path + '/nya_01/odometry.csv', recursive=True)
slam_est_logs = sorted(slam_est_logs)
est_df = pd.DataFrame([str(x) for x in slam_est_logs if '_' in str(x)], columns=['fullpath'])

print(est_df)

# Check for the sequence
def decode_gndtr_sequence_name(x):
    sequence = x.split('/')[-2]
    return sequence

def load_csv(log):
    # Open the log
    data = np.loadtxt(log, delimiter=',', skiprows=1)

    # Pose stamp
    t    = data[:, 2]/1e9
    xyz  = data[:, 3:6]
    quat = data[:, 6:10]

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

def getGTMinTime(x):
    return gndtr_df[gndtr_df['sequence'] == x]['pose_stamped'].iloc[0][0, 0]

def getGTMaxTime(x):
    return gndtr_df[gndtr_df['sequence'] == x]['pose_stamped'].iloc[0][-1, 0]

def extract_est_data(data, t_min, t_max, path = None):
    # print(t_min, 'to', t_max)
    t = data[:, 0]/1.0e9
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

# Make the associable trajectory
def make_traj(x):
    t_gt = gndtr_df[gndtr_df['sequence'] == x['sequence']]['pose_stamped'].iloc[0][:, 0]    
    valid = []
    for n in range(0, len(x['t_est'])):
        sample_time = x['t_est'][n]
        min_diff = np.min(np.abs(t_gt - sample_time))
        if min_diff < 0.05:
            valid.append(n)
    
    t_valid = x['t_est'][valid]
    P_valid = x['P_est'][valid, :]
    Q_valid = x['Q_est'][valid, :]
    
    return PoseTrajectory3D(P_valid, Q_valid, t_valid)

def makeGTTraj(x):
    pose_stamped = gndtr_df[gndtr_df['sequence'] == x['sequence']]['pose_stamped'].iloc[0]
    gtr_traj = PoseTrajectory3D(pose_stamped[:, 1:4], pose_stamped[:, 4:8], pose_stamped[:, 0])
    return trajectory_interpolation(gtr_traj, x['traj_est'].timestamps)


def calculate_metric(traj_gtr, traj_est, fullpath):
    metric = metrics.APE(pose_relation=metrics.PoseRelation.translation_part)
    metric.process_data((traj_gtr, traj_est))
    
    print(fullpath)
    save_dir = os.path.dirname(pathlib.Path(fullpath))
    plot_mode = evo.tools.plot.PlotMode.xz
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE")
    evo.tools.plot.traj(ax, plot_mode, traj_gtr, "--", "gray", "gt")
    evo.tools.plot.traj(ax, plot_mode, traj_est, "--", "red", "est")
    fig.savefig(os.path.join(save_dir, 'aligned_traj.png'))

    return float(metric.get_result(ref_name='reference', est_name='estimate').stats['rmse'])


def make_ate_txt(odom):
    return f'{round(odom, 3):.03f}'


gndtr_df['sequence'] = gndtr_df['fullpath'].apply(lambda x : decode_gndtr_sequence_name(x))

# Create the ground truth array for each sequence
gndtr_df['pose_stamped'] = gndtr_df['fullpath'].apply(lambda x : load_csv(x))


# print(est_df['fullpath'])
# Decode the sequence
est_df['sequence'] = est_df['fullpath'].apply(lambda x : decode_est_sequence_name(x))
est_df['t_min'] = est_df['sequence'].apply( lambda x : getGTMinTime(x) )
est_df['t_max'] = est_df['sequence'].apply( lambda x : getGTMaxTime(x) )


# Load the estimate data
est_df['data_est'] = est_df['fullpath'].apply(lambda x : np.loadtxt(x, delimiter=',', skiprows=1))
est_df['t_est'], est_df['P_est'], est_df['Q_est'] = zip(*map(extract_est_data, est_df['data_est'], est_df['t_min'], est_df['t_max'], est_df['fullpath']))

print(est_df)
# Sample the ground truth pose against the estimate.             
est_df['traj_est'] = est_df.apply(lambda x : make_traj(x), axis=1)
est_df['traj_gtrest'] = est_df.apply(lambda x : makeGTTraj(x), axis=1)

est_df['P_gtrest'] = est_df['traj_gtrest'].apply(lambda x : x.positions_xyz)
est_df['Q_gtrest'] = est_df['traj_gtrest'].apply(lambda x : x.orientations_quat_wxyz[:, [3, 0, 1, 2]])
est_df['t_gtrest'] = est_df['traj_gtrest'].apply(lambda x : x.timestamps)

# Align the traj
for i in est_df.index:
    est_df.at[i, 'traj_est'].align(est_df.at[i, 'traj_gtrest'])


# Calculate the ATE
est_df['ate_est'] = est_df.apply(lambda x : calculate_metric(x['traj_gtrest'], x['traj_est'], x['fullpath']), axis=1)
est_df['completeness_est'] = est_df.apply(lambda x : round((x['t_est'][-1] -  x['t_min']) / (x['t_max'] - x['t_min'])*100, 0), axis=1)

# Extracting the data to generate table of all sequences
ate_df = est_df[['sequence', 'ate_est', 'completeness_est', 'fullpath']]
# ate_df['algo'] = algo_name

def check_if_complete(x):
    # if x['ate_est'] < 20 and x['completeness_est'] < min_completeness:
    #     return float('inf')
    # else:
        return x['ate_est']

ate_df['ate_est'] = ate_df.apply(lambda x : check_if_complete(x), axis=1)

# ate_df.to_pickle(output_dir + '/ate_df.pkl')
ate_df['report'] = ate_df.apply(lambda x : make_ate_txt(x['ate_est']), axis=1)
print(ate_df)


# Making a latex export
# latex_df = ate_df[['sequence', 'report', 'completeness_est']]
# Writing the report
# print(latex_df.to_latex(index_names=['#'], header=['Seq', 'ATE', '%']))