import os
import sys
import glob
import shutil

log_dir = 'experiments/viral_new_param/'
des_dir = '/media/wz/2C96A0A60155E8F8/Dataset/NTU-VIRAL/LOG/'
if not os.path.exists(des_dir):
    os.mkdir(des_dir)
for log in os.listdir(log_dir):
    seq = glob.glob(os.path.join(log_dir, log) + '/**.yaml', recursive=False)
    seq = seq[0].split('/')[-1][:-5]
    # odom_names = os.listdir(os.path.join(log_dir, log, 'log'))
    odom_names = glob.glob(os.path.join(log_dir, log, 'log') + '/*_odom_poses.viral', recursive=False)
    if len(odom_names)==0: continue
    log_num = [int(fullname.split('/')[-1].split('_odom_poses')[0]) for fullname in odom_names]
    log_num = sorted(log_num)[-1]
    odom_src = os.path.join(log_dir, log, 'log', str(log_num)+'_odom_poses.viral')
  
    if not os.path.exists(os.path.join(des_dir, seq)):
        os.mkdir(os.path.join(des_dir, seq))
    shutil.copyfile(odom_src, os.path.join(des_dir, seq, 'odometry.viral'))

    mesh_names = os.listdir(os.path.join(log_dir, log, 'mesh'))
    print('copied ', seq)
    if len(mesh_names)==0: continue
    mesh_num = [int(name.split('_mesh_24cm')[0]) for name in mesh_names]
    mesh_num = sorted(mesh_num)[-1]
    mesh_src = os.path.join(log_dir, log, 'mesh', str(mesh_num)+'_mesh_24cm.ply')
    shutil.copyfile(mesh_src, os.path.join(des_dir, seq, str(mesh_num)+'_mesh_24cm.ply'))
    # print('copied ', seq)