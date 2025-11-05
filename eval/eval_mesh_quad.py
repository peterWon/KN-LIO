import csv
import os
from eval_mesh_utils import eval_mesh, eval_mesh_pcd,draw_error_pcd
import open3d as o3d
import numpy as np
from evo.core.transformations import quaternion_matrix 
import torch


def eval():
    ######################################## Newer College Dataset ########################################
    gt_pcd_path = "/home/wz/Data/ncd_quad_gt_pc.ply"
    # pred_mesh_path = "/home/wz/ImMesh_output/rec_mesh_500_1799_r.ply"  #ImMesh
    # pred_mesh_path = "/media/wz/2C96A0A60155E8F8/Dataset/NewerCollege/02_long_experiment/LOG/PIN-SLAM/test_ncd_2025-10-16_10-47-44/mesh/mesh_24cm.ply"
    pred_mesh_path = "/media/wz/2C96A0A60155E8F8/Dataset/NewerCollege/02_long_experiment/LOG/KN-LIO/quad_transform_500_1799_rrr.ply"

    # evaluation parameters
    down_sample_vox = 0.02
    dist_thre = 0.2
    truncation_dist_acc = 0.4
    truncation_dist_com = 2.0
    ######################################## Newer College Dataset ########################################

    # evaluation
    eval_metric = draw_error_pcd(pred_mesh_path, gt_pcd_path, down_sample_res=down_sample_vox, threshold=dist_thre, 
                            truncation_acc = truncation_dist_acc, truncation_com = truncation_dist_com, gt_bbx_mask_on = True) 

    print(eval_metric)


eval()