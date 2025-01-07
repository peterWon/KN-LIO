import csv
import os
from eval_mesh_utils import eval_mesh, eval_mesh_pcd
import open3d as o3d
import numpy as np
from evo.core.transformations import quaternion_matrix 
import torch


def eval():
    ######################################## Newer College Dataset ########################################
    gt_pcd_path = "/media/wz/2C96A0A60155E8F8/Dataset/NewerCollege/math_institute/ground_truth/maths-institute.ply"
    pred_mesh_path = "/home/wz/Data/math_rr.ply"

    # evaluation parameters
    down_sample_vox = 0.02
    dist_thre = 0.2
    truncation_dist_acc = 0.4
    truncation_dist_com = 2.0
    ######################################## Newer College Dataset ########################################

    # evaluation
    eval_metric = eval_mesh(pred_mesh_path, gt_pcd_path, down_sample_res=down_sample_vox, threshold=dist_thre, 
                            truncation_acc = truncation_dist_acc, truncation_com = truncation_dist_com, gt_bbx_mask_on = True) 

    print(eval_metric)

eval()