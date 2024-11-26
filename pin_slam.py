#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import argparse
import os
import sys

import rerun as rr
import numpy as np
import open3d as o3d
import torch
import wandb
from rich import print
from tqdm import tqdm

from dataset.dataset_indexing import set_dataset_path
from dataset.slam_dataset import SLAMDataset
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.loop_detector import (
    NeuralPointMapContextManager,
    detect_local_loop,
)
from utils.mapper import Mapper
from utils.mesher import Mesher
from utils.pgo import PoseGraphManager
from utils.tools import (
    freeze_decoders,
    get_time,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    transform_torch,
)
from utils.tracker import Tracker
from utils.visualizer import MapVisualizer

'''
    📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
     Y. Pan et al. from IPB
'''

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, nargs='?', default='config/lidar_slam/run.yaml', help='[Optional] Path to *.yaml config file, if not set, default config would be used')
parser.add_argument('dataset_name', type=str, nargs='?', help='[Optional] Name of a specific dataset, example: kitti, mulran, or rosbag (when -d is set)')
parser.add_argument('sequence_name', type=str, nargs='?', help='[Optional] Name of a specific data sequence or the rostopic for point cloud (when -d is set)')
parser.add_argument('--seed', type=int, default=42, help='Set the random seed (default 42)')
parser.add_argument('--input_path', '-i', type=str, default=None, help='Path to the point cloud input directory (this will override the pc_path in config file)')
parser.add_argument('--output_path', '-o', type=str, default=None, help='Path to the result output directory (this will override the output_root in config file)')
parser.add_argument('--range', nargs=3, type=int, metavar=('START', 'END', 'STEP'), default=None, help='Specify the start, end and step of the processed frame, for example: --range 10 1000 1')
parser.add_argument('--data_loader_on', '-d', action='store_true', help='Use specific data loader (you can use the rosbag, pcap, mcap dataloaders and some typical supported datasets)')
parser.add_argument('--visualize', '-v', action='store_true', help='Turn on the visualizer')
parser.add_argument('--cpu_only', '-c', action='store_true', help='Run only on CPU')
parser.add_argument('--log_on', '-l', action='store_true', help='Turn on the logs printing')
parser.add_argument('--rerun_on', '-r', action='store_true', help='Turn on the rerun logging')
parser.add_argument('--wandb_on', '-w', action='store_true', help='Turn on the weight & bias logging')
parser.add_argument('--save_map', '-s', action='store_true', help='Save the PIN map after SLAM')
parser.add_argument('--save_mesh', '-m', action='store_true', help='Save the reconstructed mesh after SLAM')
parser.add_argument('--save_merged_pc', '-p', action='store_true', help='Save the merged point cloud after SLAM')

args, unknown = parser.parse_known_args()

def run_pin_slam(config_path=None, dataset_name=None, sequence_name=None, seed=None, *_, **__):

    config = Config()
    if config_path is not None: # use as a function
        print(config_path)
        config.load(config_path)
        if dataset_name is not None:
            set_dataset_path(config, dataset_name, sequence_name)
        if seed is not None:
            config.seed = seed
        argv = ['pin_slam.py', config_path, dataset_name, sequence_name, str(seed)]
        run_path = setup_experiment(config, argv)
    else: # from args
        argv = sys.argv
        config.load(args.config_path)
        config.use_dataloader = args.data_loader_on
        config.seed = args.seed
        config.silence = not args.log_on
        config.wandb_vis_on = args.wandb_on
        config.rerun_vis_on = args.rerun_on
        config.o3d_vis_on = args.visualize
        config.save_map = args.save_map
        config.save_mesh = args.save_mesh
        config.save_merged_pc = args.save_merged_pc
        if args.range is not None:
            config.begin_frame, config.end_frame, config.step_frame = args.range
        if args.cpu_only:
            config.device = 'cpu'
        if args.input_path is not None:
            config.pc_path = args.input_path
        if args.output_path is not None:
            config.output_root = args.output_path
        if args.dataset_name is not None: # specific dataset [optional]
            set_dataset_path(config, args.dataset_name, args.sequence_name)
        run_path = setup_experiment(config, argv)
        print("[bold green]PIN-SLAM starts[/bold green]","📍" )
    
    print('config.gpu_id: ', config.gpu_id)
    # non-blocking visualizer
    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config)
    
    config.rerun_vis_on = False
    if config.rerun_vis_on:
        rr.init("pin_slam_rerun_viewer", spawn=True)

    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None

    # initialize the neural points
    neural_points: NeuralPoints = NeuralPoints(config)

    # loop closure detector
    lcd_npmc = NeuralPointMapContextManager(config) # npmc: neural point map context

    mapping_on = True
    # Load the decoder model
    # for the localization with pre-built map mode, set load_model as True and provide the model path in the config file
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        neural_points = loaded_model["neural_points"]
        geo_mlp.load_state_dict(loaded_model["geo_decoder"])
        if 'sem_decoder' in loaded_model.keys():
            sem_mlp.load_state_dict(loaded_model["sem_decoder"])
        if 'color_decoder' in loaded_model.keys():
            color_mlp.load_state_dict(loaded_model["color_decoder"])
        freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)
        
        print("PIN Map loaded")  
        neural_points.recreate_hash(torch.zeros(3).to(config.device), None, True, False)
        mapping_on = False # localization mode
        neural_points.temporal_local_map_on = False # don't use travel distance for filtering
        config.pgo_on = False

    # dataset
    dataset = SLAMDataset(config)

    # odometry tracker
    tracker = Tracker(config, neural_points, geo_mlp, sem_mlp, color_mlp, dataset)
    if config.load_model and not mapping_on: 
        tracker.reg_local_map = False
        
    # mapper
    mapper = Mapper(config, dataset, neural_points, geo_mlp, sem_mlp, color_mlp)


    # mesh reconstructor
    mesher = Mesher(config, neural_points, geo_mlp, sem_mlp, color_mlp)
    cur_mesh = None

    # pose graph manager (for back-end optimization) initialization
    pgm = PoseGraphManager(config) 
    init_pose = dataset.gt_poses[0] if dataset.gt_pose_provided else np.eye(4)  
    pgm.add_pose_prior(0, init_pose, fixed=True)

    last_frame = dataset.total_pc_count-1
    loop_reg_failed_count = 0

    # save merged point cloud map from gt pose as a reference map
    # if config.save_merged_pc and dataset.gt_pose_provided:
    #     print("Load and merge the map point cloud with the reference (GT) poses ... ...")
    #     dataset.write_merged_point_cloud(use_gt_pose=True, out_file_name='merged_gt_pc', 
    #     frame_step=5, merged_downsample=True)
        
    # for each frame
    # frame id as the processed frame, possible skipping done in data loader
    frame_id = 0
    initialized_dataset = False
    pbar = tqdm(total = dataset.total_pc_count + 1) 
    while frame_id < dataset.total_pc_count:
        frame_data = dataset.read_next_datastream()
        if frame_data is None:
            dataset.write_results_log()
            print('DONE!')
            break

        if len(frame_data) == 0: continue

        # I. Load data and preprocessing
        dict_keys = list(frame_data.keys())
        if "points" in dict_keys: # TODO: support multiple LiDAR
            T_lidar_start = get_time()
            points = frame_data["points"] # may also contain intensity or color
            point_ts = frame_data["point_ts"]
            frame_ts = frame_data["frame_ts"]
            
            dataset.cur_point_cloud_torch = torch.tensor(points, device=dataset.device, dtype=dataset.dtype)
            dataset.cur_point_ts_torch = torch.tensor(point_ts, device=dataset.device, dtype=dataset.dtype)
            valid_frame = dataset.preprocess_scan(frame_id)

            if not valid_frame:
                sys.exit("Not valid frame, current frameid: ", frame_id)

            # II. Odometry
            # 用frame-to-model registration，直接求SDF场梯度，用LM-ICP，而不是像一般neural slam那样算loss，优化位姿           
            cur_lidar_pose_torch, _, weight_pc_o3d, valid_flag = tracker.process_point_cloud(
                                                frame_ts, dataset.cur_source_points, dataset.cur_point_ts_torch,
                                                dataset.cur_source_colors, dataset.cur_source_normals, dataset.cur_pose_guess_torch)
            dataset.lose_track = not valid_flag
            
            # TODO
            if not valid_flag:
                continue

            if not initialized_dataset: 
                dataset.set_initial_lidar_pose(frame_ts, cur_lidar_pose_torch)
                initialized_dataset = True
            else:
                dataset.update_odom_pose(frame_ts, cur_lidar_pose_torch)
            
            if not valid_flag and config.o3d_vis_on and o3d_vis.debug_mode > 0:
                o3d_vis.stop()
                
            
            travel_dist = dataset.travel_dist[:frame_id+1]
            neural_points.travel_dist = torch.tensor(travel_dist, device=config.device, dtype=config.dtype) # always update this
                                                                                                                                                                
            T3 = get_time()
            # print('Tracking time: ', (T3-T_lidar_start)*1e3)
            # IV: Mapping and bundle adjustment
            # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
            # if the robot stop, also don't process this frame, since there's no new oberservations
            # 插入新的neural points, 更新data pool，并滤波；当追踪失败或者机器人停止时不做更新
            if mapping_on and (frame_id < 5 or (not dataset.lose_track and not dataset.stop_status)):
                mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch,
                                    dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0))
            else:
                mapper.determine_used_pose()
                neural_points.reset_local_map(dataset.cur_pose_torch[:3,3], None, frame_id) # not efficient for large map
                                        
            T5 = get_time()

            if mapping_on:
                # for the first frame, we need more iterations to do the initialization (warm-up)
                cur_iter_num = config.iters * config.init_iter_ratio if frame_id == 0 else config.iters
                if dataset.stop_status:
                    cur_iter_num = max(1, cur_iter_num-10)
                if frame_id == config.freeze_after_frame: # freeze the decoder after certain frame 
                    freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)

                # conduct local bundle adjustment (with lower frequency)
                # 按照设定频率同时更新位姿和地图点
                if config.track_on and config.ba_freq_frame > 0 and (frame_id+1) % config.ba_freq_frame == 0:
                    mapper.bundle_adjustment(config.ba_iters, config.ba_frame)
                
                # mapping with fixed poses (every frame)
                # 优化局部地图里面neaural point的特征，不更新位姿
                if frame_id % config.mapping_freq_frame == 0:
                    mapper.mapping(cur_iter_num) 
                
                T6=get_time()
                # print('Mapping time: ', (T6-T5)*1e3)

            if (config.log_freq_frame > 0 and (frame_id+1) % config.log_freq_frame == 0) or frame_id==last_frame:
                # print('Processed {} frames'.format(frame_id+1))
                dataset.write_results_log()
            
            # V: Mesh reconstruction and visualization
            cur_mesh = None
            if config.o3d_vis_on:  # if visualizer is off, there's no need to reconstruct the meshs; 
                o3d_vis.cur_frame_id = frame_id # frame id in the data folder
                dataset.update_o3d_map()

                
                if config.track_on and frame_id > 0 and (not o3d_vis.vis_pc_color) and (weight_pc_o3d is not None): 
                    dataset.cur_frame_o3d = weight_pc_o3d

                T7 = get_time()

                if frame_id == last_frame:
                    o3d_vis.vis_global = True
                    o3d_vis.ego_view = False
                    mapper.free_pool()

                neural_pcd = None
                if o3d_vis.render_neural_points or (frame_id == last_frame): # last frame also vis
                    neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1) # select from geo_feature, ts and certainty

                # reconstruction by marching cubes
                if config.mesh_freq_frame > 0:
                    if o3d_vis.render_mesh and (frame_id == 0 or frame_id == last_frame or (frame_id+1) % config.mesh_freq_frame == 0 or pgm.last_loop_idx == frame_id):              
                        # update map bbx
                        global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23) # prime number
                        dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        
                        mesh_path = None # no need to save the mesh
                        if frame_id == last_frame and config.save_mesh: # save the mesh at the last frame
                            mc_cm_str = str(round(o3d_vis.mc_res_m*1e2))
                            mesh_path = os.path.join(run_path, "mesh", 'mesh_frame_' + str(frame_id) + "_" + mc_cm_str + "cm.ply")
                        
                        # figure out how to do it efficiently
                        if not o3d_vis.vis_global: # only build the local mesh
                            # cur_mesh = mesher.recon_aabb_mesh(dataset.cur_bbx, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
                            chunks_aabb = split_chunks(global_neural_pcd_down, dataset.cur_bbx, o3d_vis.mc_res_m * 100) # reconstruct in chunks
                            cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
                        else:
                            aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                            chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m * 300) # reconstruct in chunks
                            cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
                cur_sdf_slice = None
                if config.sdfslice_freq_frame > 0:
                    if o3d_vis.render_sdf and (frame_id == 0 or frame_id == last_frame or (frame_id + 1) % config.sdfslice_freq_frame == 0):
                        slice_res_m = config.voxel_size_m * 0.2
                        sdf_bound = config.surface_sample_range_m * 4.0
                        query_sdf_locally = True
                        if o3d_vis.vis_global:
                            cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound, sdf_bound) # horizontal slice
                        else:
                            cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # horizontal slice (local)
                        if config.vis_sdf_slice_v:
                            cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0,3], slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # vertical slice (local)
                            cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                        else:
                            cur_sdf_slice = cur_sdf_slice_h
                                    
                pool_pcd = mapper.get_data_pool_o3d(down_rate=23, only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None # down rate should be a prime number
                odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
                loop_edges = pgm.loop_edges_vis if config.pgo_on else None
                o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
                o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)

                if config.rerun_vis_on:
                    if neural_pcd is not None:
                        rr.log("world/neural_points", rr.Points3D(neural_pcd.points, colors=neural_pcd.colors, radii=0.05))
                    if dataset.cur_frame_o3d is not None:
                        rr.log("world/input_scan", rr.Points3D(dataset.cur_frame_o3d.points, colors=dataset.cur_frame_o3d.colors, radii=0.03))
                    if cur_mesh is not None:
                        rr.log("world/mesh_map", rr.Mesh3D(vertex_positions=cur_mesh.vertices, triangle_indices=cur_mesh.triangles, vertex_normals=cur_mesh.vertex_normals, vertex_colors=cur_mesh.vertex_colors))
             
            if (config.save_mesh and (frame_id+1) % config.log_freq_frame == 0) or frame_id == last_frame:
                # neural_points.prune_map(config.max_prune_certainty, 0, True) # prune uncertain points for the final output    
                # neural_points.recreate_hash(None, None, False, False) # merge the final neural point map 

                neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode = 0)
                if config.save_map:
                    o3d.io.write_point_cloud(os.path.join(run_path, "map", str(frame_id)+"_neural_points.ply"), neural_pcd) # write the neural point cloud
                if cur_mesh is None:
                    output_mc_res_m = config.mc_res_m*0.6
                    chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), output_mc_res_m * 300) # reconstruct in chunks
                    mc_cm_str = str(round(output_mc_res_m*1e2))
                    mesh_path = os.path.join(run_path, "mesh", str(frame_id)+"_mesh_" + mc_cm_str + "cm.ply")
                    cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, output_mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=config.mesh_min_nn)
                # neural_points.clear_temp() # clear temp data for output
            frame_id += 1
            dataset.processed_frame += 1
            pbar.update(1)
            T_lidar_end = get_time()
            # print((T_lidar_end-T_lidar_start)* 1e3)
        elif "image" in dict_keys: # support multiple cameras
            image = frame_data["image"]
            image_ts = frame_data['image_ts']
            # tracker.process_image(image, image_ts)
            # mapper.process_image(image, image_ts)
        elif "imu" in dict_keys:
            imus = frame_data['imu']
            imus_ts = frame_data['imu_ts']
            tracker.process_imu(imus, imus_ts)

    pbar.close()
    # regular saving logs
    # if config.log_freq_frame > 0 and (frame_id+1) % config.log_freq_frame == 0:
    #     dataset.write_results_log()

    #     dataset.processed_frame += 1
    
    # # VI. Save results
    # pose_eval_results = None
    # if config.track_on:
    #     pose_eval_results = dataset.write_results()
    # if config.pgo_on and pgm.pgo_count>0:
    #     print("# Loop corrected: ", pgm.pgo_count)
    #     pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
    #     pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
    #     if config.o3d_vis_on:
    #         pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)  
    
    # neural_points.prune_map(config.max_prune_certainty, 0, True) # prune uncertain points for the final output    
    # neural_points.recreate_hash(None, None, False, False) # merge the final neural point map 

    # neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode = 0)
    # if config.save_map:
    #     o3d.io.write_point_cloud(os.path.join(run_path, "map", "neural_points.ply"), neural_pcd) # write the neural point cloud
    # if config.save_mesh and cur_mesh is None:
    #     output_mc_res_m = config.mc_res_m*0.6
    #     chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), output_mc_res_m * 300) # reconstruct in chunks
    #     mc_cm_str = str(round(output_mc_res_m*1e2))
    #     mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
    #     cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, output_mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=config.mesh_min_nn)
    # neural_points.clear_temp() # clear temp data for output
    # if config.save_map:
    #     save_implicit_map(run_path, neural_points, geo_mlp, color_mlp, sem_mlp)
    #     # lcd_npmc.save_context_dict(mapper.used_poses, run_path)

    # if config.save_merged_pc:
    #     dataset.write_merged_point_cloud() # replay: save merged point cloud map

    # if config.o3d_vis_on:
    #     while True:
    #         o3d_vis.ego_view = False
    #         o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
    #         odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
    #         o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
    
    # return pose_eval_results

if __name__ == "__main__":
    run_pin_slam(**vars(args))