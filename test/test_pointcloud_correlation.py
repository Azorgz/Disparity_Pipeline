import os
import sys
import time
import numpy as np
import torch

from kornia.geometry import depth_to_3d, depth_to_3d_v2
from kornia.morphology import dilation
from open3d.examples.pipelines.robust_icp import draw_registration_result
from torch import Tensor
from Networks.Depth_anything.metric_depth.zoedepth.models.builder import build_model
from Networks.Depth_anything.metric_depth.zoedepth.utils.config import get_config
from config.Config import configure_parser
from utils.ImagesCameras import CameraSetup, DepthTensor
import open3d as o3d
import torch.nn.functional as F

# from utils.classes.KeypointsGenerator import KeypointsGenerator

sys.path.append(os.getcwd() + '/Networks/Depth_anything/metric_depth')

device = torch.device('cuda:0')

R = CameraSetup(from_file=os.getcwd() + "/Setup_Camera/Lynred_day.yaml")
parser = get_config('zoedepth', "infer")
config = configure_parser(parser,
                          None,
                          path_config=os.getcwd() + '/Networks/Depth_anything/config_Depth_anything.yml',
                          dict_vars=None)
config.pretrained_resource = config.path_checkpoint
# Depth
NN = build_model(config).eval().to(device)
res = {'val': [], 'coeff': []}
src = 'RGB'
dst = 'RGB2'
for i in range(1):
    im_dst = R.cameras[dst].__getitem__(i)#.RGB('gray')
    im_src = R.cameras[src].__getitem__(i)

    matrix_dst = R.cameras[dst].intrinsics[:, :3, :3]
    matrix_src = R.cameras[src].intrinsics[:, :3, :3]
    dist_corr_max = 20
    # keypointDetector = KeypointsGenerator(device, detector='sift_scale', matcher='snn', num_feature=10000)
    # kpts = keypointDetector(im_dst, im_src, draw_result_inplace=False)
    depth_dst = F.interpolate(NN(Tensor(im_dst), focal=matrix_dst[0, 0, 0])['metric_depth'], im_dst.shape[-2:])
    depth_src = F.interpolate(NN(Tensor(im_src), focal=matrix_src[0, 0, 0])['metric_depth'], im_src.shape[-2:])
    # depth_dist = DepthTensor(depth_src - depth_dst)
    # (im_dst.hstack(im_src)).show()
    # print(f'{torch.median(depth_dist) * 30 * 3.6} km/h')
    # cv2.imshow('Image', im_dst.opencv())
    # cv2.imshow('depth dist', depth_dist.opencv())
    # cv2.waitKey(1)
    kernel = torch.ones(3, 3).to(device)
    depth_dst = DepthTensor(dilation(depth_dst, kernel))
    depth_src = DepthTensor(dilation(depth_src, kernel))
    # depth_dst.show()
    # depth_src.show()
    # Camera Matrix

    # to PointCloud
    points_3d_dst: Tensor = depth_to_3d(depth_dst, matrix_dst, False)  # Bx3xHxW
    points_3d_dst[:, :1] *= -1
    points_3d_dst[:, 2] *= -1
    pcd_dst = o3d.t.geometry.PointCloud()
    cloud_flat = torch.flatten(points_3d_dst.put_channel_at(-1).squeeze(), start_dim=0,
                               end_dim=1).squeeze().cpu().detach().numpy()
    cloud_flat = cloud_flat[cloud_flat[:, -1] > dist_corr_max]
    pcd_dst.point.positions = o3d.core.Tensor(cloud_flat, o3d.core.Dtype.Float32)
    pcd_dst.estimate_normals()

    points_3d_src: Tensor = depth_to_3d(depth_src, matrix_src, False)  # Bx3xHxW
    points_3d_src[:, :1] *= -1
    points_3d_src[:, 2] *= -1
    pcd_src = o3d.t.geometry.PointCloud()
    cloud_flat = torch.flatten(points_3d_src.put_channel_at(-1).squeeze(), start_dim=0,
                               end_dim=1).squeeze().cpu().detach().numpy()
    cloud_flat = cloud_flat[cloud_flat[:, -1] > dist_corr_max]
    pcd_src.point.positions = o3d.core.Tensor(cloud_flat, o3d.core.Dtype.Float32)

    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
    # voxel_sizes = o3d.utility.DoubleVector([0.1, 0.1, 0.1])

    # List of Convergence-Criteria for Multi-Scale ICP:
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=0.0001,
                                                            relative_rmse=0.0001,
                                                            max_iteration=50),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 30),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 20)]

    # `max_correspondence_distances` for Multi-Scale ICP (o3d.utility.DoubleVector):
    # max_correspondence_distances = o3d.utility.DoubleVector([0.3, 0.14, 0.07])
    max_correspondence_distances = o3d.utility.DoubleVector([0.2, 0.2, 0.2])

    # Initial alignment or source to target transform.
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    # callback_after_iteration = lambda loss_log_map: print(f'iteration : {i}')
    #     "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    #         loss_log_map["iteration_index"].item(),
    #         loss_log_map["scale_index"].item(),
    #         loss_log_map["scale_iteration_index"].item(),
    #         loss_log_map["fitness"].item(),
    #         loss_log_map["inlier_rmse"].item()))
    # Setting Verbosity to Debug, helps in fine-tuning the performance.
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    s = time.time()
    try:
        registration_ms_icp = o3d.t.pipelines.registration.multi_scale_icp(pcd_src, pcd_dst, voxel_sizes,
                                                                           criteria_list,
                                                                           max_correspondence_distances,
                                                                           init_source_to_target,
                                                                           estimation, )
        # callback_after_iteration)

        # ms_icp_time = time.time() - s
        # print("Time taken by Multi-Scale ICP: ", ms_icp_time)
        # print("Inlier Fitness: ", registration_ms_icp.fitness)
        # print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)
        # print("Transformation finale: ", registration_ms_icp.transformation)
        if registration_ms_icp.fitness * registration_ms_icp.inlier_rmse != 0.0:
            res['val'].append(registration_ms_icp.transformation)
            res['coeff'].append(1 / (registration_ms_icp.fitness * registration_ms_icp.inlier_rmse))
            draw_registration_result(pcd_src, pcd_dst, registration_ms_icp.transformation)
            #
    except:
        pass

res['val'] = np.array([val.numpy() for val in res['val']])
res['coeff'] = np.array(res['coeff'])
res_mean = (res['val'] * res['coeff'].reshape([len(res['coeff']), 1, 1])).sum(0) / res['coeff'].sum()
print(res_mean)
