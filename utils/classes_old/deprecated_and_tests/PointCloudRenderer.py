import time

import torch

# Util function for loading point clouds|
from kornia.geometry import relative_transformation, depth_to_3d

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras
)
from torch import Tensor

from Networks.KenburnDepth.KenburnDepth import KenburnDepth
from module.SetupCameras import CameraSetup
from utils.classes import ImageTensor
from utils.classes.Image.Image import DepthTensor


class PointCloudRenderer:
    def __init__(self, src_cam, target_cam):
        self.device = src_cam.device
        self.renderer = {}
        self._create_new_perspective_renderer(src_cam, target_cam)
        self.relative_pose = relative_transformation(target_cam.extrinsics.inverse(),
                                                     src_cam.extrinsics.inverse()).to(torch.float32)

    def _create_new_perspective_renderer(self, source, target):
        self.K_src = source.intrinsics
        cam_src = {'R': source.extrinsics[:, :3, :3].to(torch.float32),
                   'T': source.extrinsics[:, :3, 3].to(torch.float32),
                   'K': self.K_src.to(torch.float32),
                   'in_ndc': False,
                   'image_size': [(source.height, source.width)]}
        self.K_tgt = target.intrinsics
        cam_tgt = {'R': target.extrinsics[:, :3, :3].to(torch.float32),
                   'T': target.extrinsics[:, :3, 3].to(torch.float32),
                   'K': self.K_tgt.to(torch.float32),
                   'in_ndc': False,
                   'image_size': [(target.height, target.width)]}
        self.cam_src = PerspectiveCameras(device=self.device, **cam_src)
        self.cam_tgt = PerspectiveCameras(device=self.device, **cam_tgt)
        raster_settings_src = PointsRasterizationSettings(
            image_size=(int(source.height), int(source.width)),
            radius=0.003,
            points_per_pixel=10,
            bin_size=0)
        raster_settings_tgt = PointsRasterizationSettings(
            image_size=(int(target.height), int(target.width)),
            radius=0.003,
            points_per_pixel=10,
            bin_size=0)
        rasterizer_src = PointsRasterizer(cameras=self.cam_src, raster_settings=raster_settings_src)
        self.renderer_src = PointsRenderer(rasterizer=rasterizer_src, compositor=AlphaCompositor())
        rasterizer_tgt = PointsRasterizer(cameras=self.cam_tgt, raster_settings=raster_settings_tgt)
        self.renderer_tgt = PointsRenderer(rasterizer=rasterizer_tgt, compositor=AlphaCompositor())
        # Initialisation
        # self.renderer_src(Pointclouds(points=[torch.zeros([1, 3], device=self.device)], features=[torch.zeros([1, 3],
        #                                                                                                       device=self.device)]))
        # self.renderer_tgt(Pointclouds(points=[torch.zeros([1, 3], device=self.device)], features=[torch.zeros([1, 3],
        #                                                                                                       device=self.device)]))

    def __call__(self, image_src: ImageTensor, depth: DepthTensor, *args,
                 return_occlusion=True, post_process_image=3, depth_tgt=True,
                 post_process_depth=3, return_depth_reg=False, **kwargs) -> (ImageTensor, DepthTensor):
        if depth_tgt:
            # depth src is needed to create the pointcloud to be projected to tgt
            b, c, h, w = depth.shape
            M = depth.max()
            points_3d_tgt: Tensor = depth_to_3d(depth, self.K_tgt[:, :3, :3], normalize_points=True)  # Bx3xHxW
            points_3d_tgt = points_3d_tgt.permute(0, 2, 3, 1)  # BxHxWx3
            # points_3d_tgt *= 1/4
            # points_3d_tgt[:, 1] *= -1
            # points_3d_tgt[:, 2] *= -1
            # apply transformation to the 3d points
            # points_3d_src = transform_points(self.relative_pose[:, None], points_3d_tgt)  # BxHxWx3
            pointcloud_tgt = Pointclouds(points=[points_3d_tgt.reshape([h * w, 3])],
                                         features=[depth.reshape([h * w, 1]) / M])

            # DATA_DIR = "/home/aurelien/PycharmProjects/Disparity_Pipeline/data"
            # obj_filename = os.path.join(DATA_DIR, "PittsburghBridge/pointcloud.npz")
            #
            # # Load point cloud
            # pointcloud = np.load(obj_filename)
            # verts = torch.Tensor(pointcloud['verts']).to(self.device)
            # rgb = torch.Tensor(pointcloud['rgb'][:, :3]).to(self.device)
            # pointcloud_tgt = Pointclouds(points=[verts], features=[rgb])
            depth_src = self.renderer_tgt(pointcloud_tgt) * M
            depth_src.put_channel_at(1).show()
            time.sleep(1)


device = torch.device('cuda')
R = CameraSetup(from_file="/home/aurelien/PycharmProjects/Disparity_Pipeline/Setup_Camera.yaml")
NN = KenburnDepth(path_ckpt="/home/aurelien/PycharmProjects/Disparity_Pipeline/Networks/KenburnDepth/pretrained",
                  semantic_adjustment=False,
                  device=device)
# Depth
src = 'IR'
tgt = 'RGB'

# im_dst, idx = R.cameras[dst].random_image()
im_target = R.cameras[tgt].__getitem__(0)
im_src = R.cameras[src].__getitem__(0).RGB('gray')
matrix_tgt = R.cameras[tgt].intrinsics[:, :3, :3]
f_tgt = R.cameras[tgt].f
var_tgt = {'focal': f_tgt, 'intrinsics': matrix_tgt}
matrix_src = R.cameras[src].intrinsics[:, :3, :3]
f_src = R.cameras[src].f
var_src = {'focal': f_src, 'intrinsics': matrix_src}
depth = DepthTensor(NN(Tensor(im_target), **var_src)[0].clip(0, 200)).scale()

pc_render = PointCloudRenderer(R.cameras[src], R.cameras[tgt])

pc_render(im_src, depth, depth_tgt=True)

# # Setup
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")
#     torch.cuda.set_device(device)
# else:
#     device = torch.device("cpu")
#
# # Set paths
# DATA_DIR = "/home/aurelien/PycharmProjects/Disparity_Pipeline/data"
# obj_filename = os.path.join(DATA_DIR, "PittsburghBridge/pointcloud.npz")
#
# # Load point cloud
# pointcloud = np.load(obj_filename)
# verts = torch.Tensor(pointcloud['verts']).to(device)
# verts[:, 1] -= verts[:, 1].min()
# rgb = torch.Tensor(pointcloud['rgb']).to(device)
#
# start = time.time()
# point_cloud = Pointclouds(points=[verts], features=[rgb])
# time_pointcloud = time.time() - start
#
# # Initialize a camera.
# # Ro, To = look_at_view_transform(-0.00001, 0, 0)
# Ro, To = R.cameras[tgt].extrinsics[:, :3, :3].to(torch.float32), R.cameras[tgt].extrinsics[:, :3, 3].to(torch.float32)
# cameras_p = FoVPerspectiveCameras(device=device, R=Ro, T=To, znear=0.001)
# R1, T1 = R.cameras[tgt].extrinsics[:, :3, :3].to(torch.float32), R.cameras[src].extrinsics[:, :3, 3].to(torch.float32)
# # R1, T1 = look_at_view_transform(-0.000001, 0, 0)
# cameras_o = FoVPerspectiveCameras(device=device, R=R1, T=T1, znear=0.001)
#
# # Define the settings for rasterization and shading. Here we set the output image to be of size
# # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
# raster_settings = PointsRasterizationSettings(
#     image_size=(480, 640),
#     radius=0.003,
#     points_per_pixel=3,
#     bin_size=0
# )
#
# # Create a points renderer by compositing points using an alpha compositor (nearer points
# # are weighted more heavily). See [1] for an explanation.
# rasterizer_o = PointsRasterizer(cameras=cameras_o, raster_settings=raster_settings)
# rasterizer_p = PointsRasterizer(cameras=cameras_p, raster_settings=raster_settings)
# renderer_o = PointsRenderer(
#     rasterizer=rasterizer_o,
#     compositor=AlphaCompositor())
# renderer_p = PointsRenderer(
#     rasterizer=rasterizer_p,
#     compositor=AlphaCompositor())
# renderer_o(Pointclouds(points=[torch.zeros([1, 3], device=device)], features=[torch.zeros([1, 3], device=device)]))
# start = time.time()
# images_o = renderer_o(point_cloud)
# time_o = time.time() - start
# renderer_p(Pointclouds(points=[torch.zeros([1, 3], device=device)], features=[torch.zeros([1, 3], device=device)]))
# start = time.time()
# images_p = renderer_p(point_cloud)
# time_p = time.time() - start
#
# print(f"horto time : {time_o}, perspect time : {time_p}, time_pointcloud : {time_pointcloud}")
# plt.figure(figsize=(10, 10))
# plt.imshow(np.hstack((images_o[0, ..., :3].cpu().numpy(), images_p[0, ..., :3].cpu().numpy())))
# plt.axis("off")
# plt.show()
# time.sleep(1)
