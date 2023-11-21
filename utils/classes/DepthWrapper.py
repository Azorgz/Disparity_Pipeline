import torch
from kornia import create_meshgrid
from kornia.filters import MedianBlur
from kornia.geometry import depth_to_3d, transform_points, project_points, normalize_pixel_coordinates, \
    denormalize_pixel_coordinates
from kornia.morphology import dilation, erosion, opening, closing
from torch import Tensor
import torch.nn.functional as F
from utils.classes import ImageTensor
import numpy as np
import open3d as o3d

from utils.classes.Image import DepthTensor


class DepthWrapper:

    def __init__(self, device):
        self.device = device

    def __call__(self, image_src: ImageTensor, depth_dst: DepthTensor, matrix_src, matrix_dst, src_trans_dst, *args,
                 return_occlusion=True, post_process_image=3,
                 post_process_depth=3, return_depth_reg=False, **kwargs) -> (ImageTensor, DepthTensor):

        """Warp a tensor from a source to destination frame by the depth in the destination.

        Compute 3d points from the depth, transform them using given transformation, then project the point cloud to an
        image plane.

        Args:
            image_src: image tensor in the source frame with shape :math:`(B,C,H,W)`.
            depth_dst: depth tensor in the destination frame with shape :math:`(B,1,H,W)`.
            src_trans_dst: transformation matrix from destination to source with shape :math:`(B,4,4)`.
            camera_matrix: tensor containing the camera intrinsics with shape :math:`(B,3,3)`.
            normalize_points: whether to normalise the pointcloud. This must be set to ``True`` when the depth
               is represented as the Euclidean ray length from the camera position.

        Return:
            the warped tensor in the source frame with shape :math:`(B,3,H,W)`.
        """
        b, c, h, w = depth_dst.shape
        kernel = torch.ones(3, 3).to(self.device)
        depth_dst = dilation(depth_dst, kernel)
        points_3d_dst: Tensor = depth_to_3d(depth_dst, matrix_dst, False)  # Bx3xHxW
        # points_3d_dst[:, :1] *= -1
        # points_3d_dst[:, 2] *= -1
        # pcd = o3d.geometry.PointCloud()
        # cloud_flat = torch.flatten(points_3d_dst.put_channel_at(-1).squeeze(), start_dim=0, end_dim=1).squeeze().cpu().numpy()
        # pcd.points = o3d.utility.Vector3dVector(cloud_flat)
        # im_flat = torch.flatten(image_dst.put_channel_at(-1).squeeze(), start_dim=0, end_dim=1).cpu().numpy()
        # pcd.colors = o3d.utility.Vector3dVector(im_flat)
        # o3d.visualization.draw_geometries([pcd],
        #                                   mesh_show_wireframe=True,
        #                                   window_name="_pointCloud_",
        #                                   point_show_normal=True,
        #                                   mesh_show_back_face=True)
        # transform points from source to destination
        points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

        # apply transformation to the 3d points
        points_3d_src = transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3

        # project back to pixels
        camera_matrix_tmp: Tensor = matrix_src[:, None, None]  # Bx1x1x3x3
        points_2d_src: Tensor = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

        # normalize points between [-1 / 1]
        height, width = image_src.shape[-2:]
        points_2d_src_norm: Tensor = normalize_pixel_coordinates(points_2d_src, height, width).to(
            image_src.dtype)  # BxHxWx2

        cloud = torch.concatenate([points_2d_src, points_3d_src[:, :, :, -1:]], dim=-1)

        grid = Tensor(points_2d_src_norm).clone()
        mask_valid = torch.ones(height*width).to(torch.bool).to(self.device)
        res = {'image_reg': F.grid_sample(image_src, grid, align_corners=True)}
        if return_occlusion:
            res['occlusion'] = self.find_occlusion(cloud, [height, width])
            res['occlusion'].im_name = image_src.im_name + '_occlusion'
            # res[:, :, mask_occlusion[0, 0, :, :]] = 0
            # mask_valid = (~res['occlusion']).flatten()
            # grid[0, mask_occlusion[0, 0, :, :], :] -= 2
        if return_depth_reg:
            depth_reg = self.compute_depth_src(cloud, [height, width], mask_valid,
                                               post_process_depth=post_process_depth)
            depth_reg.im_name = image_src.im_name + '_depth'
            res['depth_reg'] = depth_reg
        return res

    def compute_depth_src(self, cloud, size_image, mask_valid, post_process_depth=0):
        # Put all the point into a H*W x 3 vector
        c = torch.tensor(cloud.flatten(start_dim=0, end_dim=2))[mask_valid]  # H*W x 3
        # Remove the point landing outside the image
        mask = ((c[:, 0] < 0) + (c[:, 0] >= size_image[1]-1) + (c[:, 1] < 0) + (c[:, 1] >= size_image[0]-1)) == 0
        c = c[mask]
        # Transform the landing positions in accurate pixels
        c_ = torch.round(c[:, :2]).to(torch.int)
        # Create a picture with for pixel value the depth of the point landing in
        result = torch.zeros([1, 1, size_image[0], size_image[1]]).to(cloud.dtype).to(self.device) # *(c[:, 2].max()+1)
        result[0, 0, c_[:, 1], c_[:, 0]] = c[:, 2]
        # mask = result == 0
        # if post_process_depth:
        #     blur = MedianBlur(post_process_depth)
        #     kernel = torch.ones([3, 3], device=self.device)
        #     res_ = result.clone()
        #     for i in range(2):
        #         res_ = blur(res_)
        #         res_ = dilation(res_, kernel)
        #     result[mask] = res_[mask]
        #     mask = result == 0
        # result[mask] = 0
        return DepthTensor(result, device=self.device).scale()

    def find_occlusion(self, cloud, size_image):
        # Put all the point into a H*W x 3 vector
        c = torch.tensor(cloud.flatten(start_dim=0, end_dim=2))  # H*W x 3
        c[:, 0] = c[:, 0] * cloud.shape[1]/size_image[0]
        c[:, 1] = c[:, 1] * cloud.shape[2]/size_image[1]
        # create a unique index for each point according where they land in the src frame and their depth
        M = torch.round(c[:, 2].max() + 1)
        max_u = torch.round(c[:, 0].max() + 1)
        c_ = torch.round(c[:, 1]) * M * max_u + torch.round(c[:, 0]) * M + torch.round(c[:, 2])
        # Remove the point landing outside the image
        mask = ((c[:, 0] < 0) + (c[:, 0] >= cloud.shape[2]) + (c[:, 1] < 0) + (c[:, 1] >= cloud.shape[1])) > 0
        c_[mask] = 0
        # sort the point in order to find the ones landing in the same pixel
        _, indexes = c_.sort(dim=0)
        c_ = c[:, :2].clone()
        c_[mask] = 0
        # Define a new ordered vector but only using the position u,v not the depth
        c_ = torch.round(c_[indexes, 1]) * max_u + torch.round(c_[indexes, 0])
        c_depth = torch.round(c[indexes, 2])
        # Trick to find the point landing in the same pixel, only the closer is removed
        c_[1:] -= c_[:-1].clone()
        c_depth[1:] = 1 - c_depth[:-1].clone()/c_depth[1:].clone()

        idx = torch.nonzero((c_ == 0) * (c_depth > 0.03) + mask[indexes])
        # Use the indexes found to create a mask of the occluded point
        idx = indexes[idx]
        result = torch.zeros([c.shape[0], 1]).to(self.device)
        result[idx] = 1
        result = result.reshape([1, 1, cloud.shape[1], cloud.shape[2]])

        # postprocessing of the mask to remove the noise due to the round operation
        # blur = MedianBlur(3)
        # result = blur(result)
        # kernel_small = torch.ones(3, 3).to(self.device)
        # result = opening(result, kernel_small)
        # result = dilation(result, kernel_small)
        return ImageTensor(result).to(torch.bool)


    # def grid_sample_3d(self, image, depth, grid, step, post_process_depth=False, post_process_image=False):
    #     if isinstance(image, DepthTensor) and post_process_depth:
    #         post_process = True
    #     elif isinstance(image, ImageTensor) and post_process_image:
    #         post_process = True
    #     else:
    #         post_process = False
    #     sign = -1 if depth.max() <= 0 else 1
    #     max_val = torch.round(torch.abs(depth).max()) * sign
    #     if step == 0:
    #         max_val = torch.round(torch.abs(depth).max()) * sign
    #         vec = torch.arange(0, max_val + 1, step=sign, device=self.device, dtype=depth.dtype).unsqueeze(0)
    #     else:
    #         vec = torch.arange(0, 1, step=1 / (step + 1), device=self.device, dtype=depth.dtype).unsqueeze(
    #             0) * max_val * (step + 1) / step
    #     d = len(vec.squeeze())
    #     depth_ = (torch.round(((depth - depth.min()) / (depth.max() - depth.min()) * step)) / step * max_val).squeeze()
    #     vec = 1 / (vec + 10e-6)  # vec.max() - vec
    #     depth_ = 1 / (depth_ + 10e-6)  # depth_.max() - depth_
    #     vol_depth = torch.ones_like(depth_, device=self.device, dtype=depth_.dtype).unsqueeze(-1) @ vec  # HxWxD
    #     vol_mask = (torch.abs(vol_depth - depth_.unsqueeze(-1) @ torch.ones_like(vec)) <= 10e-6) \
    #         .permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # Volume mask to localize the voxel
    #     image_, m, M = image.normalize(minmax=True)
    #     vol_image = image_.match_shape(depth_).unsqueeze(2).repeat(1, 1, d, 1, 1)
    #     vol_image = (vol_mask.to(torch.float32)).match_shape(
    #         vol_image) * vol_image  # Image in volume with pixel at z = disparity
    #     #
    #     # depth = F.interpolate(depth.unsqueeze(0), size=(grid.shape[1], grid.shape[2]),
    #     #                       mode='bilinear',
    #     #                       align_corners=True).squeeze(0)
    #     z = ((vol_depth - vol_depth.min()) / (vol_depth.max() - vol_depth.min()) - 0.5) * 2
    #     vol_grid = grid.unsqueeze(1).repeat(1, d, 1, 1, 1)
    #     z = z.permute(2, 0, 1).unsqueeze(0).match_shape(vol_grid[:, :, :, :, 0]).unsqueeze(-1)
    #     vol_grid = torch.cat([vol_grid[:, :, :, :], z], dim=-1)
    #     res = F.grid_sample(vol_image, vol_grid, align_corners=True)
    #     mask = torch.ones_like(image_.match_shape(res[:, :, 0, :, :]))
    #     im = torch.zeros_like(image_.match_shape(mask))
    #     for i in reversed(range(d)):
    #         temp = res[:, :, i, :, :]
    #         mask_ = temp * mask != 0
    #         im[mask_] = temp[mask_]
    #         mask = mask + ~mask_
    #     res = im
    #     # res = res.max(2).values  # Keeping only the front voxels
    #     if res.__class__ != image.__class__:
    #         res = image.__class__(res)
    #         if isinstance(res, ImageTensor) or isinstance(res, DepthTensor):
    #             res.pass_attr(image)
    #     # if isinstance(res, DepthTensor):
    #     #     res = res.normalize().scale()
    #     res = res * (M - m) + m
    #
    #     # Postprocessing for smoothness
    #     if post_process:
    #
    #         res_ = res.clone()
    #         # mask = res < res.mean()
    #         if isinstance(res_, DepthTensor):
    #             blur = MedianBlur(post_process_depth)
    #             kernel = torch.ones(5, 5).to(self.device)
    #             res_ = erosion(res_, kernel)
    #             mask = res != 0
    #         else:
    #             blur = MedianBlur(post_process_image)
    #             mask = res_ <= 0.1
    #         for i in range(1):
    #             res_ = blur(res_)
    #         res[mask] = res_[mask]
    #
    #     return res
