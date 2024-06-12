import torch
import torch.nn.functional as F
from kornia.geometry import transform_points, project_points, normalize_pixel_coordinates, depth_to_3d_v2
from torch import Tensor
from torch.nn import MaxPool2d, Sequential
from utils.classes import ImageTensor, DepthTensor
from .utils import projector


class DepthWrapper:

    def __init__(self, device):
        self.device = device

    def __call__(self, image_src: ImageTensor, image_dst: ImageTensor, depth: DepthTensor,
                 matrix_src, matrix_dst, src_trans_dst, *args,
                 return_occlusion=True, post_process_image=3, post_process_depth=3,
                 return_depth_reg=False, reverse_wrap=False, **kwargs) -> (ImageTensor, DepthTensor):

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

        if not reverse_wrap:
            upsample = kwargs['upsample'] if 'upsample' in kwargs else 1
            res = {}
            # kernel = torch.ones(3, 3).to(self.device)
            # depth_dst = dilation(depth, kernel)
            depth_dst = depth
            points_3d_dst: Tensor = depth_to_3d_v2(depth_dst[0], matrix_dst[0], False)  # Bx3xHxW

            # apply transformation to the 3d points
            points_3d_src = transform_points(src_trans_dst[:, None].to(torch.float32), points_3d_dst)  # BxHxWx3

            # project back to pixels
            camera_matrix_tmp: Tensor = matrix_src[:, None, None]  # Bx1x1x3x3
            points_2d_src: Tensor = project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

            # define a cloud with the depth information added to the pixels position
            cloud = torch.concatenate([points_2d_src, points_3d_src[:, :, :, -1:]], dim=-1)
            height, width = image_src.shape[-2:]

            if return_depth_reg:
                depth_reg = projector(cloud, [height, width],
                                      post_process=post_process_depth,
                                      numpy=True,
                                      upsample=upsample)
                depth_reg.im_name = image_src.im_name + '_depth'
                conv_upsampling = MaxPool2d((3, 5), stride=1, padding=(1, 2), dilation=1)
                conv_upsampling = Sequential(conv_upsampling)
                res['depth_reg'] = depth_reg
                res['depth_reg'][depth_reg == 0] = conv_upsampling(depth_reg)[depth_reg == 0]

            if return_occlusion:
                res['occlusion'] = self.find_occlusion(cloud, [height, width])
                res['occlusion'].name = image_src.name + '_occlusion'
            # normalize points between [-1 / 1]
            points_2d_src_norm: Tensor = normalize_pixel_coordinates(points_2d_src, height, width).to(
                image_src.dtype)  # BxHxWx2
            grid = Tensor(points_2d_src_norm)
            image_src.data = F.grid_sample(image_src.to_tensor(), grid, align_corners=True)
            res['image_reg'] = image_src
            return res
        else:
            return self._reverse_call(image_src, image_dst, depth, matrix_src, matrix_dst, torch.inverse(src_trans_dst),
                                      *args,
                                      return_occlusion=return_occlusion, post_process_image=post_process_image,
                                      post_process_depth=post_process_depth, return_depth_reg=return_depth_reg,
                                      **kwargs)

    def _reverse_call(self, image_src: ImageTensor, image_dst: ImageTensor, depth: DepthTensor,
                      matrix_src, matrix_dst, src_trans_dst, *args,
                      return_occlusion=True, post_process_image=3, post_process_depth=3,
                      return_depth_reg=False, **kwargs) -> (ImageTensor, DepthTensor):

        """Warp a tensor from a source frame to destination frame by the depth in the src frame.

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
        upsample = kwargs['upsample'] if 'upsample' in kwargs else 1
        res = {}
        # kernel = torch.ones(3, 3).to(self.device)
        # depth_src = dilation(depth, kernel)
        depth_src = depth
        points_3d_src: Tensor = depth_to_3d_v2(depth_src[0], matrix_src[0], False)  # Bx3xHxW

        # apply transformation to the 3d points
        points_3d_dst = transform_points(src_trans_dst[:, None].to(torch.float32), points_3d_src)  # BxHxWx3

        # project back to pixels
        camera_matrix_tmp: Tensor = matrix_dst[:, None, None]  # Bx1x1x3x3
        points_2d_dst: Tensor = project_points(points_3d_dst, camera_matrix_tmp)  # BxHxWx2
        cloud = torch.concatenate([points_2d_dst, points_3d_dst[:, :, :, -1:]], dim=-1)
        height, width = image_dst.image_size

        if return_depth_reg:
            depth_reg = projector(cloud, [height, width],
                                  post_process=3,
                                  numpy=True,
                                  upsample=1)
            depth_reg.name = image_dst.name + '_depth'
            conv_upsampling = MaxPool2d((3, 5), stride=1, padding=(1, 2), dilation=1)
            conv_upsampling = Sequential(conv_upsampling)
            res['depth_reg'] = depth_reg
            res['depth_reg'][depth_reg == 0] = conv_upsampling(depth_reg)[depth_reg == 0]

        if return_occlusion:
            res['image_reg'], res['occlusion'] = projector(cloud, [height, width],
                                                           post_process=post_process_image,
                                                           image=image_src,
                                                           return_occlusion=True,
                                                           numpy=True,
                                                           upsample=upsample)
            res['occlusion'].name = image_src.name + '_occlusion'
            res['image_reg'].name = image_src.name + '_reg'
        else:
            res['image_reg'] = projector(cloud, [height, width],
                                         post_process=post_process_image,
                                         image=image_src,
                                         upsample=2)
            res['image_reg'].name = image_src.name + '_reg'

        return res

    def find_occlusion(self, cloud, image_size):
        # Put all the point into a H*W x 3 vector
        c = torch.tensor(cloud.flatten(start_dim=0, end_dim=2))  # H*W x 3
        c[:, 0] = c[:, 0] * cloud.shape[1] / image_size[0]
        c[:, 1] = c[:, 1] * cloud.shape[2] / image_size[1]

        c_xy = torch.floor(c[:, :2]).to(torch.int)
        c_Xy = torch.concatenate([torch.ceil(c[:, :1]), torch.floor(c[:, 1:2])], dim=1).to(torch.int)
        c_xY = torch.concatenate([torch.floor(c[:, :1]), torch.ceil(c[:, 1:2])], dim=1).to(torch.int)
        c_XY = torch.ceil(c[:, :2]).to(torch.int)
        rays = [c_xy, c_Xy, c_xY, c_XY]
        M = torch.round(c[:, 2].max() + 1)
        max_u = torch.round(c[:, 0].max() + 1)
        masks = torch.empty([4, cloud.shape[1] * cloud.shape[2]]).to(self.device)
        mask_out = ((c[:, 0] < 0) + (c[:, 0] >= cloud.shape[2]) + (c[:, 1] < 0) + (c[:, 1] >= cloud.shape[1])) > 0
        for i, r in enumerate(rays):
            # create a unique index for each point according where they land in the src frame and their depth
            c_ = r[:, 1] * M * max_u + r[:, 0] * M + c[:, 2]
            # Remove the point landing outside the image
            c_[mask_out] = 0
            # sort the point in order to find the ones landing in the same pixel
            _, indexes = c_.sort(dim=0)
            r[mask_out] = 0
            # Define a new ordered vector but only using the position u,v not the depth
            r = r[indexes, 1] * max_u + r[indexes, 0]
            c_depth = torch.round(c[indexes, 2])
            # Trick to find the point landing in the same pixel, only the closer is removed
            r[1:] -= r[:-1].clone()
            c_depth[1:] = 1 - c_depth[:-1].clone() / c_depth[1:].clone()

            idx = torch.nonzero((r == 0) * (c_depth > 0.03) + mask_out[indexes])
            # Use the indexes found to create a mask of the occluded point
            idx = indexes[idx]
            mask = torch.zeros([c.shape[0], 1]).to(self.device)
            mask[idx] = 1
            masks[i] = mask[:, 0]
        result = masks.sum(dim=0)
        result = result.reshape([1, 1, cloud.shape[1], cloud.shape[2]])

        # postprocessing of the mask to remove the noise due to the round operation
        # vert_conv = Conv2d(1, 1, (5, 3), stride=(1, 1), padding=(2, 1), dilation=(1, 1)).to(self.device)
        # horiz_conv = Conv2d(1, 1, (3, 5), stride=(1, 1), padding=(1, 2), dilation=(1, 1)).to(self.device)
        # result = vert_conv(result) + horiz_conv(result)
        # result = (result > 0.4).to(torch.float32)
        # kernel_small = torch.ones(3, 1).to(self.device)
        # result = closing(result, kernel_small)
        # result = opening(result, kernel_small)
        # kernel_small = torch.ones(1, 3).to(self.device)
        # result = closing(result, kernel_small)

        return ImageTensor(result).to(torch.bool)

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
