import kornia
import torch
from kornia.filters import median_blur, MedianBlur
from kornia.geometry import normalize_pixel_coordinates
from kornia.morphology import closing, dilation, opening, erosion
from torch import Tensor
import torch.nn.functional as F

from utils.classes import ImageTensor
from utils.classes.Image import DepthTensor
from module.SetupCameras import CameraSetup


class DisparityWrapper:

    def __init__(self, device, setup: CameraSetup):
        self.device = device
        self.setup = setup

    def __call__(self, images_src: dict, depth_tensor: dict, cam_src: str, cam_dst: str, *args,
                 return_occlusion=True, post_process_image=3,
                 post_process_depth=3, return_depth_reg=False, **kwargs) -> (ImageTensor, DepthTensor):

        """Warp a tensor from a source to destination frame by the disparity in the destination.

        Rectify both image in a common frame where the images are parallel and then use the disparity
        of the destination to slide the source image to the destination image. Then it returns
        the transformed image into the destination frame.

        Args:
            image_src: image tensor in the source frame with shape :math:`(B,D,H,W)`.
            depth_dst: depth tensor in the destination frame with shape :math:`(B,1,H,W)`.
            cam_src: name of the src cam
            cam_dst: name of the dst cam
            method_3d: whether to use the regular grid_sample or the discrete 3d version

        Return:
            the warped tensor in the source frame with shape :math:`(B,3,H,W)`.
        """
        grid_sample = self._grid_sample
        # If we use the depth information from several cameras
        cams = [*args]

        # selection of the right stereo setup and src image
        setup = self.setup.stereo_pair(cam_src, cam_dst)
        image_src = images_src[cam_src]
        sample = {cam_src: image_src}

        # rectification of the src image into the rectified frame using the appropriate homography
        img_src_proj = list(setup(sample).values())[0]

        # Transformation of the depth into signed Disparity
        disparity = setup.depth_to_disparity(depth_tensor[cam_dst])
        sign = -1 if setup.left.name == cam_src else 1
        disparity_dst = sign * disparity

        # Rectification of the signed Disparity into the rectified frame using the appropriate homography
        sample = {cam_dst: disparity_dst}
        side, disparity_proj = list(setup(sample).items())[0]
        opp_side = 'left' if side == 'right' else 'right'

        # resampling with disparity
        img_dst = grid_sample(img_src_proj, disparity_proj.clone(), padding_mode='zeros')

        # mask = img_dst == 0
        # for cam_dst_bis in cams:
        #     setup_bis = self.setup.stereo_pair(cam_src, cam_dst_bis)
        #     # Transformation of the depth into signed Disparity
        #     disparity_bis = setup.depth_to_disparity(depth_tensor[cam_dst_bis])
        #     sign_bis = 1 if setup_bis.left.name == cam_src else -1
        #     disparity_dst_bis = sign_bis * disparity_bis
        #     # Rectification of the signed Disparity
        #     sample = {cam_dst_bis: disparity_dst_bis}
        #     side_bis, disparity_proj_bis = list(setup_bis(sample).items())[0]
        #     # Rectification of the signed Disparity
        #     sample = {cam_dst_bis: disparity_dst}
        #     disparity_dst_bis = list(setup(sample).values())[0]
        #     mask = disparity_proj[0, :, :, :] == 0
        #
        #     # grid for resampling
        #     h, w = disparity_proj_bis.shape[-2:]
        #     grid_bis = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=self.device).to(
        #         disparity_proj_bis.dtype)  # [1 H W 2]
        #     grid_bis[:, :, :, 0] -= disparity_proj_bis[0, :, :, :]
        #     grid_bis_norm: Tensor = normalize_pixel_coordinates(grid, h, w).to(image_src.dtype)  # BxHxWx2
        #     grid_bis_norm[:, :, :, 0][mask] = -2
        #     disparity_dst_bis = F.grid_sample(disparity_dst_bis, grid_bis_norm, padding_mode='zeros')
        #
        #     # side_bis = 'left' if side_bis == 'right' else 'right'
        #     # sample = {side_bis: grid_sample(disparity_proj_bis * factor, -disparity_proj_bis, **kwargs)}
        #     sample = {cam_src: list(setup_bis(sample, reverse=True, scale=False).values())[0]}
        #     disparity_proj_bis = list(setup(sample).values())[0]
        #     disparity_src[mask] = disparity_proj_bis[mask]

        sample = {side: img_dst}
        res = {'image_reg': setup(sample, reverse=True)[cam_dst]}
        if return_occlusion:
            res['occlusion'] = self.find_occlusion(disparity_proj.clone())
            res['occlusion'].im_name = image_src.im_name + '_occlusion'
        if return_depth_reg:
            disparity_src = self.compute_disp_src(disparity_proj, post_process_depth=post_process_depth)
            disparity_src.pass_attr(disparity)
            disparity_src.im_name = image_src.im_name + '_disp'
            sample = {opp_side: disparity_src}
            disparity_src = setup(sample, reverse=True)[cam_src]
            mask_occlusion = torch.abs(disparity_src) < 1
            disparity_src[mask_occlusion] = 0
            res['depth_reg'] = setup.disparity_to_depth({cam_src: disparity_src})[cam_src]
        return res

    def compute_disp_src(self, disparity, post_process_depth=3, **kwargs):
        h, w = disparity.shape[-2:]
        grid = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=self.device).to(
            disparity.dtype)  # [1 H W 2]
        grid[:, :, :, 0] -= disparity[0, :, :, :]

        # Put all the point into a H*W x 3 vector
        c = torch.tensor(disparity.flatten())  # H*W x 1

        # sort the point in order to find the ones landing in the same pixel
        _, indexes = c.sort()
        c_ = torch.tensor(grid.flatten(start_dim=0, end_dim=2))  # H*W x 2

        # Define a new ordered vector but only using the position u,v not the disparity
        c_ = torch.round(c_[indexes, :]).to(torch.int)

        # Create a picture with for pixel value the depth of the point landing in
        disparity_src = torch.zeros([1, 1, h, w]).to(disparity.dtype).to(self.device)  # *(c[:, 2].max()+1)
        disparity_src[0, 0, c_[:, 1], c_[:, 0]] = c[indexes]

        # postprocessing of the mask to remove the noise due to the round operation
        mask = disparity_src == 0
        blur = MedianBlur(3)
        res = blur(disparity_src)
        if post_process_depth:
            kernel = torch.ones(post_process_depth, post_process_depth).to(self.device)
            res = closing(disparity_src, kernel)
        disparity_src[mask] = res[mask]
        return DepthTensor(disparity_src, device=self.device).scale()

    def find_occlusion(self, disparity, **kwargs):
        h, w = disparity.shape[-2:]
        mask = disparity == 0
        grid = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=self.device).to(
            disparity.dtype)  # [1 H W 2]
        grid[:, :, :, 0] -= disparity[0, :, :, :]
        cloud = torch.concatenate([grid[0, :, :, :], torch.abs(disparity[0, :, :, :]).permute(1, 2, 0)], dim=-1)

        # Put all the point into a H*W x 3 vector
        c = torch.tensor(cloud.flatten(start_dim=0, end_dim=1))  # H*W x 3

        # create a unique index for each point according where they land in the src frame and their depth
        M = torch.round(c[:, 2].max() + 1)
        max_u = torch.round(c[:, 0].max())
        c_ = torch.round(c[:, 1]) * M * max_u + torch.round(c[:, 0]) * M - torch.round(c[:, 2])

        # sort the point in order to find the ones landing in the same pixel
        _, indexes = c_.sort(dim=0)
        c_ = c[:, :2].clone()

        # Define a new ordered vector but only using the position u,v not the disparity
        c_ = torch.round(c_[indexes, 1]) * max_u + torch.round(c_[indexes, 0])

        # Trick to find the point landing in the same pixel, only the closer is removed
        c_[1:] -= c_[:-1].clone()
        idx = torch.nonzero(c_ == 0)

        # Use the indexes found to create a mask of the occluded point
        idx = indexes[idx]
        mask_occlusion = torch.zeros([c.shape[0], 1]).to(self.device)
        mask_occlusion[idx] = 1
        mask_occlusion = mask_occlusion.reshape([1, 1, cloud.shape[0], cloud.shape[1]])

        # postprocessing of the mask to remove the noise due to the round operation
        # blur = MedianBlur(5)
        # mask_occlusion = blur(mask_occlusion)
        # kernel = torch.ones(3, 3).to(self.device)
        # mask_occlusion = erosion(mask_occlusion, kernel)
        return ImageTensor(mask + mask_occlusion).to(torch.bool)

    def _grid_sample(self, image, disparity, padding_mode='zeros', **kwargs):
        h, w = disparity.shape[-2:]
        mask = disparity[0, :, :, :] + torch.sum(image, dim=1) == 0
        grid = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=self.device).to(
            disparity.dtype)  # [1 H W 2]
        grid[:, :, :, 0] -= disparity[0, :, :, :]
        grid_norm: Tensor = normalize_pixel_coordinates(grid, h, w).to(image.dtype)  # BxHxWx2
        grid_norm[:, :, :, 0][mask] = -2
        return F.grid_sample(image, grid_norm, padding_mode=padding_mode)

    # def _grid_sample_with_occlusion(self, image, disparity, padding_mode='zeros', **kwargs):
    #
    #     res = F.grid_sample(image, grid, align_corners=True)
    #     res[:, :, mask_occlusion[0, 0, :, :]] = 0
    #     return res

        # grid[:, :, :, 0] -= disparity[0, :, :, :]

    # def _grid_sample3d(self, image, disp, step=0, post_process_depth=False, post_process_image=False, **kwargs):
    #     if isinstance(image, DepthTensor) and post_process_depth:
    #         post_process = True
    #     elif isinstance(image, ImageTensor) and post_process_image:
    #         post_process = True
    #     else:
    #         post_process = False
    #     # Build the volume according the disparity
    #     sign = -1 if disp.max() <= 0 else 1
    #
    #     if step == 0:
    #         disp_ = torch.round(disp).squeeze()
    #         max_val = (abs(disp_).max() + 1) * sign
    #         vec = torch.arange(0, max_val, step=sign, device=self.device).unsqueeze(0)
    #     else:
    #         max_val = (abs(torch.round(disp)).max() + 1) * sign
    #         disp_ = (torch.round(disp.normalize() * step) * max_val / step).squeeze()
    #         vec = torch.arange(0, 1, step=1 / (step + 1), device=self.device).unsqueeze(0) * max_val * (step + 1) / step
    #     vol_depth = torch.ones_like(disp_, device=self.device).unsqueeze(-1) @ vec  # HxWxD
    #     vol_mask = (torch.round(vol_depth - disp_.unsqueeze(-1) @ torch.ones_like(vec)) == 0) \
    #         .permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # Volume mask to localize the voxel
    #     vol_image = image.unsqueeze(2).repeat(1, 1, len(vec), 1,
    #                                           1) * vol_mask  # Image in volume with pixel at z = disparity
    #     # vol_disp = torch.ones_like(vol_mask, device=self.device) * disp.unsqueeze(2).repeat(1, 1, len(vec), 1, 1)
    #     # disparity repeated to form a volume
    #     d, h, w = vol_image.shape[-3:]
    #     grid_reg = kornia.utils.create_meshgrid(h, w, device=self.device).to(image.dtype)
    #
    #     res = torch.zeros_like(vol_image)
    #     for idx in range(d - 1):
    #         # grid_reg_ = grid_reg.clone()
    #         grid_reg[0, :, :, 0] -= (vec[0, idx + 1] - vec[0, idx]) / w * 2
    #         res[:, :, -(idx + 1), :, :] = F.grid_sample(vol_image[:, :, idx + 1, :, :], grid_reg)
    #     sign_im = 1 if image.max() > 0 else -1
    #     res = abs(res).max(2).values * sign_im
    #
    #     if res.__class__ != image.__class__:
    #         res = image.__class__(res)
    #         res.pass_attr(image)
    #     # if isinstance(res, DepthTensor):
    #     #     res = res.normalize().scale()
    #
    #     # Postprocessing for smoothness
    #     if post_process:
    #
    #         if isinstance(res, DepthTensor):
    #             blur = MedianBlur(post_process_depth)
    #             res_ = res.clone()
    #             kernel = torch.ones(5, 5).to(self.device)
    #             res_ = dilation(res_, kernel)
    #             for i in range(3):
    #                 res_ = blur(res_)
    #             res = res_
    #         elif isinstance(res, ImageTensor):
    #             blur = MedianBlur(post_process_image)
    #             res_ = res.clone()
    #             mask = abs(res) <= 0.1
    #             for i in range(3):
    #                 res_ = blur(res_)
    #             res[mask] = res_[mask]
    #         else:
    #             pass
    #
    #     return res
