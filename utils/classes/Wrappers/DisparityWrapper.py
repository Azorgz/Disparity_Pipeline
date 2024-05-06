import kornia
import torch
from kornia.filters import MedianBlur
from kornia.geometry import normalize_pixel_coordinates
from kornia.morphology import closing, dilation
from torch import Tensor
import torch.nn.functional as F
from utils.classes import DepthTensor, ImageTensor, CameraSetup
from utils.image_processing_tools import project_grid_to_image, projector


class DisparityWrapper:

    def __init__(self, device, setup: CameraSetup):
        self.device = device
        self.setup = setup

    def __call__(self, images: dict, depth_tensor: dict, cam_src: str, cam_dst: str, *args,
                 return_occlusion=True, post_process_image=3,
                 post_process_depth=3, return_depth_reg=False, reverse_wrap=False, **kwargs) -> (
            ImageTensor, DepthTensor):

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
        res = {}
        if not reverse_wrap:
            # selection of the right stereo setup and src image
            setup = self.setup.stereo_pair(cam_src, cam_dst)
            image_src = images[cam_src]
            sample = {cam_src: image_src}

            # rectification of the src image into the rectified frame using the appropriate homography
            img_src_proj = list(setup(sample).values())[0]

            # Transformation of the depth into signed Disparity
            disparity = setup.depth_to_disparity(depth_tensor[cam_dst])
            kernel = torch.ones(3, 3).to(self.device)
            disparity = dilation(disparity, kernel)
            disparity.scaled = True
            sign = -1 if setup.left.name == cam_src else 1
            disparity_dst = sign * disparity

            # Rectification of the signed Disparity into the rectified frame using the appropriate homography
            sample = {cam_dst: disparity_dst}
            side, disparity_proj = list(setup(sample).items())[0]
            opp_side = 'left' if side == 'right' else 'right'

            # resampling with disparity
            img_dst = self._grid_sample(img_src_proj, disparity_proj.clone(), padding_mode='zeros')

            if return_occlusion:
                # sample = {side: projector(disparity_proj, img_dst)}
                sample = {side: self.find_occlusion(disparity_proj, img_dst)}
                res['occlusion'] = setup(sample, reverse=True)[cam_dst].to(torch.bool)
                res['occlusion'].im_name = image_src.im_name + '_occlusion'
            if return_depth_reg:
                disparity_src = self.compute_disp_src(disparity_proj, post_process_depth=post_process_depth)
                disparity_src.pass_attr(disparity)
                disparity_src.im_name = image_src.im_name + '_disp'
                sample = {opp_side: disparity_src}
                disparity_src = setup(sample, reverse=True)[cam_src]
                res['depth_reg'] = setup.disparity_to_depth({cam_src: disparity_src})[cam_src]
            sample = {side: img_dst}
            res = {'image_reg': setup(sample, reverse=True)[cam_dst]}
            return res
        else:
            return self._reverse_call(images, depth_tensor, cam_src, cam_dst, *args,
                                      return_occlusion=return_occlusion,
                                      post_process_image=post_process_image,
                                      post_process_depth=post_process_depth,
                                      return_depth_reg=return_depth_reg, **kwargs)

    def _reverse_call(self, images: dict, depth_tensor: dict, cam_src: str, cam_dst: str, *args,
                      return_occlusion=True,
                      post_process_image=3,
                      post_process_depth=3,
                      return_depth_reg=False, **kwargs) -> (ImageTensor, DepthTensor):

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
        # selection of the right stereo setup and src image
        setup = self.setup.stereo_pair(cam_src, cam_dst)
        image_src = images[cam_src]
        sample = {cam_src: image_src}

        # rectification of the src image into the rectified frame using the appropriate homography
        img_src_proj = list(setup(sample).values())[0]

        # Transformation of the depth into signed Disparity
        disparity = setup.depth_to_disparity(depth_tensor[cam_src])
        kernel = torch.ones(3, 3).to(self.device)
        disparity = dilation(disparity, kernel)
        sign = 1 if setup.left.name == cam_src else -1
        disparity_src = sign * disparity

        # Rectification of the signed Disparity into the rectified frame using the appropriate homography
        sample = {cam_src: disparity_src}
        side, disparity_proj = list(setup(sample).items())[0]
        opp_side = 'left' if side == 'right' else 'right'

        # resampling with disparity
        size_im = disparity_proj.shape[-2:]
        grid = kornia.utils.create_meshgrid(size_im[0], size_im[1], normalized_coordinates=False,
                                            device=self.device).to(disparity_proj.dtype)  # [1 H W 2]

        grid[:, :, :, 0] -= disparity_proj[0, :, :, :]
        grid = torch.concatenate([grid, disparity_proj.permute([0, 2, 3, 1])], dim=-1)
        if return_occlusion:
            img_dst, occ = projector(grid, size_im, post_process_depth,
                                     image=img_src_proj,
                                     return_occlusion=return_occlusion,
                                     grid=True)
            sample = {opp_side: img_dst}
            res = {'image_reg': setup(sample, reverse=True)[cam_dst]}
            sample = {opp_side: occ * 1.}
            res['occlusion'] = setup(sample, reverse=True)[cam_dst].to(torch.bool)
            res['occlusion'].im_name = image_src.im_name + '_occlusion'
        else:
            img_dst = projector(grid, size_im, post_process_depth, image=img_src_proj, grid=True)
            sample = {opp_side: img_dst}
            res = {'image_reg': setup(sample, reverse=True)[cam_dst]}
        if return_depth_reg:
            disparity_dst = projector(grid, size_im, post_process_depth, grid=True)
            disparity_dst.im_name = images[cam_dst].im_name + '_disp'
            sample = {opp_side: disparity_dst}
            disparity_dst = setup(sample, reverse=True)[cam_dst]
            res['depth_reg'] = setup.disparity_to_depth({cam_dst: disparity_dst})[cam_dst]
        return res

    def compute_disp_src(self, disparity, post_process_depth=0, **kwargs):
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
        disparity_src = torch.ones([1, 1, h, w]).to(disparity.dtype).to(self.device) * (c.min())
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

    def find_occlusion(self, disparity, image, **kwargs):
        h, w = disparity.shape[-2:]
        mask = image == 0
        grid = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=self.device).to(
            disparity.dtype)  # [1 H W 2]
        grid[:, :, :, 0] -= disparity[0, :, :, :]
        M = torch.round(torch.abs(disparity).max() + 1)
        cloud = torch.concatenate([torch.round(grid[0, :, :, :]),
                                   disparity[0, :, :, :].permute(1, 2, 0)], dim=-1)
        # Put all the point into a H*W x 3 vector
        c = torch.tensor(cloud.flatten(start_dim=0, end_dim=1))  # H*W x 3
        # mask = c[:, 2] == 0
        # c[mask, :] = 0
        # create a unique index for each point according where they land in the src frame and their depth
        max_u = torch.round(cloud[:, :, 0].max() + 1)
        c_ = torch.round(c[:, 1]) * M * max_u + torch.round(c[:, 0]) * M - c[:, 2]
        # sort the point in order to find the ones landing in the same pixel
        C_, indexes = c_.sort(dim=0)

        # Define a new ordered vector but only using the position u
        c_ = torch.round(c[indexes, 0])
        c_disp = c[indexes, 2]

        # Trick to find the point landing in the same pixel, only the closer is removed
        c_[1:] -= c_.clone()[:-1]
        c_disp[1:] = 1 - c_disp.clone()[1:] / c_disp.clone()[:-1]

        idx = torch.nonzero((c_ == 0) * (c_disp > 0.03))
        idx = indexes[idx]

        # Use the indexes found to create a mask of the occluded point
        mask_occlusion = torch.zeros_like(c_).to(self.device)
        mask_occlusion[idx] = 1
        mask_occlusion = mask_occlusion.reshape([1, 1, cloud.shape[0], cloud.shape[1]])

        # postprocessing of the mask to remove the noise due to the round operation
        # blur = MedianBlur(5)
        # mask_occlusion = blur(mask_occlusion)
        # kernel = torch.ones(3, 3).to(self.device)
        # mask_occlusion = opening(mask_occlusion, kernel)
        # mask_occlusion = dilation(mask_occlusion, kernel)
        return ImageTensor(mask_occlusion + mask)

    def _grid_sample(self, image, disparity, padding_mode='zeros', **kwargs):
        h, w = disparity.shape[-2:]
        # mask = disparity[0, :, :, :] + torch.sum(image, dim=1) == 0
        grid = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=self.device).to(
            disparity.dtype)  # [1 H W 2]
        grid[:, :, :, 0] -= disparity[0, :, :, :]
        grid_norm: Tensor = normalize_pixel_coordinates(grid, h, w).to(image.dtype)  # BxHxWx2
        # grid_norm[:, :, :, 0][mask] = -2
        return F.grid_sample(image, grid_norm, padding_mode=padding_mode)
