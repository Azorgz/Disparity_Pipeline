import cv2 as cv
import kornia
import numba
import numpy as np
import torch
from kornia.filters import MedianBlur
from kornia.morphology import dilation, erosion, closing
import torch.nn.functional as F
from torch import Tensor

from utils.classes.Image.Image import DepthTensor, ImageTensor
from numba import njit, jit
import numba.types as nb_t


# from classes.Image import ImageCustom

def projector(cloud, image_size, post_process, image=None, return_occlusion=False, upsample=1 / 2, numpy=False,
              grid=False):
    if not grid:
        if not numpy:
            res = project_cloud_to_image(cloud, image_size, image, upsample)
        else:
            cloud_np = cloud.cpu().numpy()
            if image is not None:
                image_np = image.detach().cpu().numpy()
            else:
                image_np = None
            res = project_cloud_to_image_np(cloud_np, image_size, image_np, upsample)
            res = Tensor(res).to(cloud.device)
    else:
        res = project_grid_to_image(cloud, image_size, image)
    return post_process_proj(res, post_process, return_occlusion, image, image_size)


def project_cloud_to_image(cloud, image_size, image=None, upsample=1):
    if image is not None:
        assert cloud.shape[1:3] == image.shape[-2:]
        _, cha, _, _ = image.shape
        if upsample > 1:
            image = F.interpolate(image, scale_factor=upsample)
        image_flatten = torch.tensor(image.flatten(start_dim=2), dtype=cloud.dtype).squeeze(0)  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    if upsample > 1:
        cloud = F.interpolate(cloud.permute([0, 3, 1, 2]), scale_factor=upsample).permute([0, 2, 3, 1])
    c = torch.tensor(cloud.flatten(start_dim=0, end_dim=2))  # H*W x 3
    # Remove the point landing outside the image
    c[:, 0] *= cloud.shape[2] / image_size[1]
    c[:, 1] *= cloud.shape[1] / image_size[0]
    mask_out = ((c[:, 0] < 0) + (c[:, 0] >= (cloud.shape[2] - 1)) + (c[:, 1] < 0) + (
            c[:, 1] >= (cloud.shape[1] - 1))) != 0
    # Sort the point by decreasing depth
    _, indexes = c[:, -1].sort(descending=True, dim=0)
    c[mask_out] = 0
    c_sorted = c[indexes, :]
    if image is not None:
        sample = image_flatten[:, indexes]
    else:
        sample = c_sorted[:, 2:].permute([1, 0])
    # Transform the landing positions in accurate pixels
    c_x = torch.floor(c_sorted[:, 0:1]).to(torch.int)
    dist_c_x = torch.abs(c_x - c_sorted[:, 0:1])
    c_X = torch.ceil(c_sorted[:, 0:1]).to(torch.int)
    dist_c_X = torch.abs(c_X - c_sorted[:, 0:1])
    c_y = torch.floor(c_sorted[:, 1:2]).to(torch.int)
    dist_c_y = torch.abs(c_y - c_sorted[:, 1:2])
    c_Y = torch.floor(c_sorted[:, 1:2]).to(torch.int)
    dist_c_Y = torch.abs(c_Y - c_sorted[:, 1:2])

    rays = [torch.concatenate([c_x, c_y], dim=1),
            torch.concatenate([c_X, c_y], dim=1),
            torch.concatenate([c_x, c_Y], dim=1),
            torch.concatenate([c_X, c_Y], dim=1)]
    dists = [2 - torch.sqrt(dist_c_x ** 2 + dist_c_y ** 2),
             2 - torch.sqrt(dist_c_X ** 2 + dist_c_y ** 2),
             2 - torch.sqrt(dist_c_x ** 2 + dist_c_Y ** 2),
             2 - torch.sqrt(dist_c_X ** 2 + dist_c_Y ** 2)]
    if image is not None:
        result = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
    else:
        result = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
    total_dist = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
    for dist, ray in zip(dists, rays):
        temp = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
        temp_dist = torch.zeros([1, 1, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
        temp[0, :, ray[:, 1], ray[:, 0]] = sample
        temp_dist[0, 0, ray[:, 1], ray[:, 0]] = 1 / dist[:, 0]
        result += temp * temp_dist
        total_dist += temp_dist
    result[total_dist != 0] /= total_dist[total_dist != 0]
    return result


def project_cloud_to_image_np(cloud, image_size, image=None, upsample=1 / 2):
    if image is not None:
        assert cloud.shape[1:3] == image.shape[-2:]
        b, cha, w, h = image.shape
        if upsample > 1:
            w, h = w * upsample, h * upsample
            image = np.resize(image, [b, cha, w, h])
        image_flatten = image.reshape([cha, w * h])  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    if upsample != 1:
        cloud = F.interpolate(Tensor(cloud).permute([0, 3, 1, 2]), scale_factor=upsample).permute([0, 3, 2, 1]).numpy()
    c = cloud.reshape((cloud.shape[0] * cloud.shape[1] * cloud.shape[2], 3))  # H*W x 3
    # Remove the point landing outside the image
    c[:, 0] *= cloud.shape[2] / image_size[1]
    c[:, 1] *= cloud.shape[1] / image_size[0]
    mask_out = ((c[:, 0] < 0) + (c[:, 0] >= (cloud.shape[2] - 1)) + (c[:, 1] < 0) + (
            c[:, 1] >= (cloud.shape[1] - 1))) != 0
    # Sort the point by decreasing depth
    indexes = np.flip(c[:, -1].argsort(0, 'quicksort'))
    c[mask_out] = 0
    c_sorted = c[indexes, :]
    if image is not None:
        sample = image_flatten[:, indexes].transpose([1, 0])
    else:
        sample = c_sorted[:, 2:]
    # Transform the landing positions in accurate pixels
    c_x = np.int32(np.floor(c_sorted[:, 0:1]))
    dist_c_x = np.abs(c_x - c_sorted[:, 0:1])
    c_X = np.int32(np.ceil(c_sorted[:, 0:1]))
    dist_c_X = np.abs(c_X - c_sorted[:, 0:1])
    c_y = np.int32(np.floor(c_sorted[:, 1:2]))
    dist_c_y = np.abs(c_y - c_sorted[:, 1:2])
    c_Y = np.int32(np.floor(c_sorted[:, 1:2]))
    dist_c_Y = np.abs(c_Y - c_sorted[:, 1:2])

    rays = [np.concatenate([c_x, c_y], axis=1),
            np.concatenate([c_X, c_y], axis=1),
            np.concatenate([c_x, c_Y], axis=1),
            np.concatenate([c_X, c_Y], axis=1)]
    dists = [2 - np.sqrt(dist_c_x ** 2 + dist_c_y ** 2),
             2 - np.sqrt(dist_c_X ** 2 + dist_c_y ** 2),
             2 - np.sqrt(dist_c_x ** 2 + dist_c_Y ** 2),
             2 - np.sqrt(dist_c_X ** 2 + dist_c_Y ** 2)]
    if image is not None:
        result = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
    else:
        result = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
    total_dist = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
    for dist, ray in zip(dists, rays):
        temp = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        temp_dist = np.zeros([1, 1, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        temp[0, :, ray[:, 1].squeeze(), ray[:, 0].squeeze()] = sample
        temp_dist[0, 0, ray[:, 1].squeeze(), ray[:, 0].squeeze()] = 1 / dist[:, 0]
        result += temp * temp_dist
        total_dist += temp_dist
    result[total_dist != 0] /= total_dist[total_dist != 0]
    return result


def post_process_proj(result, post_process, return_occlusion, image, image_size):
    mask = result == 0
    if return_occlusion:
        occ = mask * 1.
    if post_process:
        blur = MedianBlur(post_process)
        kernel = torch.ones([5, 3], device=result.device if isinstance(result, Tensor) else 'cpu')
        res_ = result.clone() if isinstance(result, Tensor) else result.copy()
        res_ = blur(res_)
        if return_occlusion:
            occ = closing(occ, kernel)
            occ = blur(occ)
            occ = ImageTensor(F.interpolate(occ, image_size), device=result.device).to(torch.bool)
        if image is None:
            res_ = dilation(res_, kernel)
        result[mask] = res_[mask]
    result = F.interpolate(result, image_size)
    if image is not None:
        result = ImageTensor(result, device=result.device)
    else:
        result = DepthTensor(result, device=result.device).scale()
    if return_occlusion:
        return result, occ
    else:
        return result


def project_grid_to_image(grid, image_size, image=None):
    if image is not None:
        assert grid.shape[1:3] == image.shape[-2:]
        _, cha, _, _ = image.shape
        image_flatten = torch.tensor(image.flatten(start_dim=2), dtype=grid.dtype).squeeze(0)  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 2 vector
    c = torch.abs(grid.flatten(start_dim=0, end_dim=2))  # H*W x 3
    # Remove the point outside the dst frame
    c[:, 0] *= grid.shape[2] / image_size[1]
    mask_out = (c >= image_size[1] - 1) + (c < 0)
    c[mask_out] = 0
    # Sort the point by increasing disp
    _, indexes = torch.abs(c[:, -1]).sort(descending=False, dim=0)
    # c[mask_out] = 0
    c_sorted = c[indexes, :]
    if image is not None:
        sample = image_flatten[:, indexes]
    else:
        sample = c_sorted[:, 2:].permute([1, 0])
    # Transform the landing positions in accurate pixels
    c_x = torch.floor(c_sorted[:, 0:1]).to(torch.int)
    dist_c_x = torch.abs(c_x - c_sorted[:, 0:1])
    c_X = torch.ceil(c_sorted[:, 0:1]).to(torch.int)
    dist_c_X = torch.abs(c_X - c_sorted[:, 0:1])
    c_y = torch.floor(c_sorted[:, 1:2]).to(torch.int)

    rays = [torch.concatenate([c_x, c_y], dim=1),
            torch.concatenate([c_X, c_y], dim=1)]
    dists = [1 - dist_c_x, 1 - dist_c_X]

    if image is not None:
        result = torch.zeros([1, cha, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
    else:
        result = torch.zeros([1, cha, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
    total_dist = torch.zeros([1, cha, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
    for dist, ray in zip(dists, rays):
        temp = torch.zeros([1, 1, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
        temp_dist = torch.zeros([1, 1, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
        temp[0, :, ray[:, 1], ray[:, 0]] = sample
        temp_dist[0, 0, ray[:, 1], ray[:, 0]] = 1 / dist[:, 0]
        result += temp * temp_dist
        total_dist += temp_dist
    result[total_dist != 0] /= total_dist[total_dist != 0]
    return result

    # mask = result == 0
    # if post_process:
    #     blur = MedianBlur(post_process)
    #     kernel = torch.ones([5, 3], device=grid.device)
    #     res_ = result.clone()
    #     if return_occlusion:
    #         occ = mask * 1.
    #     for i in range(1):
    #         res_ = blur(res_)
    #         if return_occlusion:
    #             occ = closing(occ, kernel)
    #             occ = blur(occ)
    #         if image is None:
    #             res_ = dilation(res_, kernel)
    #     result[mask] = res_[mask]
    # result = F.interpolate(result, image_size)
    # if image is not None:
    #     result = ImageTensor(result)
    # else:
    #     result = DepthTensor(torch.abs(result)).scale()
    # if return_occlusion:
    #     occ = ImageTensor(F.interpolate(occ, image_size))
    #     return result, occ
    # else:
    #     return result
#
#
# def simple_project_cloud_to_image(cloud, image_size, post_process, image=None, return_occlusion=False):
#     if image is not None:
#         assert cloud.shape[1:3] == image.shape[-2:]
#         _, cha, _, _ = image.shape
#         image_flatten = torch.tensor(image.flatten(start_dim=2), dtype=cloud.dtype).squeeze(0)  # shape c x H*W
#     else:
#         cha = 1
#     # Put all the point into a H*W x 3 vector
#     c = torch.tensor(cloud.flatten(start_dim=0, end_dim=2))  # H*W x 3
#     # Remove the point landing outside the image
#     c[:, 0] *= cloud.shape[2] / image_size[1]
#     c[:, 1] *= cloud.shape[1] / image_size[0]
#     mask_out = ((c[:, 0] < 0) + (c[:, 0] >= (cloud.shape[2] - 1)) + (c[:, 1] < 0) + (
#             c[:, 1] >= (cloud.shape[1] - 1))) != 0
#     # Sort the point by decreasing depth
#     _, indexes = c[:, -1].sort(descending=True, dim=0)
#     c[mask_out] = 0
#     c_sorted = c[indexes, :]
#     if image is not None:
#         sample = image_flatten[:, indexes]
#         # sample[..., mask_out] = 0
#     else:
#         sample = c_sorted[:, 2:].permute([1, 0])
#     # Transform the landing positions in accurate pixels
#     c_ = torch.round(c_sorted[:, :2]).to(torch.int)
#     result = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
#     result[0, :, c_[:, 1], c_[:, 0]] = sample
#     mask = result == 0
#     if post_process:
#         blur = MedianBlur(post_process)
#         kernel = torch.ones([5, 3], device=cloud.device)
#         res_ = result.clone()
#         if return_occlusion:
#             occ = mask * 1.
#         for i in range(2):
#             res_ = blur(res_)
#             if return_occlusion:
#                 occ = blur(occ)
#             if image is None:
#                 res_ = erosion(res_, kernel)
#         result[mask] = res_[mask]
#     result = F.interpolate(result, image_size)
#     if return_occlusion:
#         occ = F.interpolate(occ, image_size).to(torch.bool)
#         return result, occ
#     else:
#         return result
#
#
# def perspective2grid(matrix, shape, device):
#     height, width = shape[0], shape[1]
#     grid_reg = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device)  # [1 H W 2]
#     z = torch.ones_like(grid_reg[:, :, :, 0])
#     grid = torch.stack([grid_reg[..., 0], grid_reg[..., 1], z], dim=-1)  # [1 H W 3]
#
#     grid_transformed = (matrix @ grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [1 H W 3]
#     alpha = grid_transformed[:, :, :, 2]
#     grid_transformed[:, :, :, 0] = 2 * (grid_transformed[:, :, :, 0] / alpha) / width - 1  # [1 H W 3]
#     grid_transformed[:, :, :, 1] = 2 * (grid_transformed[:, :, :, 1] / alpha) / height - 1  # [1 H W 3]
#     return grid_transformed[..., :2]  # [1 H W 2]
# #
# #
# # def normalization_maps(image, image2=None):
# #     """
# #     Function used by the mask generator
# #     :param image: image to normalize
# #     :return: normalized and center image
# #     """
# #     # Normalization
# #     if image2 is None:
# #         M = image.max()
# #         mi = image.min()
# #         m = image.mean()
# #     else:
# #         M = max(image.max(), image2.max())
# #         mi = min(image.min(), image2.min())
# #         m = image.mean() / 2 + image2.mean() / 2
# #         im2 = 1 - abs(image2 / 1.0 - m) / M  # Compute the distance to the mean of the image
# #         im2 = (im2 - im2.min()) / (2 * (im2.max() - im2.min())) + 0.5  # Normalize this distance between 0.5 and 1
# #         res2 = image2 * im2  # Normalize the input image between 0 and 1 and weight it by the mask computed before
# #         res2 = (res2 - res2.min()) / (res2.max() - res2.min()) * 255  # Normalize the output between 0 and 255
# #     im = 1 - abs(image / 1.0 - m) / M
# #     im = (im - im.min()) / (2 * (im.max() - im.min())) + 0.5
# #     res = image * im
# #     res = (res - res.min()) / (res.max() - res.min()) * 255
# #     if image2 is None:
# #         return ImageCustom(res)
# #     else:
# #         return ImageCustom(res), ImageCustom(res2)
# #
# #
# # def scaled_fusion(pyramid, method_interval, method_scale, pyramid2=None, method_fusion=None, first_level_scaled=False):
# #     """
# #     :param pyramid: Dictionary shaped like a Gaussian pyramid
# #     :param method_interval: function used to fuse intra-interval images
# #     :param method_scale: function used to fuse inter-scale images
# #     :param pyramid2: Optionnal, to fuse two pyramid together at the interval level
# #     :param method_fusion: Mendatory if a second pyramid is specified, function to fuse images between the two pyramids
# #     :return:
# #     """
# #     new_pyr = {}
# #     for key in pyramid.keys():
# #         ref = pyramid[key][list(pyramid[key].keys())[0]]
# #         for inter, im in pyramid[key].items():
# #             if pyramid2 is not None:
# #                 im = method_fusion(pyramid[key][inter], pyramid2[key][inter])
# #             ref = method_interval(ref, im)
# #         new_pyr[key] = ref
# #     scale = np.array(list(new_pyr.keys()))
# #     scale_diff = scale[1:] - scale[:-1]
# #     if len(scale) == 1 or first_level_scaled:
# #         return new_pyr[scale[0]]
# #     for idx, key in enumerate(reversed(scale[1:])):
# #         temp = new_pyr[key]
# #         diff = scale_diff[-(1 + idx)]
# #         for i in range(diff):
# #             temp = cv.pyrUp(temp)
# #         im_large = new_pyr[key - diff]
# #         new_pyr[key - diff] = method_scale(im_large, temp)
# #     return ImageCustom(new_pyr[scale[0]])
# #
# #
# # def laplacian_fusion(pyramid, pyramid2, mask_fusion, verbose=False):
# #     """
# #     :param pyramid: Dictionary shaped like a Gaussian pyramid
# #     :param pyramid2: Optionnal, to fuse two pyramid together at the interval level
# #     :param mask_fusion: Mendatory to fuse images between the two pyramids
# #     :return:
# #     """
# #     new_pyr = {}
# #     new_pyr2 = {}
# #     masks = {}
# #     for key in pyramid.keys():
# #         if key == 0:
# #             new_pyr[key] = pyramid[key]
# #             new_pyr2[key] = pyramid2[key]
# #         else:
# #             new_pyr[key] = pyramid[key][1]
# #             new_pyr2[key] = pyramid2[key][1]
# #         masks[key] = cv.resize(mask_fusion, (new_pyr[key].shape[1], new_pyr[key].shape[0]))
# #     scale = np.array(list(new_pyr.keys())[1:])
# #     scale_diff = scale[1:] - scale[:-1]
# #     temp = None
# #     for idx, key in enumerate(reversed(scale)):
# #         temp1 = new_pyr[key]
# #         temp2 = new_pyr2[key]
# #         if idx < len(scale) - 1:
# #             diff = scale_diff[-(1 + idx)]
# #             for i in range(diff):
# #                 temp11 = cv.pyrUp(temp1)
# #                 temp22 = cv.pyrUp(temp2)
# #         else:
# #             temp11 = temp1
# #             temp22 = temp2
# #         detail1 = new_pyr[key - diff] / 255 - temp11 / 255
# #         detail2 = new_pyr2[key - diff] / 255 - temp22 / 255
# #         details = detail1 * masks[key - diff] + detail2 * (1 - masks[key - diff])
# #         if temp is None:
# #             temp = ImageCustom(temp11 * masks[key - diff] + ImageCustom(temp22 * (1 - masks[key - diff]))
# #                                + details * 255, pyramid[0])
# #         elif idx < len(scale) - 1:
# #             temp = ImageCustom(cv.pyrUp(temp) + details * 255, pyramid[0])
# #         else:
# #             temp = temp + details * 255
# #             temp = ImageCustom((temp - temp.min()) / (temp.max() - temp.min()) * 255, pyramid[0])
# #         if verbose:
# #             cv.imshow('fusion', temp)
# #             cv.waitKey(0)
# #             cv.destroyAllWindows()
# #     return temp.unpad()
# #
# #
# # def blobdetection(image):
# #     # Setup SimpleBlobDetector parameters.
# #     params = cv.SimpleBlobDetector_Params()
# #
# #     # Change thresholds
# #     params.minThreshold = 0
# #     params.maxThreshold = 255
# #
# #     # Filter by Area.
# #     params.filterByArea = True
# #     params.minArea = 10
# #     params.maxArea = 50
# #
# #     # Filter by Circularity
# #     params.filterByCircularity = True
# #     params.minCircularity = 0.8
# #     params.maxCircularity = 1
# #
# #     # Filter by Convexity
# #     params.filterByConvexity = False
# #     params.minConvexity = 0.87
# #
# #     # Filter by Inertia
# #     params.filterByInertia = False
# #     params.minInertiaRatio = 0.01
# #
# #     detector = cv.SimpleBlobDetector_create(params)
# #     if image.dtype == np.float64:
# #         im = np.uint8(image * 255)
# #     else:
# #         im = image.copy()
# #     keypoints = detector.detect(im)
# #     print(keypoints)
# #     im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
# #                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# #     return im_with_keypoints
