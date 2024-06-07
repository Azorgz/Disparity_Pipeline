import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import MedianBlur
from kornia.morphology import dilation, closing
from torch import Tensor
from utils.classes import DepthTensor, ImageTensor


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
        b, cha, h, w = image.shape
        if upsample != 1:
            image = F.interpolate(Tensor(image), scale_factor=upsample).numpy()
        image_flatten = image.reshape([cha, w * h * upsample ** 2])  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    if upsample != 1:
        cloud = F.interpolate(Tensor(cloud).permute([0, 3, 1, 2]), scale_factor=upsample).permute([0, 2, 3, 1]).numpy()
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


def project_cloud_to_image_new(cloud, image_size, image=None, level=1):
    if image is not None:
        assert cloud.shape[1:3] == image.shape[-2:]
        b, cha, h, w = image.shape
        image_flatten = image.reshape([cha, w * h])  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    c = cloud.reshape((cloud.shape[0] * cloud.shape[1] * cloud.shape[2], 3))  # H*W x 3
    for i in range(level):
        im_size = image_size // (2 ** i)
        # c_ = F.interpolate(Tensor(cloud).permute([0, 3, 1, 2]), scale_factor=1/2**i).permute([0, 2, 3, 1]).numpy()
        c_ = c.copy()
        # Remove the point landing outside the image
        c_[:, 0] *= c_.shape[0] / im_size[1]
        c_[:, 1] *= c_.shape[1] / im_size[0]
        mask_out = ((c_[:, 0] < 0) + (c_[:, 0] >= (cloud.shape[2] - 1)) +
                    (c_[:, 1] < 0) + (c_[:, 1] >= (cloud.shape[1] - 1))) != 0
        # Sort the point by decreasing depth
        indexes = np.flip(c_[:, -1].argsort(0, 'quicksort'))
        c_[mask_out] = 0
        c_sorted = c_[indexes, :]
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
            occ = ImageTensor(F.interpolate(occ, image_size).to(torch.bool), device=result.device, permute_image=True)
        if image is None:
            res_ = dilation(res_, kernel)
        result[mask] = res_[mask]
    result = F.interpolate(result, image_size)
    if image is not None:
        result = ImageTensor(result, device=result.device, permute_image=True)
    else:
        result = DepthTensor(result, device=result.device, permute_image=True)
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
