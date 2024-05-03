import cv2 as cv
import numpy as np
import torch
from kornia import create_meshgrid
from kornia.utils import get_cuda_device_if_available
from torch import Tensor, FloatTensor, cat
from kornia.feature.responses import harris_response
from utils.classes.Image.Image import ImageTensor


def prepare_points_depth(depth):
    b, c, h, w = depth.shape
    grid = create_meshgrid(h, w, device=depth.device)
    grid3d = cat((grid, 1.0 / depth.permute(0, 2, 3, 1)), dim=-1)
    vec3d = grid3d.permute(3, 0, 1, 2).flatten(start_dim=1)
    return vec3d.permute(1, 0)


def extract_roi_from_map(mask_left: Tensor, mask_right: Tensor):
    roi = []
    pts = []

    m_roi = [ImageTensor((mask_right + mask_left > 0) * torch.ones_like(mask_right)).pad([1, 1, 1, 1]),
             ImageTensor((mask_right * mask_left > 0) * torch.ones_like(mask_right)).pad([1, 1, 1, 1])]
    m_transfo = [ImageTensor(mask_left).pad([1, 1, 1, 1]), ImageTensor(mask_right).pad([1, 1, 1, 1])]

    for m_ in m_roi:
        corner_map = Tensor(harris_response(m_)).squeeze()
        center = int(corner_map.shape[0] / 2), int(corner_map.shape[1] / 2)
        top_l = corner_map[:center[0], : center[1]]
        top_r = corner_map[:center[0], center[1]:]
        bot_l = corner_map[center[0]:, :center[1]]
        bot_r = corner_map[center[0]:, center[1]:]
        top_left = torch.argmax(top_l)
        top_left = top_left // center[1] - 1, top_left % center[1] - 1
        top_right = torch.argmax(top_r)
        top_right = top_right // center[1] - 1, top_right % center[1] + center[1] - 1
        bot_left = torch.argmax(bot_l)
        bot_left = bot_left // center[1] + center[0] - 1, bot_left % center[1] - 1
        bot_right = torch.argmax(bot_r)
        bot_right = bot_right // center[1] + center[0] - 1, bot_right % center[1] + center[1] - 1
        roi.append([int(max(top_left[0], top_right[0])), int(max(top_left[1], bot_left[1])),
                    int(min(bot_left[0], bot_right[0])), int(min(bot_right[1], top_right[1]))])

    for m_ in m_transfo:
        corner_map = Tensor(harris_response(m_)).squeeze()
        center = int(corner_map.shape[0] / 2), int(corner_map.shape[1] / 2)
        top_l = corner_map[:center[0], : center[1]]
        top_r = corner_map[:center[0], center[1]:]
        bot_l = corner_map[center[0]:, :center[1]]
        bot_r = corner_map[center[0]:, center[1]:]
        top_left = torch.argmax(top_l)
        top_left = top_left % center[1] - 1, top_left // center[1] - 1
        top_right = torch.argmax(top_r)
        top_right = top_right % center[1] + center[1] - 1, top_right // center[1] - 1,
        bot_left = torch.argmax(bot_l)
        bot_left = bot_left % center[1] - 1, bot_left // center[1] + center[0] - 1
        bot_right = torch.argmax(bot_r)
        bot_right = bot_right % center[1] + center[1] - 1, bot_right // center[1] + center[0] - 1
        pts.append([top_left, top_right, bot_left, bot_right])
    return roi[1], roi[0], FloatTensor(pts[0]), FloatTensor(pts[1])


def random_noise(mean, std, *args):
    if args is None:
        return
    else:
        noise = (np.random.random([len(args)]) - 0.5) * 2
        noise = noise / (noise ** 2).sum() * std
        noise -= noise.mean() + mean
    if len(args) == 1:
        args = float(args[0])
        noise = float(noise[0])
    return noise + args


def noise(mean, *args):
    if args is None:
        return
    else:
        if len(args) == 1:
            args = float(args[0])
    return mean + args


def list_to_dict(list_of_dict):
    res = {}
    for d in list_of_dict:
        if d.keys() == res.keys():
            for key in d.keys():
                res[key].append(d[key])
        else:
            for key in d.keys():
                res[key] = [d[key]]
    return res


def merge_dict(dict1: dict, dict2: dict, *args):
    if not dict1.keys() == dict2.keys():
        res = dict1 | dict2
        if args:
            res = merge_dict(res, *args)
    else:
        res = dict1.copy()
        for k in res.keys():
            if isinstance(res[k], dict) and isinstance(dict2[k], dict):
                res[k] = merge_dict(res[k], dict2[k])
            elif isinstance(res[k], list) and isinstance(dict2[k], list):
                if isinstance(res[k][0], list):
                    res[k] = [*res[k], dict2[k]]
                else:
                    res[k] = [res[k], dict2[k]]
                # for idx, (r1, r2) in enumerate(zip(res[k], dict2[k])):
                #     if isinstance(r1, dict) and isinstance(r2, dict):
                #         res[k][idx] = merge_dict(r1, r2)
                #     elif isinstance(r1, list) and isinstance(r2, float):
                #         res[k][idx] = [*r1, r2]
                #     elif isinstance(r1, float) and isinstance(r2, list):
                #         res[k][idx] = [r1, *r2]
                #     else:
                #         res[k][idx] = [r1, r2]
            elif (isinstance(res[k], list) and
                  (isinstance(dict2[k], float) or isinstance(dict2[k], int) or isinstance(dict2[k], str))):
                res[k] = [*res[k], dict2[k]]
            elif ((isinstance(res[k], float) or isinstance(res[k], int) or isinstance(res[k], str)) and
                  isinstance(dict2[k], list)):
                res[k] = [res[k], *dict2[k]]
            else:
                res[k] = [res[k], dict2[k]]
        if args:
            res = merge_dict(res, *args)
    return res


def flatten_dict(x):
    result = []
    if isinstance(x, dict):
        x = x.values()
    for el in x:
        if isinstance(el, dict) and not isinstance(el, str):
            result.extend(flatten_dict(el))
        else:
            result.append(el)
    return result


def map_dict_level(d: dict, level=0, map_of_dict=[], map_of_keys=[]):
    if len(map_of_dict) <= level:
        map_of_dict.append([len(d)])
        map_of_keys.append(list(d.keys()))
    else:
        map_of_dict[level].append(len(d))
    for idx, res in d.items():
        if isinstance(res, dict):
            map_of_dict, map_of_keys = map_dict_level(res, level + 1, map_of_dict, map_of_keys)
        else:
            return map_of_dict, map_of_keys
    if level == 0:
        map_of_dict.pop(0)
    return map_of_dict, map_of_keys


def drawlines(img, lines, pts):
    ''' img1 - image on which we draw the epilines for the points in img2
 lines - corresponding epilines '''
    r, c = img.shape[:2]
    for r, pt in zip(lines, pts.squeeze().cpu().numpy()):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
    return img


def create_meshgrid3d(depth: int, height: int, width: int, device: torch.device = None, type: torch.dtype = None) \
        -> ImageTensor:
    device = get_cuda_device_if_available() if device is None else device
    grid_2d = create_meshgrid(height, width, device=device).to(type).unsqueeze(1).repeat(1, depth, 1, 1, 1)  # 1xDxHxWx2
    vec = torch.arange(-1, 1 + 1 / depth, 1 / depth, device=device).to(type).unsqueeze(0)  # 1xD
    grid_z = (ImageTensor(torch.ones([height, width])).squeeze().unsqueeze(-1) @ vec).permute(2, 0, 1).unsqueeze(
        0).unsqueeze(-1)  # 1xDxHxWx1
    grid_3d = torch.cat([grid_2d, grid_z], dim=-1)

    return grid_3d
