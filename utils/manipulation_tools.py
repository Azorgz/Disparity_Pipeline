import time

import cv2 as cv
import numpy as np
import torch
from kornia import create_meshgrid
from kornia.utils import get_cuda_device_if_available
from torch import Tensor, FloatTensor
from kornia.feature.responses import harris_response
from utils.classes.Image import ImageTensor


def extract_roi_from_map(mask_left: Tensor, mask_right: Tensor):
    roi = []
    pts = []

    m_roi = [ImageTensor((mask_right+mask_left > 0) * torch.ones_like(mask_right)).pad([1, 1, 1, 1]),
             ImageTensor((mask_right*mask_left > 0) * torch.ones_like(mask_right)).pad([1, 1, 1, 1])]
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


def normalisation_tensor(image):
    m, M = image.min(), image.max()
    if m != M:
        return (image - m) / (M - m)
    else:
        return image


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

def create_meshgrid3d(depth: int, height: int, width: int, device: torch.device = None, type: torch.dtype = None)\
        -> ImageTensor:
    device = get_cuda_device_if_available() if device is None else device
    grid_2d = create_meshgrid(height, width, device=device).to(type).unsqueeze(1).repeat(1, depth, 1, 1, 1)  # 1xDxHxWx2
    vec = torch.arange(-1, 1+1/depth, 1/depth, device=device).to(type).unsqueeze(0)  # 1xD
    grid_z = (ImageTensor(torch.ones([height, width])).squeeze().unsqueeze(-1) @ vec).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)  # 1xDxHxWx1
    grid_3d = torch.cat([grid_2d, grid_z], dim=-1)

    return grid_3d
