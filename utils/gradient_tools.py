import numpy as np
import cv2 as cv
from kornia import pi
from kornia.color import hsv_to_rgb
from kornia.morphology import dilation

from utils.classes.Image import ImageCustom
from torchmetrics.functional import image_gradients
import torch

from utils.manipulation_tools import normalisation_tensor


def grad(image: ImageCustom or np.ndarray) -> ImageCustom:
    if len(image.shape) == 3:
        image = ImageCustom(image).GRAYSCALE()
    Ix = cv.Sobel(image, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT_101)
    Iy = cv.Sobel(image, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT_101)
    grad = np.sqrt(Ix ** 2 + Iy ** 2)
    grad[grad < grad.mean()] = 0
    grad[grad > grad.mean() * 5] = grad.mean() * 5
    orient = cv.phase(Ix, Iy, angleInDegrees=True)
    orient[grad == 0] = 0

    v = cv.normalize(grad, None, 0, 255, cv.NORM_MINMAX)
    s = np.ones_like(grad) * 255
    s[grad == 0] = 0
    h = cv.normalize(orient % 180, None, 0, 255, cv.NORM_MINMAX)
    h[grad == 0] = 0
    output = np.uint8(np.stack([h, s, v], axis=-1))
    output = cv.cvtColor(output, cv.COLOR_HSV2RGB)
    return ImageCustom(output)


def grad_tensor(image_tensor, device):
    _, c, _, _ = image_tensor.shape
    dy, dx = image_gradients(image_tensor)
    if c > 1:
        dx, dy = torch.sum(dx, dim=1) / 3, torch.sum(dy, dim=1) / 3
    grad_im = torch.sqrt(dx ** 2 + dy ** 2)
    m = torch.mean(grad_im)
    grad_im[grad_im < m] = 0
    grad_im[grad_im > m * 5] = m * 5
    # kernel = torch.ones(3, 3).to(device)
    # grad_im = dilation(grad_im.unsqueeze(0), kernel).squeeze(0)
    orient = torch.atan2(dy, dx)  # / np.pi * 180
    orient[grad_im == 0] = 0
    v = normalisation_tensor(grad_im)
    s = torch.ones_like(grad_im)
    s[grad_im == 0] = 0
    h = normalisation_tensor(orient % pi) * (2 * pi)
    h[grad_im == 0] = 0
    output = hsv_to_rgb(torch.stack((h, s, v), dim=1))

    return output
