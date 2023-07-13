import warnings
import PIL.Image
import cv2 as cv
import numpy as np
import torch
from PIL.Image import Image as Im
from PIL import Image
from numpy import dtype
from skimage import io
from matplotlib import pyplot as plt, patches
from matplotlib import cm
from os.path import *
from pathlib import Path
from scipy.ndimage import median_filter
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import inspect
from types import FrameType
from typing import cast, Union


def find_class(args, class_name):
    arg = {}
    for a in args:
        if isinstance(a, class_name):
            return a
        elif isinstance(a, list) or isinstance(a, tuple):
            arg = find_class(a, class_name)
        else:
            arg = None
    return arg


def update_channel_pos(im):
    shape = np.array(im.shape)
    channel_pos = np.argwhere(shape == 3)
    channel_pos = channel_pos[0][0] if len(channel_pos >= 1) else None
    if channel_pos is None:
        return -1
    else:
        return int(channel_pos)


class ImageTensor(Tensor):
    """
    A class defining the general basic framework of a TensorImage.
    It can use all the methods from Torch plus some new ones.
    An instance is created using a numpy array or a path to an image file or a PIL image
    """
    _im_type = None
    _im_name = None
    _im_pad = torch.tensor([[0, 0, 0, 0]])
    _color_mode = None
    _mode_list = ['1', 'L', 'RGB', 'RGBA', 'CMYK', 'LAB', 'HSV']
    _channel_pos = None

    def __init__(self, *args, **kwargs):
        super(ImageTensor, self).__init__()

    @staticmethod
    def __new__(cls, inp, *args, name: str = 'new_image', device: torch.device = None, cmap: str = None, **kwargs):
        # Input array is a path to an image OR an already formed ndarray instance
        color_mode = None
        if isinstance(inp, str):
            name = basename(inp).split('.')[0] if name == 'new_image' else name
            inp_ = Image.open(inp)
        elif isinstance(inp, np.ndarray):
            inp_ = to_pil_image(inp)
        elif isinstance(inp, Tensor):
            inp_ = to_pil_image(inp.squeeze())
        elif isinstance(inp, PIL.Image.Image):
            inp_ = inp
        elif isinstance(inp, ImageTensor):
            inp_ = inp.clone()
            inp_ = inp_.squeeze()
            while len(inp_.shape) < 3:
                inp_ = torch.unsqueeze(inp_, 0)
        else:
            raise NotImplementedError
        if isinstance(inp_, PIL.Image.Image):
            t = transforms.ToTensor()
            color_mode = inp_.mode
            inp_ = t(inp_)
        else:
            raise NotImplementedError

        if isinstance(device, torch.device):
            image = inp_.to(device)
        else:
            if torch.cuda.is_available():
                image = inp_.to(torch.device('cuda'))
            else:
                image = inp_.to(torch.device('cpu'))
        image = super().__new__(cls, image)
        if isinstance(inp, ImageTensor):
            image.pass_attr(inp)
        image, im_type, cmap, channel_pos = image._attribute_init_()
        color_mode = color_mode if color_mode else cmap
        # add the new attributes to the created instance of Image
        image._channel_pos = channel_pos
        image._im_type = im_type
        image._color_mode = color_mode
        image._im_name = name
        return image

    # Base Methods
    def __add__(self, other):
        if isinstance(other, DepthTensor):
            other_ = other.RGB()
        else:
            other_ = other
        return torch.Tensor.__add__(self, other_)

    def _attribute_init_(self):
        temp = self.squeeze()
        channel_pos = update_channel_pos(temp)
        if len(temp.shape) == 2:
            im_type = 'IR'
            cmap = 'L'
        elif len(temp.shape) == 3:
            if channel_pos == -1:
                return self, 'unknown', 'unknown', 'unknown'
            if all(torch.equal(temp[i, :, :], temp[i + 1, :, :]) for i in range(temp.shape[channel_pos] - 1)):
                im_type = 'IR'
                cmap = 'L'
            else:
                im_type = 'RGB'
                cmap = 'RGB'
        else:
            return self, 'unknown', 'unknown', 'unknown'
        while len(temp.shape) < 4:
            temp = temp.unsqueeze(0)
            if isinstance(channel_pos, int):
                channel_pos += 1
        return temp, im_type, cmap, channel_pos

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if res.__class__ is Tensor:
            res = ImageTensor(res)
        if res.__class__ is ImageTensor:
            arg = find_class(args, ImageTensor)
            if arg is not None:
                res.pass_attr(arg)
                if res.shape != arg.shape:
                    res.channel_pos = abs(update_channel_pos(res))
            return res
        else:
            return res

    def pass_attr(self, image, *args):
        if len(args) > 0:
            for arg in args:
                self.__dict__[arg] = image.__dict__[arg]
        else:
            self.__dict__ = image.__dict__.copy()

    # Image manipulation methods
    def pad(self, im, **kwargs):
        '''
        Pad the image to match the given Tensor/Array size or with the list of padding indicated (left, right, top, bottom)
        :param im:
        :param kwargs:
        :return:
        '''
        if isinstance(im, ImageTensor):
            im_ = im.put_channel_at(-1)
            h, w = im_.shape[-3:-1]
        elif isinstance(im, Tensor) or isinstance(im, np.ndarray):
            h, w = im.shape[-2:]
        elif isinstance(im, list) or isinstance(im, tuple):
            assert len(im) == 2 or len(im) == 4
            if len(im) == 2:
                pad_l, pad_r = int(im[0]), int(im[0])
                pad_t, pad_b = int(im[1]), int(im[1])
            if len(im) == 4:
                pad_l, pad_r, pad_t, pad_b = int(im[0]), int(im[1]), int(im[2]), int(im[3])
            pad_tuple = (pad_l, pad_r, pad_t, pad_b)
            im_ = F.pad(self, pad_tuple, **kwargs)
            return im_
        else:
            h , w = 0, 0
        h_ref, w_ref = self.put_channel_at(-1).shape[-3:-1]
        try:
            assert w >= w_ref and h >= h_ref
        except AssertionError:
            return self.clone()
        pad_l = int((w - w_ref) // 2 + (w - w_ref) % 2)
        pad_r = int((w - w_ref) // 2 - (w - w_ref) % 2)
        pad_t = int((h - h_ref) // 2 + (h - h_ref) % 2)
        pad_b = int((h - h_ref) // 2 - (h - h_ref) % 2)
        pad_tuple = (pad_l, pad_r, pad_t, pad_b)
        im_ = F.pad(self, pad_tuple, **kwargs)
        return im_

    def put_channel_at(self, idx=1):
        return torch.movedim(self, self.channel_pos, idx)

    def match_shape(self, other):
        temp = self.put_channel_at()
        b = other.put_channel_at(-1)
        _, h, w, _ = b.shape
        temp = F.interpolate(temp, size=(h, w), mode='bilinear', align_corners=True)
        return temp.put_channel_at(self.channel_pos)

    def opencv(self):
        if self.color_mode == 'L':
            a = np.ascontiguousarray(Tensor.numpy(self.squeeze().cpu()) * 255, dtype=np.uint8)
        else:
            a = np.ascontiguousarray(self.RGB().put_channel_at(-1).squeeze().cpu().numpy()[..., [2, 1, 0]] * 255,
                                     dtype=np.uint8)
        return a

    def show(self, num=None, cmap='gray', roi: list = None, point: Union[list, Tensor] = None):
        im_display = [*self]
        if not num:
            num = self.im_name
        fig, ax = plt.subplots(ncols=len(im_display), num=num, squeeze=False)
        for i, img in enumerate(im_display):
            img = img.unsqueeze(0)
            img.pass_attr(self)
            if img.im_type == 'IR':
                img, cmap = img.put_channel_at(-1).squeeze(), cmap
            else:
                img, cmap = img.put_channel_at(-1).squeeze(), None
            if point is not None:
                for center in point.squeeze():
                    center = center.cpu().long().numpy()
                    img = ImageTensor(cv.circle(img.opencv(), center, 5, (0, 255, 0), -1)[..., [2, 1, 0]])
            ax[0, i].imshow(img.cpu().numpy(), cmap=cmap)
            if roi is not None:
                for r, color in zip(roi, ['r', 'g', 'b']):
                    rect = patches.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0]
                                             , linewidth=2, edgecolor=color, facecolor='none')
                    ax[0, i].add_patch(rect)

            ax[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        return ax

    @property
    def im_name(self) -> str:
        return self._im_name

    @im_name.setter
    def im_name(self, name) -> None:
        self._im_name = name

    @property
    def im_type(self) -> str:
        return self._im_type

    @im_type.setter
    def im_type(self, t) -> None:
        warnings.warn("The attribute can't be modified")

    @property
    def mode_list(self) -> list:
        return self._mode_list

    @property
    def channel_pos(self) -> int:
        return self._channel_pos

    @channel_pos.setter
    def channel_pos(self, pos) -> None:
        self._channel_pos = pos

    @channel_pos.deleter
    def channel_pos(self) -> None:
        warnings.warn("The attribute can't be deleted")

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @color_mode.setter
    def color_mode(self, v) -> None:
        """
        :param c_mode: str following the Modes of a Pillow Image
        :param colormap: to convert a GRAYSCALE image to a Palette (=colormap) colored image
        """
        if isinstance(v, list) or isinstance(v, tuple):
            c_mode = v[0]
            colormap = v[1]['colormap'] if v[1] else 'inferno'
        else:
            c_mode = v
            colormap = 'inferno'
        c_mode = c_mode.upper()
        assert c_mode in self.mode_list
        if c_mode == self.color_mode:
            pass
        elif self.color_mode == 'L':
            x = np.linspace(0.0, 1.0, 256)
            cmap_rgb = Tensor(cm.get_cmap(plt.get_cmap(colormap))(x)[:, :3]).to(self.device).squeeze()
            temp = (self * 255).long().squeeze()
            new = ImageTensor(cmap_rgb[temp].permute(2, 0, 1), color_mode='RGB')
            self.data = new.data
            self.pass_attr(new, '_color_mode', '_channel_pos')
        elif self.color_mode == '1':
            warnings.warn("The boolean image can't be colored")
            c_mode = self.color_mode
        if not c_mode == self.color_mode:
            temp = self.clone()
            while len(temp.shape) > 3:
                temp = temp.squeeze(0)
            im_pil = to_pil_image(temp, mode=self._color_mode)
            im_pil = im_pil.convert(c_mode)
            new = ImageTensor(im_pil)
            self.data = new.data
            self.pass_attr(new, '_color_mode', '_channel_pos')

    @color_mode.deleter
    def color_mode(self):
        warnings.warn("The attribute can't be deleted")

    def RGB(self, cmap='inferno'):
        """
        Implementation equivalent at the attribute setting : im.color_mode = 'rgb' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = 'RGB', {'colormap': cmap}
        return im

    def RGBA(self, cmap='inferno'):
        """
        Implementation equivalent at the attribute setting : im.color_mode = 'rgba' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = 'RGB', {'colormap': cmap}
        return im

    def GRAYSCALE(self):
        """
        Implementation equivalent at the attribute setting : im.color_mode = 'L' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = 'L', {}
        return im

    def CMYK(self, cmap='inferno'):
        """
        Implementation equivalent at the attribute setting : im.color_mode = 'cmyk' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = 'CMYK', {'colormap': cmap}
        return im

    def LAB(self, cmap='inferno'):
        """
        Implementation equivalent at the attribute setting : im.color_mode = 'lab' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = 'LAB', {'colormap': cmap}
        return im

    def HSV(self, cmap='inferno'):
        """
        Implementation equivalent at the attribute setting : im.color_mode = 'hsv' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = 'HSV', {'colormap': cmap}
        return im

    def BINARY(self):
        """
        Implementation equivalent at the attribute setting : im.color_mode = '1' but create a new ImageTensor
        """
        im = self.clone()
        im.color_mode = '1', {}
        return im

    # def save(self, path=None):
    #     if not path:
    #         path = join(Path(dirname(__file__)).parent.absolute(), 'output')
    #     path = join(path, self.name + ".jpg")
    #     io.imsave(path, self, plugin=None, check_contrast=True)
    #
    # def median_filter(self, size=3):
    #     return ImageCustom(median_filter(self, size), self)
    #     # im.current_value = np.asarray(self)
    #
    # def gaussian_filter(self, sigma=2.0):
    #     im = ImageCustom(cv.GaussianBlur(self, (0, 0), sigma), self)
    #     # im.current_value = np.asarray(self)
    #     return im
    #
    # def mean_shift(self, value=0.5):
    #     if value > 1:
    #         value = value / 255
    #     if self.dtype == np.uint8:
    #         i = self.copy() / 255
    #         return np.uint8(i ** (np.log(value) / np.log(i.mean())) * 255)
    #     else:
    #         i = self.copy()
    #         return i ** (np.log(value) / np.log(i.mean()))
    #
    # def saliency(self, radius=0, color=True):
    #     if color:
    #         i = self.copy()
    #     else:
    #         i = self.GRAYSCALE()
    #     if radius == 0:
    #         return ImageCustom(abs(i / 255 - (i / 255).mean()) * 255).GRAYSCALE()
    #     else:
    #         return ImageCustom(abs(i / 255 - i.median_filter(size=radius) / 255) * 255).GRAYSCALE()
    #
    # def padding_2n(self, level=3, pad_type='zeros'):
    #     '''
    #     :param level: integer, the number of time the image has to be downscalable without loss after padding
    #     :param pad_type: 'zeros' for constant zeros padding, 'reflect_101' for 'abcba' padding, 'replicate' for 'abccc' padding... See opencv doc
    #     :return: image dowscalable without loss
    #     '''
    #     assert isinstance(level, int), print("level has to be an integer")
    #     m, n = self.shape[:2]
    #     # if m % 2**level == 0 and n % 2**level == 0:
    #     #     return self
    #
    #     # Padding number for the height
    #     temp = m
    #     pad_v = 0
    #     l = 0
    #     while l < level:
    #         if temp % 2 != 0:
    #             pad_v += 1 * 2 ** l
    #             l += 1
    #             temp = (temp + 1) / 2
    #         else:
    #             temp = temp / 2
    #             l += 1
    #     pad_v = pad_v / 2
    #     # Padding number for the width
    #     temp = n
    #     pad_h = 0
    #     l = 0
    #     while l < level:
    #         if temp % 2 != 0:
    #             pad_h += 1 * 2 ** l
    #             l += 1
    #             temp = (temp + 1) / 2
    #         else:
    #             temp = temp / 2
    #             l += 1
    #     pad_h = pad_h / 2
    #
    #     l_pad = int(pad_h if pad_h % 1 == 0 else pad_h + 0.5)
    #     r_pad = int(pad_h if pad_h % 1 == 0 else pad_h - 0.5)
    #     t_pad = int(pad_v if pad_v % 1 == 0 else pad_v + 0.5)
    #     b_pad = int(pad_v if pad_v % 1 == 0 else pad_v - 0.5)
    #
    #     if pad_type == 'zeros':
    #         borderType = cv.BORDER_CONSTANT
    #         value = 0
    #     elif pad_type == 'reflect_101':
    #         borderType = cv.BORDER_REFLECT_101
    #         value = None
    #     elif pad_type == 'replicate':
    #         borderType = cv.BORDER_REPLICATE
    #         value = None
    #     elif pad_type == 'reflect':
    #         borderType = cv.BORDER_REFLECT
    #         value = None
    #     elif pad_type == 'reflect_101':
    #         borderType = cv.BORDER_REFLECT_101
    #         value = None
    #     im = ImageCustom(cv.copyMakeBorder(self, t_pad, b_pad, l_pad, r_pad, borderType, None, value=value), self)
    #     padding_final = np.array([t_pad, l_pad, b_pad, r_pad])
    #     im.pad = im.pad + padding_final
    #     return im
    #
    # def unpad(self):
    #     '''
    #     :return: Unpadded image
    #     '''
    #     t, l, b, r = self.pad
    #     if t != 0:
    #         self = self[t:, :]
    #     if l != 0:
    #         self = self[:, l:]
    #     if b != 0:
    #         self = self[:-b, :]
    #     if r != 0:
    #         self = self[:, :-r]
    #     self.pad = np.zeros_like(self.pad)
    #     return self
    #
    # def pyr_scale(self, octave=3, gauss=False, verbose=False):
    #     im = self.padding_2n(level=octave, pad_type='reflect_101')
    #     pyr_scale = {0: im}
    #     if verbose:
    #         print(f"level 0 shape : {pyr_scale[0].shape}")
    #     for lv in range(octave):
    #         pyr_scale[lv + 1] = ImageCustom(cv.pyrDown(pyr_scale[lv]), self)
    #         if gauss:
    #             pyr_scale[lv + 1] = ImageCustom(cv.GaussianBlur(pyr_scale[lv + 1], (5, 5), 0), pyr_scale[lv + 1])
    #         if verbose:
    #             print(f"level {lv + 1} shape : {pyr_scale[lv + 1].shape}")
    #     return pyr_scale
    #
    # def pyr_gauss(self, octave=3, interval=4, sigma0=1, verbose=False):
    #     k = 2 ** (1 / interval)
    #     im = self.padding_2n(level=octave, pad_type='reflect_101')
    #     pyr_gauss = {0: im}
    #     if verbose:
    #         print(f"level 0 shape : {pyr_gauss[0].shape}")
    #     for lv in range(octave):
    #         sigma = sigma0 * (2 ** lv)
    #         if lv != 0:
    #             pyr_gauss[lv + 1] = {0: ImageCustom(cv.pyrDown(pyr_gauss[lv][0]), im)}
    #         else:
    #             pyr_gauss[lv + 1] = {0: ImageCustom(pyr_gauss[lv], im)}
    #         for inter in range(interval):
    #             sigmaS = (k ** inter) * sigma
    #             pyr_gauss[lv + 1][inter + 1] = ImageCustom(cv.GaussianBlur(pyr_gauss[lv + 1][0], (0, 0), sigmaS), self)
    #         if verbose:
    #             print(f"level {lv + 1} shape : {pyr_gauss[lv + 1][0].shape}")
    #             cv.imshow('pyr_gauss', pyr_gauss[lv + 1][0].BGR())
    #             cv.waitKey(0)
    #     cv.destroyAllWindows()
    #     return pyr_gauss
    #
    # def expand_dims(self):
    #     im = self.copy()
    #     if len(self.shape) < 3:
    #         im = np.stack([im, im, im], axis=-1)
    #     return ImageCustom(im, self)
    #
    # def new_axis(self):
    #     im = self.copy()
    #     return np.atleast_3d(im)
    #
    # def match_shape(self, im2, keep_ratio=True, channel=False):
    #     im = self.copy()
    #     if im.shape == im2.shape:
    #         return im
    #     if (not (im.ndim == im2.ndim) or im.shape[-1] != im2.shape[-1]) and channel:
    #         if len(im2.shape) == 2:
    #             im = im[:, :, 0] / 3 + im[:, :, 1] / 3 + im[:, :, 2] / 3
    #         else:
    #             im = im.expand_dims()
    #     if not keep_ratio:
    #         im = cv.resize(im, [im2.shape[1], im2.shape[0]])
    #     else:
    #         h, w = im.shape[:2]
    #         h2, w2 = im2.shape[:2]
    #         ratio_h = h / h2
    #         ratio_w = w / w2
    #         if ratio_h == ratio_w:
    #             im = ImageCustom(cv.resize(im, [im2.shape[1], im2.shape[0]]))
    #         elif ratio_h < ratio_w:
    #             if im.dims == 3:
    #                 temp = np.zeros([h2, w2, im.shape[-1]])
    #             else:
    #                 temp = np.zeros([h2, w2])
    #             im = ImageCustom(cv.resize(im, [im2.shape[1], int(im.shape[0] / ratio_w)]))
    #             pad = im2.shape[0] - im.shape[0]
    #             if pad % 2 == 0:
    #                 pad = int(pad / 2)
    #                 temp[pad:-pad, :] = im
    #             else:
    #                 pad = int((pad + 1) / 2)
    #                 if pad > 1:
    #                     temp[pad:-(pad - 1), :] = im
    #                 else:
    #                     temp[pad:, :] = im
    #             im = temp.copy()
    #             del (temp)
    #         else:
    #             temp = np.zeros_like(im2)
    #             im = ImageCustom(cv.resize(im, [int(im.shape[1] / ratio_h), im2.shape[0]]))
    #             pad = im2.shape[1] - im.shape[1]
    #             if pad % 2 == 0:
    #                 pad = int(pad / 2)
    #                 temp[:, pad:-pad] = im
    #             else:
    #                 pad = int((pad + 1) / 2)
    #                 temp[:, pad:-(pad - 1)] = im
    #             im = temp.copy()
    #             del (temp)
    #     return ImageCustom(im, self)


class DepthTensor(ImageTensor):
    """
    A SubClass of Image Tensor to deal with Disparity/Depth value > 1.
    If the Tensor is modified, the maximum value always be referenced
    """
    _max_value = 0
    _ori_shape = None
    _mode_list = ['L', 'RGB']

    @staticmethod
    def __new__(cls, im: Union[ImageTensor, Tensor]):
        inp = im.squeeze()
        assert len(inp.shape) == 2
        max_value = inp.max()
        if isinstance(inp, ImageTensor):
            inp_ = inp / max_value
        else:
            inp_ = ImageTensor(inp / max_value)
        inp_ = super().__new__(cls, inp_)
        inp_._max_value = max_value
        inp_._ori_shape = inp_.shape[-2:]
        return inp_

    def __add__(self, other):
        if isinstance(other, ImageTensor):
            self_ = self.RGB()
        else:
            self_ = self.clone()
        return torch.Tensor.__add__(self_, other)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if res.__class__ is Tensor:
            res = DepthTensor(res)
        if res.__class__ is DepthTensor:
            arg = find_class(args, DepthTensor)
            res.pass_attr(arg)
            if res.shape != arg.shape:
                res.channel_pos = abs(update_channel_pos(res))
            return res
        else:
            return res

    def show(self, num=None, cmap='inferno', roi: list = None):
        if not num:
            num = self.im_name
        fig, ax = plt.subplots(num=num)
        im_display = self.squeeze()
        im_display = (im_display - im_display.min()) / (im_display.max() - im_display.min())
        if len(im_display.shape) > 2:
            im_display, cmap = im_display.permute(1, 2, 0), None
        else:
            im_display, cmap = im_display, cmap
        ax.imshow(im_display.cpu().numpy(), cmap=cmap)
        if roi is not None:
            for r, color in zip(roi, ['r', 'g', 'b']):
                rect = patches.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0]
                                         , linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
        plt.xticks([]), plt.yticks([])
        plt.show()
        return ax

    @property
    def max_value(self):
        return self._max_value

    @property
    def ori_shape(self):
        return self._ori_shape


class ImageCustom(np.ndarray):
    """
    A class defining the general basic framework of an image
    An instance is created using a numpy array or a path to an image file
    """

    def __new__(cls, inp, *args, name='new_image', dtype=None):
        # Input array is a path to an image OR an already formed ndarray instance
        if isinstance(inp, str):
            name = basename(inp)
            inp = io.imread(inp)
            if len(inp.shape) > 2:
                if np.sum(inp[:, :, 0] - inp[:, :, 1]) == 0:
                    inp = inp[:, :, 0]
        if dtype is not None:
            image = np.asarray(inp).astype(dtype).view(cls)
        else:
            if (inp.dtype == np.float64 or inp.dtype == np.float32) and inp.max() <= 1:
                image = np.float64(np.asarray(inp).view(cls))
            elif ((inp.dtype == np.float64 or inp.dtype == np.float32) and inp.max() > 1) or inp.dtype == np.uint8:
                image = np.uint8(np.asarray(inp).view(cls))
            elif inp.dtype == np.uint16:
                image = np.uint8(np.asarray(inp).view(cls) / 256)
            else:
                raise TypeError(f'{inp.dtype} is not a type supported by this constructor')
        image.name = name
        # add the new attributes to the created instance of Image
        if len(image.shape) == 2:
            image.cmap = 'GRAYSCALE'
        elif len(image.shape) == 3:
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            image.cmap = 'RGB'
        if image.name[-4] == '.':
            image.name = image.name[:-4]
        elif image.name[-5] == '.':
            image.name = image.name[:-5]
        image.pad = np.array([0, 0, 0, 0])
        if len(args) > 0:
            if isinstance(args[0], ImageCustom):
                image.pass_attr(args[0])
            elif isinstance(args[0], dict):
                image.__dict__.update(args[0])
        return image

    def __str__(self):
        ##
        # Redefine the way of printing
        if len(self.shape) == 0:
            return str(self.view(np.ndarray))
        else:
            return f"Resolution : {self.shape[1]}x{self.shape[0]}px\n" \
                   f"Current Domain : {self.cmap}\n"

    def gradient_orientation(self, mod=False):
        """
        :param mod: if true, gradient modulo pi
        :return: gradient intensity and orientation
        """
        Ix = cv.Sobel(self, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT_101)
        Iy = cv.Sobel(self, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT_101)
        grad = np.sqrt(Ix ** 2 + Iy ** 2)
        orient = cv.phase(Ix, Iy, angleInDegrees=True)
        if mod:
            orient = cv.normalize(abs(orient - 180), None, 0, 255, cv.NORM_MINMAX)
        return grad, orient

    def diff(self, other):
        assert len(self.shape) == len(other.shape), print("The two images dont have the same number of layer")
        assert self.dtype == other.dtype, print("The two images dont have the same type")
        if self.dtype == np.float64 and self.max() < 1:
            im = abs(np.asarray(self) - np.asarray(other))
        else:
            im = abs(np.asarray(self) / 255 - np.asarray(other) / 255) * 255
        return ImageCustom(im, self)

    def add(self, other):
        assert len(self.shape) == len(other.shape), print("The two images dont have the same number of layer")
        assert self.dtype == other.dtype, "The two images dont have the same type"
        if self.dtype == np.float64 and self.max() < 1:
            im = np.asarray(self) + np.asarray(other)
            im = np.minimum(im, np.ones_like(im))
        else:
            im = abs(np.asarray(self) / 255 + np.asarray(other) / 255) * 255
            im = np.minimum(im, np.ones_like(im) * 255)
        return ImageCustom(im, self)

    def __array_finalize__(self, image):
        # see InfoArray.__array_finalize__ for comments
        if image is None:
            return
        self.cmap = getattr(image, 'cmap', None)
        self.origin = getattr(image, 'origin', None)
        self.name = getattr(image, 'name', None)
        self.pad = getattr(image, 'pad', None)
        # self.current_value = getattr(image, 'current_value', None)

    def pass_attr(self, image):
        self.__dict__ = image.__dict__.copy()

    def show(self, num=None, figsize=(20, 20), dpi=40, facecolor=None, edgecolor=None, frameon=True, clear=False,
             cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None,
             extent=None, *, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None,
             data=None, **kwargs):
        plt.figure(num=num, figsize=figsize, dpi=dpi, facecolor=facecolor,
                   edgecolor=edgecolor, frameon=frameon, clear=clear)
        if self.cmap == "Vis" or "EDGES":
            cmap = 'gray'
            plt.imshow(self, cmap=cmap, norm=norm, aspect=aspect, interpolation=interpolation, alpha=alpha, vmin=vmin,
                       vmax=vmax, origin=origin, extent=extent, interpolation_stage=interpolation_stage,
                       filternorm=filternorm, filterrad=filterrad, resample=resample, url=url, data=data, **kwargs)
            plt.xticks([]), plt.yticks([])

        else:
            plt.imshow(self, cmap=cmap, norm=norm, aspect=aspect, interpolation=interpolation, alpha=alpha, vmin=vmin,
                       vmax=vmax, origin=origin, extent=extent, interpolation_stage=interpolation_stage,
                       filternorm=filternorm, filterrad=filterrad, resample=resample, url=url, data=data, **kwargs)
            plt.xticks([]), plt.yticks([])
        plt.show()

    def GRAYSCALE(self, true_value=True):
        if self.max() <= 1:
            self = np.uint8(255 * self)
        if (self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES') and len(self.shape) == 2:
            return self.copy()
        else:
            if self.cmap == 'GRAYSCALE':
                true_value = False
            if true_value:
                if self.cmap == 'RGB':
                    i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2GRAY), self)
                    i.cmap = 'GRAYSCALE'
                    return i
                elif self.cmap == 'BGR':
                    i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2GRAY), self)
                    i.cmap = 'GRAYSCALE'
                    return i
                else:
                    return self.RGB().GRAYSCALE()
            else:
                i = np.mean(self.copy(), axis=-1)
                i = ImageCustom(i, self)
                i.cmap = 'GRAYSCALE'
                return i

    def HSV(self):
        hsv = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            hsv[:, :, 0] = np.zeros_like(self)
            hsv[:, :, 1] = np.zeros_like(self)
            hsv[:, :, 2] = self.copy()
            i = ImageCustom(hsv, self)
        elif self.cmap == 'HSV':
            return self
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2HSV), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2HSV), self)
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2HSV), self)
        i.cmap = 'HSV'
        return i

    def HLS(self):
        hls = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            hls[:, :, 0] = np.zeros_like(self)
            hls[:, :, 2] = np.zeros_like(self)
            hls[:, :, 1] = self.copy()
            i = ImageCustom(hls, self)
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2HLS), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2HLS), self)
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2HLS), self)
        i.cmap = 'HLS'
        return i

    def YCrCb(self):
        ycc = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            ycc[:, :, 1] = np.zeros_like(self)
            ycc[:, :, 2] = np.zeros_like(self)
            ycc[:, :, 0] = self.copy()
            i = ImageCustom(ycc, self)
        elif self.cmap == 'YSCrCb':
            return self
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2YCrCb), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2YCrCb), self)
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2YCrCb), self)
        i.cmap = 'YCrCb'
        return i

    def BGR(self):
        if self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2BGR), self)
        elif self.cmap == 'LAB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LAB2BGR), self)
        elif self.cmap == 'LUV':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LUV2BGR), self)
        elif self.cmap == 'HSV':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HSV2BGR), self)
        elif self.cmap == 'BGR':
            return self.copy()
        elif self.cmap == 'YCrCb':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_YCrCb2BGR), self)
        elif self.cmap == 'HLS':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HLS2BGR), self)
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2BGR), self)
        i.cmap = 'BGR'
        return i

    def RGB(self, colormap='inferno'):
        if self.cmap == 'RGB' and len(self.shape) == 3:
            return self.copy()
        elif self.cmap == 'HSV':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HSV2RGB), self)
        elif self.cmap == 'LAB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LAB2RGB), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2RGB), self)
        elif self.cmap == 'YCrCb':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_YCrCb2RGB), self)
        elif self.cmap == 'HLS':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HLS2RGB), self)
        elif self.cmap == 'LUV':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LUV2RGB), self)
        else:
            rgb = np.empty([self.shape[0], self.shape[1], 3])
            x = np.linspace(0.0, 1.0, 256)
            cmap_rgb = cm.get_cmap(plt.get_cmap(colormap))(x)[np.newaxis, :, :3]
            if self.ndim == 2:
                rgb[:, :, 0] = cmap_rgb[0, self[:, :], 0]
                rgb[:, :, 1] = cmap_rgb[0, self[:, :], 1]
                rgb[:, :, 2] = cmap_rgb[0, self[:, :], 2]
                rgb *= 255
            else:
                rgb[:, :, 0] = cmap_rgb[0, self[:, :, 0], 0]
                rgb[:, :, 1] = cmap_rgb[0, self[:, :, 1], 1]
                rgb[:, :, 2] = cmap_rgb[0, self[:, :, 2], 2]
                rgb *= 255
            i = ImageCustom(rgb, self, dtype=np.uint8)
        i.cmap = 'RGB'
        return i

    def LAB(self):
        lab = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            lab[:, :, 0] = self.copy()
            lab[:, :, 1] = np.zeros_like(self)
            lab[:, :, 2] = np.zeros_like(self)
            i = ImageCustom(lab, self)
        elif self.cmap == 'LAB':
            return self.copy()
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2LAB), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2LAB), self)
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2LAB), self)
        i.cmap = 'LAB'
        return i

    def LUV(self):
        luv = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            luv[:, :, 0] = self.copy()
            luv[:, :, 1] = np.zeros_like(self)
            luv[:, :, 2] = np.zeros_like(self)
            i = ImageCustom(luv, self)
        elif self.cmap == 'LUV':
            return self.copy()
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2LUV), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2LUV), self)
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2LUV), self)
        i.cmap = 'LUV'
        return i

    def save(self, path=None):
        if not path:
            path = join(Path(dirname(__file__)).parent.absolute(), 'output')
        path = join(path, self.name + ".jpg")
        io.imsave(path, self, plugin=None, check_contrast=True)

    def median_filter(self, size=3):
        return ImageCustom(median_filter(self, size), self)
        # im.current_value = np.asarray(self)

    def gaussian_filter(self, sigma=2.0):
        im = ImageCustom(cv.GaussianBlur(self, (0, 0), sigma), self)
        # im.current_value = np.asarray(self)
        return im

    def mean_shift(self, value=0.5):
        if value > 1:
            value = value / 255
        if self.dtype == np.uint8:
            i = self.copy() / 255
            return np.uint8(i ** (np.log(value) / np.log(i.mean())) * 255)
        else:
            i = self.copy()
            return i ** (np.log(value) / np.log(i.mean()))

    def saliency(self, radius=0, color=True):
        if color:
            i = self.copy()
        else:
            i = self.GRAYSCALE()
        if radius == 0:
            return ImageCustom(abs(i / 255 - (i / 255).mean()) * 255).GRAYSCALE()
        else:
            return ImageCustom(abs(i / 255 - i.median_filter(size=radius) / 255) * 255).GRAYSCALE()

    def padding_2n(self, level=3, pad_type='zeros'):
        '''
        :param level: integer, the number of time the image has to be downscalable without loss after padding
        :param pad_type: 'zeros' for constant zeros padding, 'reflect_101' for 'abcba' padding, 'replicate' for 'abccc' padding... See opencv doc
        :return: image dowscalable without loss
        '''
        assert isinstance(level, int), print("level has to be an integer")
        m, n = self.shape[:2]
        # if m % 2**level == 0 and n % 2**level == 0:
        #     return self

        # Padding number for the height
        temp = m
        pad_v = 0
        l = 0
        while l < level:
            if temp % 2 != 0:
                pad_v += 1 * 2 ** l
                l += 1
                temp = (temp + 1) / 2
            else:
                temp = temp / 2
                l += 1
        pad_v = pad_v / 2
        # Padding number for the width
        temp = n
        pad_h = 0
        l = 0
        while l < level:
            if temp % 2 != 0:
                pad_h += 1 * 2 ** l
                l += 1
                temp = (temp + 1) / 2
            else:
                temp = temp / 2
                l += 1
        pad_h = pad_h / 2

        l_pad = int(pad_h if pad_h % 1 == 0 else pad_h + 0.5)
        r_pad = int(pad_h if pad_h % 1 == 0 else pad_h - 0.5)
        t_pad = int(pad_v if pad_v % 1 == 0 else pad_v + 0.5)
        b_pad = int(pad_v if pad_v % 1 == 0 else pad_v - 0.5)

        if pad_type == 'zeros':
            borderType = cv.BORDER_CONSTANT
            value = 0
        elif pad_type == 'reflect_101':
            borderType = cv.BORDER_REFLECT_101
            value = None
        elif pad_type == 'replicate':
            borderType = cv.BORDER_REPLICATE
            value = None
        elif pad_type == 'reflect':
            borderType = cv.BORDER_REFLECT
            value = None
        elif pad_type == 'reflect_101':
            borderType = cv.BORDER_REFLECT_101
            value = None
        im = ImageCustom(cv.copyMakeBorder(self, t_pad, b_pad, l_pad, r_pad, borderType, None, value=value), self)
        padding_final = np.array([t_pad, l_pad, b_pad, r_pad])
        im.pad = im.pad + padding_final
        return im

    def unpad(self):
        '''
        :return: Unpadded image
        '''
        t, l, b, r = self.pad
        if t != 0:
            self = self[t:, :]
        if l != 0:
            self = self[:, l:]
        if b != 0:
            self = self[:-b, :]
        if r != 0:
            self = self[:, :-r]
        self.pad = np.zeros_like(self.pad)
        return self

    def pyr_scale(self, octave=3, gauss=False, verbose=False):
        im = self.padding_2n(level=octave, pad_type='reflect_101')
        pyr_scale = {0: im}
        if verbose:
            print(f"level 0 shape : {pyr_scale[0].shape}")
        for lv in range(octave):
            pyr_scale[lv + 1] = ImageCustom(cv.pyrDown(pyr_scale[lv]), self)
            if gauss:
                pyr_scale[lv + 1] = ImageCustom(cv.GaussianBlur(pyr_scale[lv + 1], (5, 5), 0), pyr_scale[lv + 1])
            if verbose:
                print(f"level {lv + 1} shape : {pyr_scale[lv + 1].shape}")
        return pyr_scale

    def pyr_gauss(self, octave=3, interval=4, sigma0=1, verbose=False):
        k = 2 ** (1 / interval)
        im = self.padding_2n(level=octave, pad_type='reflect_101')
        pyr_gauss = {0: im}
        if verbose:
            print(f"level 0 shape : {pyr_gauss[0].shape}")
        for lv in range(octave):
            sigma = sigma0 * (2 ** lv)
            if lv != 0:
                pyr_gauss[lv + 1] = {0: ImageCustom(cv.pyrDown(pyr_gauss[lv][0]), im)}
            else:
                pyr_gauss[lv + 1] = {0: ImageCustom(pyr_gauss[lv], im)}
            for inter in range(interval):
                sigmaS = (k ** inter) * sigma
                pyr_gauss[lv + 1][inter + 1] = ImageCustom(cv.GaussianBlur(pyr_gauss[lv + 1][0], (0, 0), sigmaS), self)
            if verbose:
                print(f"level {lv + 1} shape : {pyr_gauss[lv + 1][0].shape}")
                cv.imshow('pyr_gauss', pyr_gauss[lv + 1][0].BGR())
                cv.waitKey(0)
        cv.destroyAllWindows()
        return pyr_gauss

    def expand_dims(self):
        im = self.copy()
        if len(self.shape) < 3:
            im = np.stack([im, im, im], axis=-1)
        return ImageCustom(im, self)

    def new_axis(self):
        im = self.copy()
        return np.atleast_3d(im)

    def match_shape(self, im2, keep_ratio=True, channel=False):
        im = self.copy()
        if im.shape == im2.shape:
            return im
        if (not (im.ndim == im2.ndim) or im.shape[-1] != im2.shape[-1]) and channel:
            if len(im2.shape) == 2:
                im = im[:, :, 0] / 3 + im[:, :, 1] / 3 + im[:, :, 2] / 3
            else:
                im = im.expand_dims()
        if not keep_ratio:
            im = cv.resize(im, [im2.shape[1], im2.shape[0]])
        else:
            h, w = im.shape[:2]
            h2, w2 = im2.shape[:2]
            ratio_h = h / h2
            ratio_w = w / w2
            if ratio_h == ratio_w:
                im = ImageCustom(cv.resize(im, [im2.shape[1], im2.shape[0]]))
            elif ratio_h < ratio_w:
                if im.dims == 3:
                    temp = np.zeros([h2, w2, im.shape[-1]])
                else:
                    temp = np.zeros([h2, w2])
                im = ImageCustom(cv.resize(im, [im2.shape[1], int(im.shape[0] / ratio_w)]))
                pad = im2.shape[0] - im.shape[0]
                if pad % 2 == 0:
                    pad = int(pad / 2)
                    temp[pad:-pad, :] = im
                else:
                    pad = int((pad + 1) / 2)
                    if pad > 1:
                        temp[pad:-(pad - 1), :] = im
                    else:
                        temp[pad:, :] = im
                im = temp.copy()
                del (temp)
            else:
                temp = np.zeros_like(im2)
                im = ImageCustom(cv.resize(im, [int(im.shape[1] / ratio_h), im2.shape[0]]))
                pad = im2.shape[1] - im.shape[1]
                if pad % 2 == 0:
                    pad = int(pad / 2)
                    temp[:, pad:-pad] = im
                else:
                    pad = int((pad + 1) / 2)
                    temp[:, pad:-(pad - 1)] = im
                im = temp.copy()
                del (temp)
        return ImageCustom(im, self)
