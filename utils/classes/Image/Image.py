import inspect
import os
import warnings
from os.path import *
from types import FrameType
from typing import Union, cast

import PIL.Image
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt, patches
from torch import Tensor
from torchvision.transforms.functional import to_pil_image

from utils.misc import time_fct


def find_class(args, class_name):
    arg = {}
    for a in args:
        if isinstance(a, class_name):
            return a
        elif isinstance(a, list) or isinstance(a, tuple):
            arg = find_class(a, class_name)
            if arg != {}:
                return arg
        else:
            arg = None
    return arg


def update_channel_pos(im):
    shape = np.array(im.shape)
    channel_pos = np.argwhere(shape == 3)
    channel_pos = channel_pos[0][0] if len(channel_pos >= 1) else \
        (np.argwhere(shape == 1)[0][0] if len(np.argwhere(shape == 1)) >= 1 else None)
    if channel_pos is None:
        return -1
    else:
        return int(channel_pos)


def pil_to_numpy(im):
    """
    Converts a PIL Image object to a NumPy array.
    Source : Fast import of Pillow images to NumPy / OpenCV arrays Written by Alex Karpinsky

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray: The NumPy array representing the image.
    """
    im.load()

    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)

        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


def find_best_grid(param):
    srt = int(np.floor(np.sqrt(param)))
    i = 0
    while srt * (srt + i) < param:
        i += 1
    return srt, srt + i


class ImageTensor(Tensor):
    """
    A class defining the general basic framework of a TensorImage.
    It can use all the methods from Torch plus some new ones.
    To create a new instance:
    --> From file (a str or path pointing towards an image file)
    --> From numpy array (shape h, w, c / h, w, c, a or h, w in case of mono-layered image)
    --> From a torch tensor (shape c, h, w  or b, c, h, w for batched tensor)
    An instance is created using a numpy array or a path to an image file or a PIL image
    """
    _im_type = None
    _im_name = None
    _im_pad = torch.tensor([[0, 0, 0, 0]])
    _color_mode = None
    _mode_list = ['1', 'L', 'RGB', 'RGBA', 'CMYK', 'LAB', 'HSV']
    _channel_pos = None
    _depth = '8'
    _batched = False

    def __init__(self, *args, **kwargs):
        super(ImageTensor, self).__init__()

    @staticmethod
    def __new__(cls, inp, *args, name: str = 'new_image',
                device: torch.device = None, cmap: str = None, **kwargs):
        # Input array is a path to an image OR an already formed ndarray instance
        if isinstance(inp, str):
            name = basename(inp).split('.')[0] if name == 'new_image' else name
            inp_ = Image.open(inp)
        elif isinstance(inp, np.ndarray):
            if inp.dtype == np.float64:
                inp_ = to_pil_image(np.float32(inp))
            else:
                inp_ = to_pil_image(inp)
        elif isinstance(inp, Tensor):
            # Case of batched tensors
            if len(inp.squeeze().shape) > 3 or (len(inp.shape) == 3 and inp.shape[0] > 3):
                inp_ = inp.clone()
            else:
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
            array = pil_to_numpy(inp_)
            color_mode = inp_.mode.split(';')[0]
            batched = False
            if len(inp_.mode.split(';')) > 1:
                color_mode = 'L'
                depth = inp_.mode.split(';')[1]
                inp_ = torch.from_numpy(array / 255)
            else:
                depth = '8'
                inp_ = torch.from_numpy(array)
            if len(inp_.shape) == 3:
                inp_ = inp_.permute([2, 0, 1])
            # t = transforms.ToTensor()
            # color_mode = inp_.mode
            # inp_ = t(inp_)
        elif isinstance(inp_, torch.Tensor):
            if len(inp_.shape) == 4:
                _, c, _, _ = inp_.shape
            elif len(inp_.shape) == 3:
                c, _, _ = inp_.shape
            else:
                raise NotImplementedError
            if c == 3:
                color_mode = 'RGB'
            elif c == 4:
                color_mode = 'RGBA'
            else:
                color_mode = 'L'
            depth = '8'
            if inp_.max() <= 1:
                inp_ *= 255
            batched = True
        else:
            raise NotImplementedError

        if isinstance(device, torch.device):
            image = inp_.to(device) / 255
        else:
            if torch.cuda.is_available():
                image = inp_.to(torch.device('cuda')) / 255
            else:
                image = inp_.to(torch.device('cpu')) / 255
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
        image._depth = depth
        image._batched = batched
        return image

    # Base Methods
    # def __s(self, i, j, sequence):
    #     super().__setslice__(i, j, sequence)

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
                # temp = temp[:1, :, :]
                im_type = 'IR'
                cmap = 'RGB'
            else:
                im_type = 'RGB'
                cmap = 'RGB'
        elif len(temp.shape) == 4:
            return self, 'IR', 'RGB', 1
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

    def batch(self, images: list):
        """
        Function to batch ImageTensor together
        :param images: list of ImageTensor to batch
        :return: batched ImageTensor
        """
        assert isinstance(images, list)
        assert len(images) > 0
        for i, im in enumerate(images):
            if im.shape != self.shape:
                images[i] = im.match_shape(self)
        new = torch.concatenate([self, *images], dim=0)
        new.pass_attribute(self)
        new.batched = True
        return new

    def interpolate(self, items):
        """
        :param items:  List/array or Tensor of shape (N,2) of tuple of coordinate to interpolate
        :return: Tensor of interpolated values
        """
        try:
            device = items.device
        except AttributeError:
            device = 'cpu'
        h, w = self.put_channel_at(1).shape[-2:]
        N = len(self) if self.batched else 1
        grid = items.unsqueeze(1).repeat(N, 1, 1, 1)
        grid[:, :, :, 0] = grid[:, :, :, 0] * 2 / w - 1
        grid[:, :, :, 1] = grid[:, :, :, 1] * 2 / h - 1
        return F.grid_sample(Tensor(self).to(device), grid, align_corners=True).squeeze()

    # Image manipulation methods
    def pad(self, im, **kwargs):
        '''
        Pad the image to match the given Tensor/Array size or with the list of padding indicated (left, right, top, bottom)
        :param im:
        :param kwargs:
        :return: a copy of self but padded
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
            h, w = 0, 0
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

    def hstack(self, *args, **kwargs):
        temp = [self.put_channel_at(-1).permute(1, 2, 0, 3)]
        for im in args:
            temp.append(im.put_channel_at(-1).permute(1, 2, 0, 3))
        res = torch.hstack(temp).permute(2, 3, 0, 1)
        res.pass_attr(self)
        return res

    def vstack(self, *args, **kwargs):
        temp = [self.put_channel_at(-1).permute(1, 2, 0, 3)]
        for im in args:
            temp.append(im.put_channel_at(-1).permute(1, 2, 0, 3))
        res = torch.vstack(temp).permute(2, 3, 0, 1)
        res.pass_attr(self)
        return res

    def pyrDown(self):
        out = self.put_channel_at(1)
        _, _, height, width = self.shape
        # downsample
        out: ImageTensor = F.interpolate(
            out,
            size=(int(float(height) / 2), int(float(width) // 2)),
            mode='bilinear',
            align_corners=True,
        )
        out.put_channel_at(self.channel_pos)
        out.pass_attr(self)
        return out

    def pyrUp(self):
        out = self.put_channel_at(1)
        _, _, height, width = self.shape
        # upsample
        out: ImageTensor = F.interpolate(
            out,
            size=(height * 2, width * 2),
            mode='bilinear',
            align_corners=True,
        )
        out.put_channel_at(self.channel_pos)
        out.pass_attr(self)
        return out

    def put_channel_at(self, idx=1):
        return torch.movedim(self, self.channel_pos, idx)

    def match_shape(self, other: Union[Tensor, tuple, list], keep_ratio=False):
        temp = self.put_channel_at()
        if isinstance(other, tuple) or isinstance(other, list):
            shape = other
            dims = len(other)
        elif isinstance(other, ImageTensor) or isinstance(other, DepthTensor):
            b = other.put_channel_at()
            dims = len(b.shape) - 2
            shape = b.shape[-dims:]
        else:
            b = other.clone()
            dims = len(b.shape) - 2
            shape = b.shape[-dims:]
        mode = 'bilinear' if dims <= 2 else 'trilinear'
        if keep_ratio:
            shape_temp = temp.shape[-dims:]
            ratio = torch.tensor(shape_temp) / torch.tensor(shape)
            ratio = ratio.max()
            temp = F.interpolate(temp, mode=mode, scale_factor=float((1 / ratio).cpu().numpy()))
        else:
            temp = F.interpolate(temp, size=shape, mode=mode, align_corners=True)
        if keep_ratio:
            temp = temp.pad(other)
        return temp.put_channel_at(self.channel_pos)

    def normalize(self, minmax=False, keep_abs_max=False):
        if keep_abs_max:
            a = torch.abs(self)
            m = a.min()
            M = a.max()
            a = (a - m) / (M - m)
        else:
            m = self.min()
            M = self.max()
            a = (self - m) / (M - m)
        if minmax:
            return a, m, M
        else:
            return a

    # utils methods
    def opencv(self, **kwargs):
        if self.color_mode == 'L':
            a = np.ascontiguousarray(Tensor.numpy(self.squeeze().cpu()) * 255, dtype=np.uint8)
        else:
            a = np.ascontiguousarray(self.RGB().put_channel_at(-1).squeeze().cpu().numpy()[..., [2, 1, 0]] * 255,
                                     dtype=np.uint8)
        return a

    def to_numpy(self, datatype=np.uint8, **kwargs):
        if datatype == np.uint8:
            factor = 255
        else:
            factor = 1
        if self.color_mode == 'L':
            if self.requires_grad:
                a = np.ascontiguousarray(Tensor.numpy(self.detach().squeeze().cpu()) * factor, dtype=datatype)
            else:
                a = np.ascontiguousarray(Tensor.numpy(self.squeeze().cpu()) * factor, dtype=datatype)
        else:
            if self.requires_grad:
                a = np.ascontiguousarray(self.detach().RGB().cpu().put_channel_at(-1).squeeze().numpy() * factor,
                                         dtype=datatype)
            else:
                a = np.ascontiguousarray(self.RGB().put_channel_at(-1).squeeze().cpu().numpy() * factor, dtype=datatype)
        return a

    @torch.no_grad()
    def show(self, num=None, cmap='gray', roi: list = None, point: Union[list, Tensor] = None):
        im_display = self.put_channel_at(1)
        batch, layers = im_display.shape[:2]
        if layers > 3 or batch > 1:
            im_display._multiple_show(num=num, cmap=cmap)
        else:
            if not num:
                num = self.im_name
            fig, ax = plt.subplots(ncols=1, nrows=1, num=num, squeeze=False)
            if im_display.im_type == 'IR':
                img, cmap = im_display.put_channel_at(-1).squeeze(), cmap
            else:
                img, cmap = im_display.put_channel_at(-1).squeeze(), None
            if point is not None:
                if img.im_type == 'IR':
                    img = img.RGB('gray')
                for center in point.squeeze():
                    center = center.cpu().long().numpy()
                    img = ImageTensor(cv.circle(img.opencv(), center, 5, (0, 255, 0), -1)[..., [2, 1, 0]])
                img = img.put_channel_at(-1).squeeze()
            ax[0, 0].imshow(img.to_numpy(), cmap=cmap)
            if roi is not None:
                for r, color in zip(roi, ['r', 'g', 'b']):
                    rect = patches.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0]
                                             , linewidth=2, edgecolor=color, facecolor='none')
                    ax[0, 0].add_patch(rect)
            ax[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            plt.show()
            return ax[0, 0]

    def _multiple_show(self, num=None, cmap='gray'):
        im = self.put_channel_at()
        batch, layers = im.shape[:2]
        if layers > 3:
            im.im_type = 'IR'
            im_display = [*im.reshape([batch * layers, 1, *self.shape[2:]])]
        else:
            im_display = [*im]
        if not num:
            num = self.im_name
        rows, cols = find_best_grid(len(im_display))
        fig = plt.figure(num=num)
        axes = [(fig.add_subplot(rows, cols, r * cols + c + 1) if (r * cols + c + 1 <= len(im_display)) else None) for r
                in range(rows) for c in range(cols)]
        for i, img in enumerate(im_display):
            img = img.unsqueeze(0)
            img.pass_attr(self)
            if img.im_type == 'IR':
                img, cmap = img.put_channel_at(-1).squeeze(), cmap
            else:
                img, cmap = img.put_channel_at(-1).squeeze(), None
            axes[i].imshow(img.to_numpy(), cmap=cmap)
        for a in axes:
            if a is not None:
                a.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        return axes

    def save(self, path, name=None, ext='png', **kwargs):
        name = self.im_name + f'.{ext}' if name is None else name
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if not cv.imwrite(path + f'/{name}', self.opencv()):
            raise Exception("Could not write image")

    def to_tensor(self):
        """
        Remove all attributes to keep only the data as a torch tensor.
        :return: Tensor
        """
        return torch.Tensor(self.data)

    @property
    def im_name(self) -> str:
        return self._im_name

    @im_name.setter
    def im_name(self, name) -> None:
        self._im_name = name

    @property
    def batched(self) -> bool:
        return self._batched

    @batched.setter
    def batched(self, value) -> None:
        self._batched = value

    @property
    def depth(self) -> str:
        return self._depth

    @depth.setter
    def depth(self, value) -> None:
        self._depth = value

    @property
    def im_type(self) -> str:
        return self._im_type

    @im_type.setter
    def im_type(self, t) -> None:
        self._im_type = t
        # warnings.warn("The attribute can't be modified")

    @property
    def mode_list(self) -> list:
        return self._mode_list

    @property
    def channel_pos(self) -> int:
        if self._channel_pos is not None:
            return self._channel_pos
        else:
            self.channel_pos = update_channel_pos(self)
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
            # cmap_rgb = Tensor(cm.get_cmap(plt.get_cmap(colormap))(x)[:, :3]).to(self.device).squeeze()
            cmap_rgb = Tensor(cm[colormap](x)[:, :3]).to(self.device).squeeze()
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


class DepthTensor(ImageTensor):
    """
    A SubClass of Image Tensor to deal with Disparity/Depth value > 1.
    If the Tensor is modified, the maximum value always be referenced
    """
    _max_value = 0
    _min_value = 0
    _scaled = False
    _ori_shape = None
    _mode_list = ['L', 'RGB']
    _im_type = 'Depth'
    _color_mode = 'L'

    @staticmethod
    def __new__(cls, im: Union[ImageTensor, Tensor], device: torch.device = None):
        inp = im.squeeze()
        assert len(inp.shape) == 2
        max_value = inp.max()
        min_value = inp.min()
        inp_ = (inp - min_value) / (max_value - min_value)
        inp_ = super().__new__(cls, inp_, device=device)
        inp_._max_value = max_value
        inp_._min_value = min_value
        inp_._ori_shape = inp_.shape[-2:]
        inp_.im_type = 'Depth'
        return inp_

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if res.__class__ is Tensor:
            res = DepthTensor(res).scale()
        if res.__class__ is DepthTensor:
            arg = find_class(args, DepthTensor)
            if arg is not None:
                res.pass_attr(arg)
                if res.shape != arg.shape:
                    res.channel_pos = abs(update_channel_pos(res))
            return res
        else:
            return res

    def show(self, num=None, cmap='inferno', roi: list = None, point: Union[list, Tensor] = None):
        im_display = [*self]
        if not num:
            num = self.im_name
        fig, ax = plt.subplots(ncols=len(im_display), num=num, squeeze=False)
        for i, img in enumerate(im_display):
            im_display = img.squeeze()
            # im_display = (im_display - im_display.min()) / (im_display.max() - im_display.min())
            if len(im_display.shape) > 2:
                im_display, cmap = im_display.permute(1, 2, 0), None
            else:
                im_display, cmap = im_display, cmap
            if point is not None:
                for center in point.squeeze():
                    center = center.cpu().long().numpy()
                    im_display = cv.circle(im_display.opencv(), center, 5, (0, 255, 0), -1)[..., [2, 1, 0]]
            ax[0, i].imshow(im_display.cpu().numpy(), cmap=cmap)
            if roi is not None:
                for r, color in zip(roi, ['r', 'g', 'b']):
                    rect = patches.Rectangle((r[1], r[0]), r[3] - r[1], r[2] - r[0]
                                             , linewidth=2, edgecolor=color, facecolor='none')
                    ax[0, i].add_patch(rect)

            ax[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        return ax

    def save(self, path, name=None, save_image=False, **kwargs):
        if save_image:
            ImageTensor(self.inverse_depth(remove_zeros=True)
                        .normalize()).RGB().save(path, name=name)
        else:
            name = self.im_name + '.tiff' if name is None else name
            im = self.opencv()
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            if not cv.imwrite(path + f'/{name}', im):
                raise Exception("Could not write image")

    def opencv(self, **kwargs):
        if self.color_mode == 'L':
            a = np.ascontiguousarray(Tensor.numpy(self.unscale().squeeze().cpu()) * 255 ** 2, dtype=np.uint16)
        else:
            a = np.ascontiguousarray(
                Tensor.numpy(self.unscale().put_channel_at(-1).squeeze().cpu())[..., [2, 1, 0]] * 255 ** 2,
                dtype=np.uint16)
        return a

    def clamp(self, mini=None, maxi=None, *, out=None):
        self.max_value = min(maxi, self.max_value)
        self.min_value = max(mini, self.min_value)
        return torch.clamp(self, min=mini, max=maxi)

    def normalize(self, minmax=False, keep_abs_max=True):
        a = super().normalize(minmax=minmax, keep_abs_max=keep_abs_max)
        return a

    @property
    def scaled(self):
        return self._scaled

    @scaled.setter
    def scaled(self, value):
        self._scaled = value

    def scale(self):
        new = self.clone()
        if not new.scaled:
            new.scaled = True
            return new * (new.max_value - new.min_value) + new.min_value
        else:
            return new

    def unscale(self):
        new = self.clone()
        if new.scaled:
            new.scaled = False
            return (new - new.min_value) / (new.max_value - new.min_value)
        else:
            return new

    def inverse_depth(self, remove_zeros=False, remove_max=True, factor=100):
        temp = self.clone()
        if remove_zeros:
            temp[temp == 0] = temp.max()
        if remove_max:
            temp[temp == temp.max()] = temp.min()
        temp = factor / (temp + 1)
        return temp.normalize()

    @property
    def color_mode(self) -> str:
        return self._color_mode

    @color_mode.setter
    def color_mode(self, v) -> None:
        pass

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, v):
        self._max_value = v

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, v):
        self._min_value = v

    @property
    def ori_shape(self):
        return self._ori_shape
