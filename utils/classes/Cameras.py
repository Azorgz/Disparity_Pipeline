import math
import os
import warnings
from pathlib import Path
import numpy as np
import torch
from kornia.geometry import PinholeCamera, angle_axis_to_rotation_matrix
from torch import Tensor
import inspect
from types import FrameType
from typing import cast, Union

from utils.classes.Image import ImageTensor


class BaseCamera(PinholeCamera):
    """
    setup: is not used for know, gives the other camera names in the rig
    is_positioned : Boolean specifying if the extrinsic matrix is known or by default.
    A positioned Camera can be used for depth-based projection and disparity projection in case of coplanarity
     with the source
    f : distance focal of the Camera Optic in meter
    aperture : Aperture used (not used for now)
    pixel_size : A Tuple giving the physical size of the pixels in (w h) order
    FOV: Value computed with the given value of Focal and pixel size
    """
    _setup = None
    _is_positioned = False
    _is_ref = False
    _f = None
    _aperture = None
    _pixel_size = None
    _FOV_v = None
    _FOV_h = None
    _name = 'BaseCam'

    def __init__(self, intrinsics: Union[Tensor, np.ndarray], extrinsics: Union[Tensor, np.ndarray],
                 path: (str or Path), name: str, device, is_positioned: bool, f: float, pixel_size: tuple,
                 aperture: float,
                 x, y, z, rx, ry, rz) -> None:
        self._name = name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._aperture = aperture
        assert os.path.exists(path)
        self.path = path
        h, w, im_type, im_calib = self._init_size_()
        self._im_type = im_type
        self._im_calib = im_calib
        if intrinsics is None:
            intrinsics = self._init_intrinsics_(h, w, f, pixel_size)
        else:
            intrinsics = torch.tensor(intrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
        if extrinsics is None:
            self.is_positioned = False
            extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            extrinsics = torch.tensor(extrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
            self.is_positioned = is_positioned
        self._f = f if f is not None else 1e-3
        self._pixel_size = pixel_size if pixel_size is not None else (
            self.f / intrinsics[0, 0, 0].cpu(), self.f / intrinsics[0, 0, 0].cpu())
        self._FOV_v = round(2 * math.atan(self.pixel_size[1] * h / (2 * self.f)) * 180 / math.pi, 1)
        self._FOV_h = round(2 * math.atan(self.pixel_size[0] * w / (2 * self.f)) * 180 / math.pi, 1)
        h = torch.tensor(h).unsqueeze(0).to(self.device)
        w = torch.tensor(w).unsqueeze(0).to(self.device)
        super(BaseCamera, self).__init__(intrinsics, extrinsics, h, w)

    def __str__(self):
        optical_parameter = self.optical_parameter()
        global_parameter = {i: j for i, j in self.save_dict().items() if i not in optical_parameter}
        string1 = '\n'.join([': '.join([str(key), str(v)]) for key, v in global_parameter.items()])
        gap = "\n\noptical parameters : \n".upper()
        string2 = '\n'.join([': '.join([str(key), str(v)]) for key, v in optical_parameter.items()])
        return string1 + gap + string2

    def optical_parameter(self):
        return {'f': (self.f * 10 ** 3, "mm"),
                'pixel_size': ((self.pixel_size[0] * 10 ** 6, self.pixel_size[1] * 10 ** 6), 'um'),
                'width': (float(self.width.cpu()), 'pixels'),
                'height': (float(self.height.cpu()), 'pixels'),
                'aperture': (self.aperture, ''),
                'FOVh': (self.FOV_h, '°'),
                'FOVv': (self.FOV_v, '°')}

    def save_dict(self):
        return {'name': self.name,
                'path': self.path,
                'intrinsics': self.intrinsics.squeeze().cpu().numpy().tolist(),
                'extrinsics': self.extrinsics.squeeze().cpu().numpy().tolist(),
                'is_ref': self.is_ref,
                'is_positioned': self.is_positioned,
                'im_type': self.im_type,
                'f': self.f,
                'pixel_size': list(self.pixel_size),
                'aperture': self.aperture}

    def _init_size_(self) -> tuple:
        try:
            im_path = ''
            for im in os.listdir(self.path):
                if 'calibration_image' in im:
                    im_path = f'{self.path}/{im}'
                    break
            im_calib = ImageTensor(im_path)
        except Exception as e:
            # print('There is no Calibration image, the calibration default image will be the 1st of '
            #       'the list')
            im_path = f'{self.path}/{sorted(os.listdir(self.path))[0]}'
            im_calib = ImageTensor(im_path)
        _, c, h, w = im_calib.shape
        if c == 1:
            im_type = 'IR'
        else:
            if np.array_equal(im_calib[:, :, 0], im_calib[:, :, 1]) and \
                    np.array_equal(im_calib[:, :, 0], im_calib[:, :, 2]):
                im_type = 'IR'
            else:
                im_type = 'RGB'
        return h, w, im_type, im_calib

    def _init_intrinsics_(self, h: int, w: int, f, pixel_size) -> Tensor:
        """
        :param h: Height of the images
        :param w: Width of the images
        :return: Init the intrinsic matrix of the camera with default parameter
        """
        if f is not None and pixel_size is not None:
            d = int(f / pixel_size[0]), int(f / pixel_size[1])
        else:
            d = (int(np.sqrt(h ** 2 + w ** 2) / 2), int(np.sqrt(h ** 2 + w ** 2) / 2))
        return torch.tensor([[d[0], 0, int(w / 2), 0],
                             [0, d[1], int(h / 2), 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.double).unsqueeze(0).to(self.device)

    def _init_extrinsics_(self, x, y, z, rx, ry, rz) -> Tensor:
        """ :return: init the camera in the space at the position O(0, 0, 0) wo rotation """
        x = x if x is not None else 0
        y = y if y is not None else 0
        z = z if z is not None else 0
        mat_tr = Tensor(np.array([x, y, z, 1]))
        rx = rx if rx is not None else 0
        ry = ry if ry is not None else 0
        rz = rz if rz is not None else 0
        mat_rot = angle_axis_to_rotation_matrix(Tensor(np.array([[rx, ry, rz]])))
        mat = torch.zeros([1, 4, 4])
        mat[:, :3, :3] = mat_rot
        mat[:, :, -1] = mat_tr
        return mat.inverse().to(dtype=torch.double).to(self.device)

    def reset(self):
        """Only settable by the _del_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '_del_camera_':
            self.is_ref = False
            self.is_positioned = False

    def update_setup(self, camera_ref, cameras) -> None:
        """Only settable by the _add_camera_,the _reset_ and _del_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '_add_camera_' or name == '_del_camera_' or name == '_reset_':
            self.setup = cameras
            self.is_ref = self.name == camera_ref
            self.is_positioned = True if self.is_ref else self.is_positioned

    def update_name(self, idx) -> None:
        setattr(self, 'name', f'{self.name}_{idx}')

    def update_pos(self, extrinsics=None, x=None, y=None, z=None, x_pix=None, y_pix=None, rx=None, ry=None, rz=None):
        x = x if x is not None else (
            x_pix * self.pixel_size / self.f if x_pix is not None else self.intrinsics[0, 0, 3].cpu())
        y = y if y is not None else (
            y_pix * self.pixel_size / self.f if y_pix is not None else self.intrinsics[0, 1, 3].cpu())
        if extrinsics is None:
            self.extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            self.extrinsics = extrinsics
        self.is_positioned = True

    # def is_in_fov(self, point:Union[Tensor, tuple, list, np.ndarray]):
    #     try:
    #         assert isinstance(point, Tensor) or isinstance(point, tuple) or isinstance(point, list) or isinstance(point, np.ndarray):
    #         assert len(point) == 3
    #     except AssertionError:
    #         print("the point must be a sequence of length 3")

    def pixel_size_at(self, distance=0):
        """
        :param distance: distance of interest or list of a point of interest coordinates
        :return: The size of a pixel in the image at such a distance
        """
        if isinstance(distance, list) or isinstance(distance, tuple):
            if len(distance) == 3:
                distance = Tensor(distance).to(self.device)
                distance = torch.sqrt(torch.sum((distance - self.center) ** 2))
        if not distance:
            distance = torch.sqrt(torch.sum(self.center ** 2))
        fov_v = 2 * math.tan(self.FOV_v / 360 * math.pi) * distance
        fov_h = 2 * math.tan(self.FOV_h / 360 * math.pi) * distance
        return fov_h / self.width, fov_v / self.height

    def __getitem__(self, index):
        im_path = f'{self.path}/{sorted(os.listdir(self.path))[index]}'
        im = ImageTensor(im_path, device=self.device)
        return im

    @property
    def center(self):
        return Tensor([self.tx, self.ty, self.tz]).to(self.device)

    @property
    def f(self):
        return self._f

    @property
    def aperture(self):
        return self._aperture

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def FOV_v(self):
        return self._FOV_v

    @property
    def FOV_h(self):
        return self._FOV_h

    @property
    def name(self):
        return self._name

    @property
    def extrinsics(self):
        return self._extrinsics

    @extrinsics.setter
    def extrinsics(self, value):
        """Only settable by the __init__, __new__, update_pos methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__' or name == 'update_pos' or name == '__init__':
            if not isinstance(value, Tensor):
                value = Tensor(value)
            if value.device != self.device:
                value = value.to(self.device)
            if value.dtype != torch.float64:
                value = value.view(torch.float64)
            if value.shape != torch.Size([1, 4, 4]):
                value = value.unsqueeze(0)
            self._extrinsics = value

    @property
    def setup(self):
        return self._setup

    @setup.setter
    def setup(self, setup):
        """Only settable by the update_rig method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_setup' or name == '_reset_':
            self._setup = setup

    @setup.deleter
    def setup(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def im_type(self):
        return self._im_type

    @im_type.setter
    def im_type(self, im_type):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__init__':
            self._im_type = im_type

    @im_type.deleter
    def im_type(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def im_calib(self):
        return self._im_calib

    @im_calib.setter
    def im_calib(self, im_calib):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__init__':
            self._im_calib = im_calib

    @im_calib.deleter
    def im_calib(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def is_ref(self):
        return self._is_ref

    @is_ref.setter
    def is_ref(self, is_ref):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_camera_ref' or name == 'reset':
            self._is_ref = is_ref

    @is_ref.deleter
    def is_ref(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def is_positioned(self):
        return self._is_positioned

    @is_positioned.setter
    def is_positioned(self, is_positioned):
        """Only settable by the _register_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'register_camera' or name == '_reset_' or '__init__' or '__new__':
            self._is_positioned = is_positioned

    @is_positioned.deleter
    def is_positioned(self):
        warnings.warn("The attribute can't be deleted")


class RGBCamera(BaseCamera):
    def __init__(self, intrinsics: Union[Tensor, np.ndarray], extrinsics: Union[Tensor, np.ndarray],
                 path: (str or Path),
                 device=None, name='RGB', im_type='RGB', is_ref=False, is_positioned=False,
                 f=None, pixel_size=None, aperture=2.0, x=None, y=None, z=None, rx=None, ry=None, rz=None) -> None:
        super(RGBCamera, self).__init__(intrinsics, extrinsics, path, name, device,
                                        f=f, pixel_size=pixel_size, aperture=aperture, is_positioned=is_positioned,
                                        x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
        assert self.im_type == 'RGB', 'The Folder does not contain RGB images'

    # def init3d(self):


class IRCamera(BaseCamera):
    def __init__(self, intrinsics: Union[Tensor, np.ndarray], extrinsics: Union[Tensor, np.ndarray],
                 path: (str or Path),
                 device=None, name='IR', im_type='RGB', is_ref=False, is_positioned=False,
                 f=None, pixel_size=None, aperture=2.0, x=None, y=None, z=None, rx=None, ry=None, rz=None) -> None:
        super(IRCamera, self).__init__(intrinsics, extrinsics, path, name, device,
                                       f=f, pixel_size=pixel_size, aperture=aperture, is_positioned=is_positioned,
                                       x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)
        assert self.im_type == 'IR', 'The Folder does not contain IR images'

