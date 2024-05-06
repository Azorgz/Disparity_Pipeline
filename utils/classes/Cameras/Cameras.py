import math
import os
import warnings
from pathlib import Path
import numpy as np
import torch
from kornia.geometry import PinholeCamera, axis_angle_to_rotation_matrix
from torch import Tensor
import inspect
from types import FrameType
from typing import cast, Union
from utils.misc import print_tuple
from utils.classes.Image.Image import ImageTensor
from glob import glob
import imagesize

ext_available = ['/*.png', '/*.jpg', '/*.jpeg', '/*.tif', '/*.tiff']


class BaseCamera(PinholeCamera):
    '''
    A camera object subclassing the camera from Kornia.
    :param path: Always required (valid path or in the future stream url)
    :param device: not mandatory (defined as gpu if a gpu is available)
    :param name: not mandatory, defined as 'BaseCam' by default
    :param id: not mandatory, defined as 'BaseCam' by default
    :param is_positioned: not mandatory, defined as False by default. This parameter is True if the relative pose of the camera in the camera rig is known
    :param aperture: only cosmetic parameter for now
    :param intrinsics: if define it will create a camera with this matrix. If no other parameters is specified, the FOV/f/pixel size/sensor size will be defined by default
    :param f: focal in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or pixel_size...
    :param pixel_size: pixel size in micrometer, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or focal...
    :param sensor_size: sensor size in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as focal or pixel_size...
    :param HFOV: horizontal angular field of view in degrees
    :param VFOV: vertical angular field of view in degrees
    :param sensor_resolution: defined by the images find in path resolution. If set it's only a control value emetting a warning if the source images are not at matching resolution
    :param extrinsics: if define the camera will be positionned using that matrix, ignoring the following parameters. If no pose parameter is given, the pose will be canonical by default
    :param x: relative position in x defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param y: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param z: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param rx: relative rotation around x defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param ry: relative rotation around y defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param rz: relative rotation around z defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param in_degree: set to True if you express the position angles in degrees
    '''
    _is_positioned = False
    _is_ref = False
    _aperture = None
    _setup = None
    _name = 'BaseCam'
    _id = 'BaseCam'
    _im_type = None
    _f = None
    _pixel_size = None
    _VFOV = None
    _HFOV = None
    _aspect_ratio = None
    _sensor_size = None
    _sensor_resolution = None
    _path = []
    _files = []

    def __init__(self,
                 path: (str or Path) = None,
                 device=None,
                 name='BaseCam',
                 id='BaseCam',
                 is_positioned=False,
                 aperture=None,
                 # Extrinsic args
                 intrinsics: Union[Tensor, np.ndarray] = None,
                 f: (float or tuple) = None,
                 pixel_size: (float or tuple) = None,
                 sensor_size: tuple = None,
                 HFOV: float = None,
                 VFOV: float = None,
                 aspect_ratio: float = None,
                 sensor_resolution: tuple = None,
                 # Extrinsic args
                 extrinsics: Union[Tensor, np.ndarray] = None,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 rx: float = None,
                 ry: float = None,
                 rz: float = None,
                 in_degree: bool = False,
                 **kwargs) -> None:
        # General parameters
        self._id = id
        self._name = name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._aperture = aperture
        self._init_path(path)
        h, w, im_type, im_calib = self._init_size_()
        self._sensor_resolution = sensor_resolution if sensor_resolution is not None else (w, h)
        if self._sensor_resolution[0] != w or self._sensor_resolution[1] != h:
            warnings.warn(
                f'The specified sensor resolution {sensor_resolution} and the source image resolution {(w, h)} are different')
        self._im_type = im_type
        self._im_calib = im_calib

        # Intrinsics parameters definition
        intrinsics, parameters = self._init_intrinsics_(
            intrinsics=intrinsics,
            f=f,
            pixel_size=pixel_size,
            sensor_size=sensor_size,
            HFOV=HFOV,
            VFOV=VFOV,
            aspect_ratio=aspect_ratio)
        self._f = parameters['f']
        self._pixel_size = parameters['pixel_size']
        self._sensor_size = parameters['sensor_size']
        self._VFOV = parameters['VFOV']
        self._HFOV = parameters['HFOV']
        self._aspect_ratio = parameters['aspect_ratio']

        if extrinsics is None:
            self.is_positioned = False
            extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            extrinsics = torch.tensor(extrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
            self.is_positioned = is_positioned

        h = torch.tensor(self.sensor_resolution[1]).unsqueeze(0).to(self.device)
        w = torch.tensor(self.sensor_resolution[0]).unsqueeze(0).to(self.device)
        super(BaseCamera, self).__init__(intrinsics, extrinsics, h, w)

    def __str__(self):
        optical_parameter = self.optical_parameter()
        global_parameter = {i: j for i, j in self.save_dict().items() if i not in optical_parameter}
        string1 = '\n'.join([': '.join([str(key), str(v)]) for key, v in global_parameter.items()])
        gap = "\n\noptical parameters : \n".upper()
        string2 = '\n'.join([': '.join([str(key), print_tuple(v)]) for key, v in optical_parameter.items()])
        return string1 + gap + string2

    def optical_parameter(self):
        return {'f': (self.f[0] * 10 ** 3, "mm"),
                'pixel_size': ((self.pixel_size[0] * 10 ** 6, self.pixel_size[1] * 10 ** 6), 'um'),
                'sensor_size': ((self.sensor_size[0] * 10 ** 3, self.sensor_size[1] * 10 ** 3), 'mm'),
                'width': (float(self.width.cpu()), 'pixels'),
                'height': (float(self.height.cpu()), 'pixels'),
                'aperture': (self.aperture, ''),
                'HFOV': (self.HFOV, '°'),
                'VFOV': (self.VFOV, '°')}

    def save_dict(self):
        return {'name': self.name,
                'id': self.id,
                'path': self.path,
                'intrinsics': self.intrinsics.squeeze().cpu().numpy().tolist(),
                'extrinsics': self.extrinsics.squeeze().cpu().numpy().tolist(),
                'is_ref': self.is_ref,
                'is_positioned': self.is_positioned,
                'im_type': self.im_type,
                'f': [self.f[0] * 1e3, self.f[1] * 1e3],
                'sensor_resolution': [self.sensor_resolution[0], self.sensor_resolution[1]],
                'aperture': self.aperture}

    def _init_path(self, path):
        if isinstance(path, list):
            for p in path:
                self._init_path(p)
        else:
            try:
                assert os.path.exists(path)
                self.path = [*self.path, path]
                files_grabbed = []
                for ext in ext_available:
                    files_grabbed.extend(glob(path + ext))
                self.files = sorted([*self.files, *files_grabbed])
                return 0
            except AssertionError:
                print(f'There is no image file at {path}, a source is necessary to configure a new camera')
                raise AssertionError

    def _init_size_(self) -> tuple:
        im_path = ''
        im_size = None
        for f in self.files:
            if im_size is None:
                im_size = imagesize.get(f)
            else:
                assert im_size == imagesize.get(f), print(
                    f'Several images size have been found, image_size enforced at {im_size}')
            if 'calibration_image' in f:
                if im_path == '':
                    im_path = f
                # else:
                #     print('There are several Calibration images, the calibration default image will be the 1st found')
        if im_path == '':
            # print('There is no Calibration image, the calibration default image will be the 1st of the list')
            im_path = self.files[0]
        im_calib = ImageTensor(im_path)
        _, c, h, w = im_calib.shape
        im_type = im_calib.im_type
        return h, w, im_type, im_calib

    def _init_intrinsics_(self, **kwargs):
        if kwargs['intrinsics'] is not None:
            intrinsics, parameters = intrinsics_parameters_from_matrix(self.sensor_resolution, **kwargs)
            intrinsics = torch.tensor(intrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
        else:
            parameters = intrinsics_parameters_wo_matrix(self.sensor_resolution, **kwargs)
            intrinsics = self._init_intrinsics_matrix(self.sensor_resolution[1], self.sensor_resolution[0],
                                                      parameters['f'], parameters['pixel_size'])
        return intrinsics, parameters

    def _init_intrinsics_matrix(self, h: int, w: int, f, pixel_size) -> Tensor:
        """
        :param h: Height of the images
        :param w: Width of the images
        :return: Init the intrinsic matrix of the camera with default parameter
        """
        if f is not None and pixel_size is not None:
            d = int(f[0] / pixel_size[0]), int(f[1] / pixel_size[1])
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
        mat_rot = axis_angle_to_rotation_matrix(Tensor(np.array([[rx, ry, rz]])))
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
            self.is_ref = self.id == camera_ref
            self.is_positioned = True if self.is_ref else self.is_positioned

    def update_id(self, idx) -> None:
        setattr(self, 'id', f'{self.id}_{idx}')

    def update_pos(self, extrinsics=None, x=None, y=None, z=None, x_pix=None, y_pix=None, rx=None, ry=None, rz=None):
        x = x if x is not None else (
            x_pix * self.pixel_size[0] / self.f[0] if x_pix is not None else self.extrinsics[0, 0, 3].cpu())
        y = y if y is not None else (
            y_pix * self.pixel_size[1] / self.f[1] if y_pix is not None else self.extrinsics[0, 1, 3].cpu())
        if extrinsics is None:
            self.extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            self.extrinsics = extrinsics
        self.is_positioned = True

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
        fov_v = 2 * math.tan(self.VFOV / 360 * math.pi) * distance
        fov_h = 2 * math.tan(self.HFOV / 360 * math.pi) * distance
        return fov_h / self.width, fov_v / self.height

    @torch.no_grad()
    def __getitem__(self, index, autopad=False, **kwargs):
        im_path = self.files[index]
        im = ImageTensor(im_path, device=self.device)
        if autopad:
            temp = torch.zeros(*im.shape[:-2], self.sensor_resolution[1], self.sensor_resolution[0])
            im = im.pad(temp)
        return im

    def random_image(self, autopad=False, **kwargs):
        # list_im = os.listdir(self.path))
        index = torch.randint(0, len(self.files), [1])
        im = self.__getitem__(index, autopad=autopad)
        return im, index

    @property
    def center(self):
        return Tensor([self.tx, self.ty, self.tz]).to(self.device)

    @property
    def f(self):
        return self._f

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def aperture(self):
        return self._aperture

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    def sensor_size(self):
        return self._sensor_size

    @property
    def sensor_resolution(self):
        return self._sensor_resolution

    @property
    def aspect_ratio(self):
        return self._aspect_ratio

    @property
    def VFOV(self):
        return self._VFOV

    @property
    def HFOV(self):
        return self._HFOV

    @property
    def pixel_FOV(self):
        return (self.VFOV / self.sensor_resolution[0], self.HFOV / self.sensor_resolution[1])

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def files(self):
        return self._files

    @files.setter
    def files(self, value):
        self._files = value

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
    '''
    A RGB camera object subclassing the camera from Kornia.
    :param path: Always required (valid path or in the future stream url)
    :param device: not mandatory (defined as gpu if a gpu is available)
    :param name: not mandatory, defined as 'RGB cam' by default
    :param id: not mandatory, defined as 'RGB cam' by default
    :param is_positioned: not mandatory, defined as False by default. This parameter is True if the relative pose of the camera in the camera rig is known
    :param aperture: only cosmetic parameter for now
    :param intrinsics: if define it will create a camera with this matrix. If no other parameters is specified, the FOV/f/pixel size/sensor size will be defined by default
    :param f: focal in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or pixel_size...
    :param pixel_size: pixel size in micrometer, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or focal...
    :param sensor_size: sensor size in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as focal or pixel_size...
    :param HFOV: horizontal angular field of view in degrees
    :param VFOV: vertical angular field of view in degrees
    :param sensor_resolution: defined by the images find in path resolution. If set it's only a control value emetting a warning if the source images are not at matching resolution
    :param extrinsics: if define the camera will be positionned using that matrix, ignoring the following parameters. If no pose parameter is given, the pose will be canonical by default
    :param x: relative position in x defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param y: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param z: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param rx: relative rotation around x defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param ry: relative rotation around y defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param rz: relative rotation around z defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param in_degree: set to True if you express the position angles in degrees
    '''

    def __init__(self,
                 path: (str or Path) = None,
                 device=None,
                 name='RGB cam',
                 id='RGB cam',
                 is_positioned=False,
                 aperture=None,
                 # Extrinsic args
                 intrinsics: Union[Tensor, np.ndarray] = None,
                 f: (float or tuple) = None,
                 pixel_size: (float or tuple) = None,
                 sensor_size: tuple = None,
                 HFOV: float = None,
                 VFOV: float = None,
                 aspect_ratio: float = None,
                 sensor_resolution: tuple = None,
                 # Extrinsic args
                 extrinsics: Union[Tensor, np.ndarray] = None,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 rx: float = None,
                 ry: float = None,
                 rz: float = None,
                 in_degree: bool = False,
                 **kwargs) -> None:
        super(RGBCamera, self).__init__(
            path=path,
            device=device,
            name=name,
            id=id,
            is_positioned=is_positioned,
            aperture=aperture,
            # Extrinsic args
            intrinsics=intrinsics,
            f=f,
            pixel_size=pixel_size,
            sensor_size=sensor_size,
            HFOV=HFOV,
            VFOV=VFOV,
            aspect_ratio=aspect_ratio,
            sensor_resolution=sensor_resolution,
            # Extrinsic args
            extrinsics=extrinsics,
            x=x,
            y=y,
            z=z,
            rx=rx,
            ry=ry,
            rz=rz,
            in_degree=in_degree,
            **kwargs)
        assert self.im_type == 'RGB', 'The Folder does not contain RGB images'

    # def init3d(self):


class IRCamera(BaseCamera):
    '''
    An IR camera object subclassing the camera from Kornia.
    :param path: Always required (valid path or in the future stream url)
    :param device: not mandatory (defined as gpu if a gpu is available)
    :param name: not mandatory, defined as 'IR cam' by default
    :param id: not mandatory, defined as 'IR cam' by default
    :param is_positioned: not mandatory, defined as False by default. This parameter is True if the relative pose of the camera in the camera rig is known
    :param aperture: only cosmetic parameter for now
    :param intrinsics: if define it will create a camera with this matrix. If no other parameters is specified, the FOV/f/pixel size/sensor size will be defined by default
    :param f: focal in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or pixel_size...
    :param pixel_size: pixel size in micrometer, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or focal...
    :param sensor_size: sensor size in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as focal or pixel_size...
    :param HFOV: horizontal angular field of view in degrees
    :param VFOV: vertical angular field of view in degrees
    :param sensor_resolution: defined by the images find in path resolution. If set it's only a control value emetting a warning if the source images are not at matching resolution
    :param extrinsics: if define the camera will be positionned using that matrix, ignoring the following parameters. If no pose parameter is given, the pose will be canonical by default
    :param x: relative position in x defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param y: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param z: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param rx: relative rotation around x defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param ry: relative rotation around y defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param rz: relative rotation around z defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param in_degree: set to True if you express the position angles in degrees
    '''

    def __init__(self,
                 path: (str or Path) = None,
                 device=None,
                 name='IR cam',
                 id='IR cam',
                 is_positioned=False,
                 aperture=None,
                 # Extrinsic args
                 intrinsics: Union[Tensor, np.ndarray] = None,
                 f: (float or tuple) = None,
                 pixel_size: (float or tuple) = None,
                 sensor_size: tuple = None,
                 HFOV: float = None,
                 VFOV: float = None,
                 aspect_ratio: float = None,
                 sensor_resolution: tuple = None,
                 # Extrinsic args
                 extrinsics: Union[Tensor, np.ndarray] = None,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 rx: float = None,
                 ry: float = None,
                 rz: float = None,
                 in_degree: bool = False,
                 **kwargs) -> None:
        super(IRCamera, self).__init__(
            path=path,
            device=device,
            name=name,
            id=id,
            is_positioned=is_positioned,
            aperture=aperture,
            # Extrinsic args
            intrinsics=intrinsics,
            f=f,
            pixel_size=pixel_size,
            sensor_size=sensor_size,
            HFOV=HFOV,
            VFOV=VFOV,
            aspect_ratio=aspect_ratio,
            sensor_resolution=sensor_resolution,
            # Extrinsic args
            extrinsics=extrinsics,
            x=x,
            y=y,
            z=z,
            rx=rx,
            ry=ry,
            rz=rz,
            in_degree=in_degree,
            **kwargs)
        assert self.im_type == 'IR', 'The Folder does not contain IR images'
        i = self.__getitem__(0)


def intrinsics_parameters_from_matrix(sensor_resolution, **kwargs) -> (np.ndarray, dict):
    assert kwargs['intrinsics'] is not None
    intrinsics = kwargs['intrinsics']
    if kwargs['f'] is not None:
        f = (kwargs['f'] / 1e3, kwargs['f'] / 1e3) if not len_2(kwargs['f']) else (
            kwargs['f'][0] / 1e3, kwargs['f'][1] / 1e3)
        pixel_size = (f[0] / intrinsics[0, 0], f[1] / intrinsics[1, 1])
        sensor_size = (sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1])
        HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[0] / pixel_size[1]
    elif kwargs['pixel_size'] is not None:
        pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(kwargs['pixel_size']) else (
            kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
        f = (intrinsics[0, 0] * pixel_size[0], intrinsics[1, 1] * pixel_size[1])
        sensor_size = (sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1])
        HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[0] / pixel_size[1]
    elif kwargs['sensor_size'] is not None:
        sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
        pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        f = (intrinsics[0, 0] * pixel_size[0], intrinsics[1, 1] * pixel_size[1])
        HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
        aspect_ratio = pixel_size[0] / pixel_size[1]
    else:
        f = None
        pixel_size = None
        sensor_size = None
        HFOV = kwargs['HFOV']
        VFOV = kwargs['VFOV']
        aspect_ratio = kwargs['aspect_ratio']
    return intrinsics, {'f': (round(f[0] * 1e3, 3) / 1e3, round(f[1] * 1e3, 3) / 1e3),
                        'pixel_size': (round(pixel_size[0] * 1e6, 3) / 1e6, round(pixel_size[1] * 1e6, 3) / 1e6),
                        'sensor_size': (round(sensor_size[0] * 1e3, 3) / 1e3, round(sensor_size[1] * 1e3, 3) / 1e3),
                        'aspect_ratio': aspect_ratio,
                        'HFOV': round(HFOV, 2),
                        'VFOV': round(VFOV, 2)}


def intrinsics_parameters_wo_matrix(sensor_resolution, **kwargs) -> dict:
    # With f known :
    if kwargs['f'] is not None:
        f = (kwargs['f'] / 1e3, kwargs['f'] / 1e3) if not len_2(kwargs['f']) else (
            kwargs['f'][0] / 1e3, kwargs['f'][1] / 1e3)
        if kwargs['pixel_size'] is not None:
            pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(
                kwargs['pixel_size']) else (kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
            sensor_size = (sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1])
            HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
            VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
            aspect_ratio = pixel_size[0] / pixel_size[1]
        elif kwargs['sensor_size'] is not None:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
            HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
            VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
            aspect_ratio = pixel_size[0] / pixel_size[1]
        elif kwargs['HFOV'] is not None and kwargs['VFOV'] is not None:
            HFOV = kwargs['HFOV']
            VFOV = kwargs['VFOV']
            sensor_size = (np.arctan(HFOV / 360 * np.pi) * 2 * f[0], np.arctan(VFOV / 360 * np.pi) * 2 * f[1])
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
            aspect_ratio = pixel_size[0] / pixel_size[1]
        else:
            HFOV = kwargs['HFOV']
            VFOV = kwargs['VFOV']
            sensor_size = None
            pixel_size = None
            aspect_ratio = 1

    # With the sensor size known :
    elif kwargs['pixel_size'] is not None or kwargs['sensor_size'] is not None:
        if kwargs['pixel_size'] is not None:
            pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(kwargs['pixel_size']) \
                else (kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
            sensor_size = sensor_resolution[0] * pixel_size[0], sensor_resolution[1] * pixel_size[1]
        elif kwargs['sensor_size'] is not None:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (sensor_size[0] / sensor_resolution[0], sensor_size[1] / sensor_resolution[1])
        else:
            sensor_size = (kwargs['sensor_size'][0] / 1e3, kwargs['sensor_size'][1] / 1e3)
            pixel_size = (kwargs['pixel_size'] / 1e6, kwargs['pixel_size'] / 1e6) if not len_2(kwargs['pixel_size']) \
                else (kwargs['pixel_size'][0] / 1e6, kwargs['pixel_size'][1] / 1e6)
        if (kwargs['HFOV'] is not None) + (kwargs['VFOV'] is not None):
            if kwargs['HFOV'] is not None and kwargs['VFOV'] is not None:
                HFOV = kwargs['HFOV']
                VFOV = kwargs['VFOV']
                f = (
                    sensor_size[0] / (2 * np.tan(HFOV / 360 * np.pi)),
                    sensor_size[1] / (2 * np.tan(VFOV / 360 * np.pi)))
            elif kwargs['HFOV'] is not None:
                HFOV = kwargs['HFOV']
                f = (
                    sensor_size[0] / (2 * np.tan(HFOV / 360 * np.pi)),
                    sensor_size[0] / (2 * np.tan(HFOV / 360 * np.pi)))
                VFOV = 2 * np.arctan(sensor_size[1] / (2 * f[1])) * 180 / np.pi
            else:
                VFOV = kwargs['VFOV']
                f = (
                    sensor_size[1] / (2 * np.tan(VFOV / 360 * np.pi)),
                    sensor_size[1] / (2 * np.tan(VFOV / 360 * np.pi)))
                HFOV = 2 * np.arctan(sensor_size[0] / (2 * f[0])) * 180 / np.pi
        else:
            f = None
            pixel_size = kwargs['pixel_size']
            sensor_size = kwargs['sensor_size']
            HFOV = None
            VFOV = None
            aspect_ratio = kwargs['aspect_ratio']
        aspect_ratio = pixel_size[0] / pixel_size[1]
    else:
        f = kwargs['f']
        pixel_size = kwargs['pixel_size']
        sensor_size = kwargs['sensor_size']
        HFOV = kwargs['HFOV']
        VFOV = kwargs['VFOV']
        aspect_ratio = kwargs['aspect_ratio']
    return {'f': (round(f[0] * 1e3, 3) / 1e3, round(f[1] * 1e3, 3) / 1e3),
            'pixel_size': (round(pixel_size[0] * 1e6, 3) / 1e6, round(pixel_size[1] * 1e6, 3) / 1e6),
            'sensor_size': (round(sensor_size[0] * 1e3, 3) / 1e3, round(sensor_size[1] * 1e3, 3) / 1e3),
            'aspect_ratio': aspect_ratio,
            'HFOV': round(HFOV, 2),
            'VFOV': round(VFOV, 2)}


def len_2(vec):
    if isinstance(vec, tuple) or isinstance(vec, list) or isinstance(vec, torch.Tensor) or isinstance(vec, np.ndarray):
        return True if len(vec) == 2 else False
    else:
        return False
