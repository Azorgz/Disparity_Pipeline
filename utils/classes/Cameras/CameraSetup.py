import inspect
import os
import warnings
from types import FrameType
from typing import cast, Union

import numpy as np
import torch
import yaml
from kornia.geometry import find_fundamental, essential_from_fundamental, motion_from_essential_choose_solution, \
    relative_transformation, \
    rotation_matrix_to_angle_axis
from torch import Tensor
from torch.linalg import vector_norm
from utils.classes.Cameras import IRCamera, RGBCamera
from utils.classes.Geometry.KeypointsGenerator import KeypointsGenerator
# from utils.classes.Registration import Registration
from utils.classes.Cameras.StereoSetup import StereoSetup, DepthSetup
from utils.misc import path_leaf
from utils.visualization import show_epipolar


class StereoPairs:
    _setup = []
    _names = []
    _left = []
    _right = []

    def __new__(cls, *args, **kwargs):
        new = super(StereoPairs, cls).__new__(cls)
        new.setup = []
        new.names = []
        new.left = []
        new.right = []
        return new

    def __add__(self, stereo_setup: Union[StereoSetup, DepthSetup]):
        if stereo_setup.name not in self.names:
            if isinstance(stereo_setup, StereoSetup):
                self.setup.append(stereo_setup)
                self.names.append(stereo_setup.name)
                self.left.append(stereo_setup.left.id)
                self.right.append(stereo_setup.right.id)
            else:
                self.setup.append(stereo_setup)
                self.names.append(stereo_setup.name)
                self.left.append(stereo_setup.ref.id)
                self.right.append(stereo_setup.target.id)
        return self

    def __sub__(self, stereo_setup: StereoSetup or str):
        if isinstance(stereo_setup, str):
            if stereo_setup in self.names:
                idx = self.names.index(stereo_setup)
                del self.setup[idx]
                del self.names[idx]
                del self.left[idx]
                del self.right[idx]
        elif isinstance(stereo_setup, StereoSetup) or isinstance(stereo_setup, DepthSetup):
            if stereo_setup.name not in self.names:
                idx = self.names.index(stereo_setup.name)
                del self.setup[idx]
                del self.names[idx]
                del self.left[idx]
                del self.right[idx]
        return self

    def __call__(self, *args, verbose=True, **kwargs) -> StereoSetup or int:
        """
        Given two cameras names it returns the name of the existing pair formed by these cameras
        :return: name of the pair
        """
        if len(args) == 1:
            if isinstance(args[0], int) and 0 <= args[0] < len(self):
                return self.setup[args[0]]
            else:
                if verbose:
                    print(f'There is no recorded pair with index {args[0]}')
                return -1
        elif len(args) >= 2:
            if isinstance(args[0], str) and isinstance(args[1], str):
                idx = self.find_pair(args[0], args[1])
                if idx is not None:
                    return self.setup[idx]
                elif len(args) >= 3:
                    return self(args[2:])
                else:
                    if verbose:
                        print(f'There is no recorded pair with {args[0]} and {args[1]}')
                    return -1
            elif isinstance(args[0], int):
                return self(args[0])
            elif isinstance(args[1], int):
                return self(args[1])
            elif len(args) >= 3:
                return self(args[2:])
            else:
                if verbose:
                    print(f'There is no recorded pair corresponding with these args')
                return -1
        else:
            if verbose:
                print(f'There is no recorded pair corresponding with these args')
            return -1

    def find_pair(self, cam1: str, cam2: str) -> int or None:
        if f'{cam1}&{cam2}' in self.names:
            return self.names.index(f'{cam1}&{cam2}')
        elif f'{cam2}&{cam1}' in self.names:
            return self.names.index(f'{cam2}&{cam1}')
        else:
            return None

    def __len__(self):
        return len(self.setup)

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value

    @property
    def setup(self):
        return self._setup

    @setup.setter
    def setup(self, value):
        self._setup = value


class CameraSetup:
    """
    A class that manage all the cameras, there relative positions, and the exiting stereo pairs
    The CameraSetup can be initialized from a yaml file or from a list of cameras. It's used as a collection of all the
    cameras called by there respective id (that parameter could differ from the camera's name)
    """
    _cameras = {}
    _cameras_IR = {}
    _cameras_RGB = {}
    _camera_ref = None
    _model = None
    _stereo_pair = StereoPairs()
    _depth_pair = StereoPairs()
    _base2Ref = torch.eye(4)
    _coplanarity_tolerance = 0.15
    _max_depth = 200
    _min_depth = 0

    def __init__(self, *args, device=None, from_file=False, model=None, name=None, max_depth=None, min_depth=None,
                 **kwargs):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        if max_depth:
            self._max_depth = max_depth
        if min_depth:
            self._min_depth = min_depth
        # self.registration = Registration(self.device, self.model)
        # self.manual_calibration_available = True if self.registration.model is not None else False
        if from_file:
            assert os.path.exists(from_file)
            self._init_from_file_(from_file, self.device)
            self.name = path_leaf(from_file)
        else:
            self(*args, **kwargs)
            self.name = name if name is not None else 'Camera_Setup'

    def __new__(cls, *args, **kwargs):
        new = super(CameraSetup, cls).__new__(cls)
        new.cameras = {}
        new.cameras_IR = {}
        new.cameras_RGB = {}
        new.camera_ref = None
        new.stereo_pair = StereoPairs()
        new.depth_pair = StereoPairs()
        return new

    def __call__(self, *args, print_info=False, **kwargs):
        cameras = []
        if len(args) > 0:
            for cam in args:
                if isinstance(cam, IRCamera) or isinstance(cam, RGBCamera):
                    if cam in self.cameras.values():
                        cameras.append(self._del_camera_(cam, print_info))
                    else:
                        cameras.append(self._add_camera_(cam, print_info))
                elif cam in self.cameras.keys():
                    cameras.append(self._del_camera_(cam, print_info))
                else:
                    warnings.warn('Only a RGBCamera or IRCamera can be added/deleted')
        return cameras

    def __str__(self):
        list_ir = {key: f' ({"Ref" if cam.is_ref else ("In position" if cam.is_positioned else "Position unknown")})'
                   for key, cam
                   in self.cameras_IR.items()}
        list_rgb = {key: f' ({"Ref" if cam.is_ref else ("In position" if cam.is_positioned else "Position unknown")})'
                    for key, cam
                    in self.cameras_RGB.items()}
        # string = f'Cameras : {", ".join([self.cameras[key].id for key in self.cameras.keys()])}\n'
        string = f'Cameras IR : {", ".join([self.cameras[key].name + f" ({key})" + list_ir[key] for key in self.cameras_IR.keys()])}\n'
        string += f'Cameras RGB : {", ".join([self.cameras[key].name + f" ({key})" + list_rgb[key] for key in self.cameras_RGB.keys()])}\n'
        string += f'The Reference Camera is : {self.camera_ref}'
        return string

    def _init_from_file_(self, path, device=None):
        with open(path, "r") as file:
            conf = yaml.safe_load(file)
        ref = conf['camera_ref']
        cameras = {}
        for cam, v in conf['cameras'].items():
            v['intrinsics'] = np.array(v['intrinsics'])
            v['extrinsics'] = np.array(v['extrinsics'])
            v['device'] = device
            if v['im_type'] == 'IR':
                cameras[cam] = IRCamera(**v)
            else:
                cameras[cam] = RGBCamera(**v)
            if cam == ref:
                self(cameras[cam])
        for cam in conf['cameras'].keys():
            if cam != ref:
                self(cameras[cam])
        if conf['stereo_pair']['left']:
            self.calibration_for_stereo(**conf['stereo_pair'])
        if conf['depth_pair']['ref']:
            self.calibration_for_depth(**conf['depth_pair'])

    def save(self, path, name='Setup_Camera.yaml'):
        dict_setup = {'cameras': {key: cam.save_dict() for key, cam in self.cameras.items()},
                      'camera_ref': self.camera_ref,
                      'stereo_pair': {'left': self.stereo_pair.left,
                                      'right': self.stereo_pair.right,
                                      'name': self.stereo_pair.names},
                      'depth_pair': {'ref': self.depth_pair.left,
                                     'target': self.depth_pair.right,
                                     'name': self.depth_pair.names}}
        name = f'{path}/{name}'
        with open(name, 'w') as file:
            yaml.dump(dict_setup, file)

    def _add_camera_(self, camera: Union[IRCamera, RGBCamera], print_info) -> IRCamera or RGBCamera:
        k = 0
        while camera.id in self.cameras.keys():
            camera.update_id(k)
            k += 1
        self.cameras[camera.id] = camera
        if self.nb_cameras == 1:
            self.update_camera_ref(camera.id)
        if isinstance(camera, IRCamera):
            self.cameras_IR[camera.id] = camera
        else:
            self.cameras_RGB[camera.id] = camera
        for cam in self.cameras.values():
            cam.update_setup(self.camera_ref, [k for k in self.cameras.keys() if k != cam.id])
        # self.cameras_calibration[camera.id] = {'matrix': torch.eye(3), 'crop': Tensor([0, 0, 0, 0])}
        if print_info:
            print(f'The {camera.im_type} Camera {camera.name} has been added to the Rig as {camera.id}')
        return camera

    def _del_camera_(self, camera: Union[IRCamera, RGBCamera, str], print_info) -> IRCamera or RGBCamera:
        for key, v in self.cameras.items():
            if camera == key:
                if isinstance(self.cameras[camera], IRCamera):
                    self.cameras_IR.pop(camera)
                else:
                    self.cameras_RGB.pop(camera)
                if self.camera_ref == camera:
                    self.camera_ref = None
                    self._set_default_new_ref_(camera)
                camera = self.cameras.pop(camera)
                break
            elif camera is v:
                if isinstance(self.cameras[camera.id], IRCamera):
                    self.cameras_IR.pop(camera.id)
                else:
                    self.cameras_RGB.pop(camera.id)
                if self.camera_ref == camera.id:
                    self.camera_ref = None
                    self._set_default_new_ref_(camera.id)
                camera = self.cameras.pop(camera.id)
                break
        camera.reset()
        for cam in self.cameras.values():
            cam.update_setup(self.camera_ref, [k for k in self.cameras.keys() if k != cam.id])
        if print_info:
            print(f'The {camera.im_type} Camera {camera.name} has been removed from the Rig')
        return camera

    def _set_default_new_ref_(self, name):
        """Here the variable name specify the camera which CANT be the new Reference"""
        new = None
        for key, cam in self.cameras.items():
            if new is None and key != name:
                new = key
            if cam.is_positioned and key != name:
                new = key
                break
        self.update_camera_ref(new)

    @torch.no_grad()
    def recover_pose_from_keypoints(self, cam: Union[IRCamera, RGBCamera, str], *args, ref=None,
                                    t=None) -> IRCamera or RGBCamera:
        """
        This function re-set the extrinsic parameters of a camera using consecutively the fundamental matrix
        determination by the 8-points method, then extracting the Essential matrix from it (according the intrinsic is known...)
        and finally recovering R and T matrix from Essential.
        :param t: The known translation distance for one direction (scaling parameter)
        :param cam: Camera to position
        :param ref: is no ref is given, the reference camera of the system will be used
        :return: the newly positioned camera
        """
        if args:
            cams = [cam, *args]
            t = t if (t is not None and len(t) == len(cams)) else None
            for idx, c in enumerate(cams):
                t_cam = t[idx] if t is not None else t
                ref = self.recover_pose_from_keypoints(c, ref=ref, t=t_cam)
        else:
            if not isinstance(cam, str):
                cam = cam.id
            if self._check_if_cam_is_in_setup_(cam):
                Kpts_gen = KeypointsGenerator(self.device, detector='SIFT', matcher='SNN')
                cam_ref = ref if ref is not None and isinstance(ref, str) else self._camera_ref
                ref = ref if isinstance(ref, Tensor) else None
                assert cam_ref != cam
                assert self.cameras[cam_ref].is_positioned

                K1, K2 = self.cameras[cam_ref].intrinsics[:, :3, :3].to(torch.float32), \
                    self.cameras[cam].intrinsics[:, :3, :3].to(torch.float32)
                # Keypoints extraction followings the chosen method
                im_dst = self.cameras[cam].im_calib
                im_src = self.cameras[cam_ref].im_calib

                calib_method = 'auto'
                # calib_method = 'manual' if im_src.im_type != im_dst.im_type else 'auto'
                pts_src, pts_dst = Kpts_gen(im_src, im_dst, pts_ref=ref, method=calib_method, min_kpt=20, th=0.85,
                                            draw_result=True)
                pts_src, pts_dst = pts_src.to(self.device), pts_dst.to(self.device)
                # pts_src_opencv, pts_dst_opencv = np.float32(pts_src.squeeze().cpu()), np.float32(pts_dst.squeeze().cpu())

                # Computation of the fundamental matrix and display of the epipolar lines
                # F_mat, mask = cv.findFundamentalMat(pts_src_opencv, pts_dst_opencv, cv.FM_LMEDS)
                # _, H1, H2 = cv.stereoRectifyUncalibrated(pts_src_opencv, pts_dst_opencv, F_mat, im_dst.shape[-2:])
                # img1_rectified = ImageTensor(cv.warpPerspective(im_src.opencv(), H1, (im_src.shape[-1], im_src.shape[-2]))[..., [2, 1, 0]])
                # img2_rectified = ImageTensor(cv.warpPerspective(im_dst.opencv(), H2, (im_dst.shape[-1], im_dst.shape[-2]))[..., [2, 1, 0]])
                # pts_src, pts_dst = Kpts_gen(img1_rectified, img2_rectified, pts_ref=ref, method=calib_method, min_kpt=20, th=0.85)
                # pts_src_opencv, pts_dst_opencv = np.float32(pts_src.squeeze().cpu()), np.float32(pts_dst.squeeze().cpu())
                # F_mat, mask = cv.findFundamentalMat(pts_src_opencv, pts_dst_opencv, cv.FM_LMEDS)
                # _, H1, H2 = cv.stereoRectifyUncalibrated(pts_src_opencv, pts_dst_opencv, F_mat, img1_rectified.shape[-2:])
                # img1_rectified = ImageTensor(cv.warpPerspective(img1_rectified.opencv(), H1, (img1_rectified.shape[-1], img1_rectified.shape[-2]))[..., [2, 1, 0]])
                # img2_rectified = ImageTensor(cv.warpPerspective(img2_rectified.opencv(), H2, (img2_rectified.shape[-1], img2_rectified.shape[-2]))[..., [2, 1, 0]])
                # img1_rectified.show()
                # img2_rectified.show()
                # im_src_new = ImageTensor(cv.warpPerspective(im_src.opencv(), H1, (im_src.shape[-1]+100, im_src.shape[-2]+100))[..., [2, 1, 0]])
                # im_dst_new = ImageTensor(cv.warpPerspective(im_dst.opencv(), H2, (im_dst.shape[-1]+100, im_dst.shape[-2]+100))[..., [2, 1, 0]])
                # (im_src_new*0.5 + im_dst_new).show()
                # F_mat = cv.normalizeFundamental(F_mat)
                # pts_src, pts_dst = pts_src.to(self.device), pts_dst.to(self.device)
                # pts_src_norm = normalize_points_with_intrinsics(pts_src, K1)
                # pts_dst_norm = normalize_points_with_intrinsics(pts_dst, K2)
                F_mat = find_fundamental(pts_src, pts_dst)
                F_mat = Tensor(F_mat).to(self.device).unsqueeze(0)
                show_epipolar(im_src, im_dst, F_mat, pts_src, pts_dst)

                # Extraction of the Essential Matrix from the Fundamental & decomposition into motion
                E_mat = essential_from_fundamental(F_mat.squeeze(0), K1, K2)
                R, T, pts_3d = motion_from_essential_choose_solution(E_mat, K1, K2, pts_src, pts_dst, mask=None)

                if t[0]:
                    T = T / T[0, 0, 0] * t[0]
                elif t[1]:
                    T = T / T[0, 1, 0] * t[1]
                elif t[2]:
                    T = T / T[0, 2, 0] * t[2]
                extrinsics = torch.eye(4, dtype=torch.float64).unsqueeze(0)
                extrinsics[0, :3, :3] = R.inverse().squeeze()
                extrinsics[0, :3, -1] = -T.squeeze()
                self.cameras[cam].update_pos(extrinsics)
                return pts_src

    def update_camera_ref(self, cam: Union[IRCamera, RGBCamera, str]):
        if not isinstance(cam, str):
            cam = cam.id
        if self._check_if_cam_is_in_setup_(cam) and cam != self.camera_ref:
            if self.camera_ref:  # If we already have a camera Ref
                self.cameras[self.camera_ref].is_ref = False  # It's not the Ref anymore
                if not self.cameras[cam].is_positioned:  # If the new Ref wasn't positioned
                    self.cameras[self.camera_ref].is_positioned = False  # The old ref is not positioned
            self.camera_ref = cam
            self.cameras[self.camera_ref].is_ref = True  # The Ref is not positioned,it's intrinsics matrix is Id(4)
            self.cameras[self.camera_ref].is_positioned = False
            # matrix_ref = torch.linalg.inv(self.cameras[cam].extrinsics[0, :3, :3])
            # matrix_ref = self.cameras[cam].extrinsics[0, :3, :3].transpose(-2, -1)
            # tr_ref = self.cameras[cam].extrinsics[0, :, -1]
            for key in self.cameras.keys():
                if key != cam and self.cameras[key].is_positioned:
                    extrinsic = relative_transformation(self.cameras[cam].extrinsics, self.cameras[key].extrinsics)
                    self.cameras[key].update_pos(extrinsics=extrinsic)
                else:
                    self.cameras[key].update_pos(torch.eye(4, dtype=torch.float64))

    def update_camera_relative_position(self, name, extrinsics=None, x=None, y=None, z=None, x_pix=None, y_pix=None,
                                        rx=None, ry=None, rz=None):
        """
        Update the camera position in regard to the Ref Camera position.
        :param name: id of the Camera to be moved
        :param extrinsics: new extrinsics matrix (optional)
        :param x: new x position in regard to the Cam_ref position
        :param y: new y position in regard to the Cam_ref position
        :param z: new z position in regard to the Cam_ref position
        :param x_pix: new x position in pixel in regard to the Cam_ref position
        :param y_pix: new y position in pixel in regard to the Cam_ref position
        :param rx: new rx angle in regard to the Cam_ref position
        :param ry: new ry angle in regard to the Cam_ref position
        :param rz: new rz angle in regard to the Cam_ref position
        :return: None
        """
        if self._check_if_cam_is_in_setup_(name):
            if name != self.camera_ref or self.nb_cameras == 1:
                self.cameras[name].update_pos(extrinsics, x, y, z, x_pix, y_pix, rx, ry, rz)
            else:
                for cam in self.cameras.keys():
                    if cam != name:
                        self.update_camera_ref(cam)
                        self.cameras[name].update_pos(extrinsics, x, y, z, x_pix, y_pix, rx, ry, rz)
                        self.update_camera_ref(name)
                        break

    def move_camera_from_its_position(self, name, dx=None, dy=None, dz=None, drx=None, dry=None, drz=None):
        """
        Move a Camera in the Setup from its position to its position + the given delta in each direction/angle
        The units are meters/rad
        :param name: id of the Camera to be moved
        :param dx: displacement in x direction in meter
        :param dy: displacement in y direction in meter
        :param dz: displacement in z direction in meter
        :param drx: rotation around the x-axis in rad
        :param dry: rotation around the y-axis in rad
        :param drz: rotation around the z-axis in rad
        :return: None
        """
        if self._check_if_cam_is_in_setup_(name):
            if name == self.camera_ref:
                self.update_camera_relative_position(name, x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
            else:
                rx, ry, rz = rotation_matrix_to_angle_axis(self.cameras[name].extrinsics[:, :3, :3]).to(
                    device='cpu').numpy()
                x, y, z = self.cameras[name].translation_vector[0, :, 0].to(device='cpu').numpy()
                self.update_camera_relative_position(name, x=dx + x, y=dy + y, z=dz + z,
                                                     rx=drx + rx, ry=dry + ry, rz=drz + rz)

    def calibration_for_stereo(self, left, right, name=None):
        """
        For now only working for horizontal Setup
        :param name: id of the created StereoPair. Default : the type of the stereo Cameras
        :param left: name of the left Camera
        :param right: name of the right Camera
        :return:
        """
        if isinstance(left, list) and isinstance(right, list) and isinstance(name, list):
            assert len(left) == len(right) == len(name)
            for l, r, n in zip(left, right, name):
                self.calibration_for_stereo(l, r, name=n)
        elif self._check_if_cam_is_in_setup_(left) and self._check_if_cam_is_in_setup_(right):
            try:
                assert self.is_coplanar(left, right)
            except AssertionError:
                warnings.warn("The chosen Cameras are not coplanar")
                return 0
            left = self.cameras[left]
            right = self.cameras[right]
            if left.im_type != right.im_type:
                warnings.warn(f"The chosen Cameras are not using the same modality. "
                              f"left Camera is {left.im_type} and right Camera is {right.im_type}")
            if left.extrinsics[0, 0, -1] < right.extrinsics[0, 0, -1]:
                left, right = right, left
            elif left.extrinsics[0, 0, -1] == right.extrinsics[0, 0, -1]:
                warnings.warn('The Camera are not spaced enough to compute disparity')
            s = StereoSetup(left, right, self.device, name, depth_max=self.max_depth, depth_min=self.min_depth)
            self.stereo_pair += s

    def calibration_for_depth(self, ref, target, name=None):
        if isinstance(ref, list) and isinstance(target, list) and isinstance(name, list):
            assert len(ref) == len(target) == len(name)
            for r, t, n in zip(ref, target, name):
                self.calibration_for_depth(r, t, name=n)
        elif self._check_if_cam_is_in_setup_(ref) and self._check_if_cam_is_in_setup_(target):
            ref = self.cameras[ref]
            target = self.cameras[target]
            if ref.extrinsics[0, 0, -1] < target.extrinsics[0, 0, -1]:
                ref, target = target, ref
            elif torch.equal(ref.extrinsics[0, 0, -1], target.extrinsics[0, 0, -1]):
                warnings.warn('The Camera have the same pose, impossible to compute depth')
                return 0
            try:
                assert ref.is_positioned and target.is_positioned
            except AssertionError:
                warnings.warn("The chosen Cameras are not Positioned")
                return 0
            try:
                assert ref.im_type == target.im_type
            except AssertionError:
                warnings.warn("The chosen Cameras are not using the same modality")
                return 0
            try:
                assert torch.equal(ref.intrinsics, target.intrinsics)
            except AssertionError:
                warnings.warn("The chosen Cameras don't have the same intrinsics matrix")
                return 0
            d = DepthSetup(ref, target, self.device, name=name, depth_max=self.max_depth, depth_min=self.min_depth)
            self.depth_pair += d

    # def update_model(self, model):
    #     self.registration.model = model
    #     self.manual_calibration_available = True

    def _check_if_cam_is_in_setup_(self, cam: Union[IRCamera, RGBCamera, str]):
        if not isinstance(cam, str):
            name = cam.id
        else:
            name = cam
        try:
            assert name in self.cameras.keys(), "This Camera doesn't exist"
        except AssertionError:
            return False
        return True

    def is_parallel(self, cam1, cam2):
        try:
            assert (self.cameras[cam1].is_positioned or self.cameras[cam1].is_ref) \
                   and (self.cameras[cam2].is_positioned or self.cameras[cam2].is_ref)
        except AssertionError:
            warnings.warn('The Camera position is not known')
        x1 = self.cameras[cam1].extrinsics[0, :, 0]
        x1 /= (vector_norm(x1) if vector_norm(x1) != 0 else 1)
        y1 = self.cameras[cam1].extrinsics[0, :, 1]
        y1 /= (vector_norm(y1) if vector_norm(y1) != 0 else 1)
        z2 = self.cameras[cam2].extrinsics[0, :, 2]
        z2 /= (vector_norm(z2) if vector_norm(z2) != 0 else 1)
        tol = torch.sqrt(torch.dot(x1, z2) ** 2 + torch.dot(y1, z2) ** 2)
        return tol <= self.coplanarity_tolerance

    def is_coplanar(self, cam1, cam2):
        try:
            assert self.is_parallel(cam1, cam2)
        except AssertionError:
            warnings.warn('The Cameras image plans are not Parallel')
            return False
        tz1 = self.cameras[cam1].extrinsics[0, 2, 3]
        tz2 = self.cameras[cam2].extrinsics[0, 2, 3]
        tol = torch.abs(tz1 - tz2)
        return tol <= self.coplanarity_tolerance * 10

    def depth_ready(self, cam=None):
        if cam:
            return self.cameras[cam].is_positioned
        else:
            return [cam for cam, v in self.cameras.items() if v.is_positioned]

    def disparity_ready(self, cam1=None, cam2=None):
        cam1, cam2 = [cam2, cam1] if cam1 is None else [cam1, cam2]
        if cam1 is None and cam2 is None:
            return {cam: [cam2 for cam2, v2 in self.cameras.items() if
                          (self.stereo_pair(cam, cam2, verbose=False) != -1) and cam != cam2]
                    for cam, v in self.cameras.items() if cam in ''.join(self.stereo_pair.names)}
        else:
            if cam2 is not None:
                return True if self.stereo_pair(cam1, cam2, verbose=False) != -1 else False
            else:
                res = []
                for idx, l in self.stereo_pair.left:
                    if l == cam1:
                        res.append(self.stereo_pair.right[idx])
                for idx, r in self.stereo_pair.right:
                    if r == cam1:
                        res.append(self.stereo_pair.left[idx])
                if res:
                    return res
                else:
                    return False

    @property
    def depth_pair(self):
        return self._depth_pair

    @depth_pair.setter
    def depth_pair(self, value):
        self._depth_pair = value

    @property
    def stereo_pair(self):
        return self._stereo_pair

    @stereo_pair.setter
    def stereo_pair(self, value):
        self._stereo_pair = value

    @property
    def coplanarity_tolerance(self):
        return self._coplanarity_tolerance

    @property
    def pos(self):
        return {key: cam.extrinsics for key, cam in self.cameras.items()}

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def min_depth(self):
        return self._min_depth

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_model' or name == '__new__' or name == '__init__':
            self._model = model.to(self.device) if model is not None else None

    @model.deleter
    def model(self):
        self._model = None
        self.manual_calibration_available = False

    @property
    def camera_ref(self):
        return self._camera_ref

    @camera_ref.setter
    def camera_ref(self, camera):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_camera_ref' or name == '_del_camera_' or name == '_set_default_new_ref_' or name == '__new__':
            self._camera_ref = camera

    @camera_ref.deleter
    def camera_ref(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def nb_cameras_IR(self):
        return len(self.cameras_IR)

    @property
    def nb_cameras_RGB(self):
        return len(self.cameras_RGB)

    @property
    def nb_cameras(self):
        return len(self.cameras)

    def __len__(self):
        return self.nb_cameras

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__':
            self._cameras = value

    @cameras.deleter
    def cameras(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def cameras_IR(self):
        return self._cameras_IR

    @cameras_IR.setter
    def cameras_IR(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__':
            self._cameras_IR = value

    @cameras_IR.deleter
    def cameras_IR(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def cameras_RGB(self):
        return self._cameras_RGB

    @cameras_RGB.setter
    def cameras_RGB(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__':
            self._cameras_RGB = value

    @cameras_RGB.deleter
    def cameras_RGB(self):
        warnings.warn("The attribute can't be deleted")
