from typing import Union
import cv2 as cv
import numpy as np
import torch
from kornia.geometry import StereoCamera, relative_transformation, get_perspective_transform, warp_perspective
from torch import Tensor, FloatTensor
from torch.nn.functional import grid_sample
from utils.classes.Cameras import IRCamera, RGBCamera
from utils.manipulation_tools import extract_roi_from_map


class StereoSetup(StereoCamera):
    """
    A class which add a few properties to the Kornia class StereoCamera
    """
    _left = None
    _right = None
    _name = 'StereoSetup'
    _ROI = [0, 0, 0, 0]

    def __init__(self, left: Union[IRCamera, RGBCamera], right: Union[IRCamera, RGBCamera],
                 device: torch.device, name: str = None, accuracy: float = 0.25, z_min: float = 0.5,
                 depth_min: float = None, depth_max: float = None):
        self._left = left
        self._right = right
        # Needed parameter for the calibration left-right
        cameraMatrix1 = left.intrinsics[0, :3, :3].cpu().numpy()
        distCoeffs1 = np.array([0, 0, 0, 0])
        cameraMatrix2 = right.intrinsics[0, :3, :3].cpu().numpy()
        distCoeffs2 = np.array([0, 0, 0, 0])
        self._shape_left = np.array(left.im_calib.shape[-2:])
        self._shape_right = np.array(right.im_calib.shape[-2:])
        self._new_shape = self.shape_left if self.shape_left[0] * self.shape_left[1] > self.shape_right[0] * \
                                             self.shape_right[1] else self.shape_right
        self.scale = 0.25
        self.dx = int(self.scale * self._new_shape[1])
        self.dy = int(self.scale * self._new_shape[0])
        self._new_shape = (self.new_shape[0] + self.dy, self.new_shape[1] + self.dx)
        self.scale_left = self._new_shape / self._shape_left
        self.scale_right = self._new_shape / self._shape_right

        relative = relative_transformation(right.extrinsics.inverse(), left.extrinsics.inverse())
        R = relative[0, :3, :3].cpu().numpy()
        T = relative[0, :3, -1].cpu().numpy() + [-1e-8, 0, 0]

        R1, R2, P1, P2, _, _, _ = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                                   (self.new_shape[1], self.new_shape[0]), R, T,
                                                   alpha=-1,
                                                   newImageSize=
                                                   (self.new_shape[1], self.new_shape[0]))

        super(StereoSetup, self).__init__(Tensor(P1).to(device=device).unsqueeze(0),
                                          Tensor(P2).to(device=device).unsqueeze(0))
        self.depth_min = Tensor([z_min]).to(self.device) if depth_min is None else depth_min
        self.depth_max = Tensor([abs(P2[0, -1] * accuracy)]).to(self.device) if depth_max is None else depth_max
        self._init_maps_(R1, R2, P1, P2)
        self._name = name if name is not None else f'{left.id}&{right.id}'

        self._init_roi_()
        self._ROI = [self.roi_max, self.roi_min]
        self.last_call = {}

    def _init_maps_(self, R1, R2, P1, P2):
        cameraMatrix1 = self.left.intrinsics[0, :3, :3].cpu().numpy()
        distCoeffs1 = np.array([0, 0, 0, 0])
        cameraMatrix2 = self.right.intrinsics[0, :3, :3].cpu().numpy()
        distCoeffs2 = np.array([0, 0, 0, 0])

        # Transformation map
        self.map_right = np.zeros([2, *self.new_shape])
        self.map_right[0], self.map_right[1] = \
            cv.initUndistortRectifyMap(cameraMatrix2,
                                       distCoeffs2,
                                       R2, P2,
                                       (self.new_shape[1], self.new_shape[0]),
                                       cv.CV_32FC1)

        self.map_left = np.zeros([2, *self.new_shape])
        self.map_left[0], self.map_left[1] = \
            cv.initUndistortRectifyMap(cameraMatrix1,
                                       distCoeffs1,
                                       R1,
                                       P1,
                                       (self.new_shape[1], self.new_shape[0]),
                                       cv.CV_32FC1)

        scale_im_right = cameraMatrix2[0, 0] / P2[0, 0], cameraMatrix2[1, 1] / P2[1, 1]
        scale_im_left = cameraMatrix1[0, 0] / P1[0, 0], cameraMatrix1[1, 1] / P1[1, 1]
        scale = scale_im_left[0] * scale_im_right[0], scale_im_left[1] * scale_im_right[1]

        self.map_right[0] = 2 * (self.map_right[0] / scale_im_right[1] - self.new_shape[1] / 2 / scale[1]) / (
                self.new_shape[1] / (1 + self.scale))
        self.map_right[1] = 2 * (self.map_right[1] / scale_im_right[0] - self.new_shape[0] / 2 / scale[0]) / (
                self.new_shape[0] / (1 + self.scale))
        self.map_left[0] = 2 * (self.map_left[0] / scale_im_left[1] - self.new_shape[1] / 2 / scale[1]) / (
                self.new_shape[1] / (1 + self.scale))
        self.map_left[1] = 2 * (self.map_left[1] / scale_im_left[0] - self.new_shape[0] / 2 / scale[0]) / (
                self.new_shape[0] / (1 + self.scale))

        self.map_right = Tensor(self.map_right).permute(1, 2, 0).unsqueeze(0).to(self.device)
        self.map_left = Tensor(self.map_left).permute(1, 2, 0).unsqueeze(0).to(self.device)

    def _init_map_inv_(self):
        n_h, n_w = self.new_shape
        l_h, l_w = self.shape_left
        r_h, r_w = self.shape_right
        dx, dy = (n_w - l_w) / 2, (n_h - l_h) / 2
        self.pts_left_ori = FloatTensor([(dx, dy), (l_w + dx, dy), (dx, l_h + dy), (l_w + dx, l_h + dy)]).unsqueeze(0)
        dx, dy = (n_w - r_w) / 2, (n_h - r_h) / 2
        self.pts_right_ori = FloatTensor([(dx, dy), (r_w + dx, dy), (dx, r_h + dy), (r_w + dx, r_h + dy)]).unsqueeze(0)
        self.homography_l = get_perspective_transform(self.pts_left, self.pts_left_ori).to(self.device)
        self.homography_r = get_perspective_transform(self.pts_right, self.pts_right_ori).to(self.device)

    def _init_roi_(self):
        pass
        mask_left = (1 >= self.map_left) * (self.map_left >= -1) * torch.ones_like(self.map_left)
        mask_right = (1 >= self.map_right) * (self.map_right >= -1) * torch.ones_like(self.map_right)
        mask_left = mask_left[:, :, :, 0] * mask_left[:, :, :, 1]
        mask_right = mask_right[:, :, :, 0] * mask_right[:, :, :, 1]
        self.roi_min, self.roi_max, self.pts_left, self.pts_right = extract_roi_from_map(mask_left, mask_right)
        self.pts_left, self.pts_right = self.pts_left.unsqueeze(0), self.pts_right.unsqueeze(0)
        self.current_roi_min, self.current_roi_max = self.roi_min.copy(), self.roi_max.copy()
        self._init_map_inv_()

    def __call__(self, sample, *args, reverse=False, cut_roi_min=False, cut_roi_max=False, **kwargs):
        if cut_roi_min:
            cut_roi_max = False
        if not reverse:
            self.last_call = {'cut_roi_min': cut_roi_min, 'cut_roi_max': cut_roi_max}
            left = self.left.id
            right = self.right.id
            new_sample = {}
            for key, im in sample.items():
                if key == left:
                    temp = grid_sample(im, self.map_left, align_corners=True)
                    # temp[temp==0] = im.min() if im.min() >=0 else im.max()
                    if cut_roi_min:
                        temp = self.cut_to_roi(temp, self.roi_min)
                    elif cut_roi_max:
                        temp = self.cut_to_roi(temp, self.roi_max)
                    new_sample['left'] = temp
                elif key == right:
                    temp = grid_sample(im, self.map_right, align_corners=True)
                    # temp[temp == 0] = im.min() if im.min() >=0 else im.max()
                    if cut_roi_min:
                        temp = self.cut_to_roi(temp, self.roi_min)
                    elif cut_roi_max:
                        temp = self.cut_to_roi(temp, self.roi_max)
                    new_sample['right'] = temp
        else:
            new_sample = self._reversed_call_(sample, *args)
        return new_sample

    def _reversed_call_(self, sample, *args):
        left = self.left.id
        right = self.right.id
        new_sample = {}
        cut_roi_min = self.last_call['cut_roi_min']
        cut_roi_max = self.last_call['cut_roi_max']
        for key, im in sample.items():
            if key == 'left':
                if cut_roi_min:
                    temp = self.expand_from_roi(im, self.roi_min)
                elif cut_roi_max:
                    temp = self.expand_from_roi(im, self.roi_max)
                else:
                    temp = im.clone()
                temp = warp_perspective(temp, self.homography_l, self.new_shape)
                new_sample[left] = self.cut_to_original(temp)
            elif key == 'right':
                if cut_roi_min:
                    temp = self.expand_from_roi(im, self.roi_min, side='right')
                elif cut_roi_max:
                    temp = self.expand_from_roi(im, self.roi_max, side='right')
                else:
                    temp = im.clone()
                temp = warp_perspective(temp, self.homography_r, self.new_shape)
                new_sample[right] = self.cut_to_original(temp, side='right')
        return new_sample

    def disparity_to_depth(self, sample: dict, *args):
        for key, t in sample.items():
            mask = t == 0
            t = self.reproject_disparity_to_3D((t + 1e-8).put_channel_at(-1))
            t = t[:, :, :, -1].unsqueeze(1)
            sample[key] = torch.clip(t, self.depth_min, self.depth_max)
            sample[key].pass_attr(t)
            sample[key].max_value = self.depth_max
            sample[key].min_value = self.depth_min
            sample[key].scaled = True
        return sample

    def depth_to_disparity(self, depth, *args):
        mask = depth == 0
        t_ = depth.clone()
        t_[mask] = 1
        disp = self.tx / t_ * self.fx
        disp.max_value = disp.max()
        disp.min_value = disp.min()
        disp[mask] = 0
        disp.scaled = True
        return disp

    @staticmethod
    def cut_to_roi(im, roi):
        return im[:, :, roi[0]:roi[2], roi[1]:roi[3]]

    def cut_to_original(self, im, side='left'):
        if side == 'left':
            n_s = self.new_shape
            s = self.shape_left
        else:
            n_s = self.new_shape
            s = self.shape_right
        dy = int((n_s[0] - s[0]) / 2)
        dx = int((n_s[1] - s[1]) / 2)
        if dx > 0 and dy > 0:
            new = im[..., dy:-dy, dx:-dx]
        elif dx > 0:
            new = im[..., dx:-dx]
        elif dy > 0:
            new = im[..., dy:-dy, :]
        else:
            new = im.clone()
        return new

    def expand_from_roi(self, im, roi, side='left'):
        if side == 'left':
            n_s = self.new_shape
        else:
            n_s = self.new_shape
        temp = im.__class__(torch.zeros(1, im.shape[1], *n_s, device=self.device))
        temp.pass_attr(im)
        temp[..., roi[0]:roi[2], roi[1]:roi[3]] = im
        return temp

    def adjust_current_roi(self, roi):
        self.current_roi_max[0] = self.current_roi_max[0] - roi[0]
        self.current_roi_max[1] = self.current_roi_max[1] - roi[1]
        self.current_roi_max[2] = self.current_roi_max[2] - roi[0]
        self.current_roi_max[3] = self.current_roi_max[3] - roi[1]
        self.current_roi_min[0] = self.current_roi_min[0] - roi[0]
        self.current_roi_min[1] = self.current_roi_min[1] - roi[1]
        self.current_roi_min[2] = self.current_roi_min[2] - roi[0]
        self.current_roi_min[3] = self.current_roi_min[3] - roi[1]

    def show_image_calibration(self):
        im_left = self.left.im_calib
        im_right = self.right.im_calib
        sample = {self.left.id: im_left, self.right.id: im_right}
        sample_1 = self(sample)
        # sample_2 = self(sample_1, reverse=True)
        sample_1['right'].show(num='right image', point=self.pts_right)
        sample_1['left'].show(num='left image', point=self.pts_left)
        (sample_1['right'] * 0.5 + sample_1['left'] * 0.5).show(roi=[self.current_roi_max, self.current_roi_min])


    @property
    def shape_left(self):
        return self._shape_left

    @property
    def shape_right(self):
        return self._shape_right

    @property
    def new_shape(self):
        return self._new_shape

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def name(self):
        return self._name

    @property
    def ROI(self):
        return self._ROI

    @property
    def roi(self):
        return self._ROI


class DepthSetup:
    _ref = None
    _target = None
    _name = 'DepthSetup'
    _pose = torch
    _intrinsics = torch.eye(3)
    _f = 10e-3
    _shape = 0, 0
    _depth_min = 0
    _depth_max = 10

    def __init__(self, ref: Union[IRCamera, RGBCamera], target: Union[IRCamera, RGBCamera],
                 device: torch.device, name: str = None, depth_min: float = None, depth_max: float = None):
        assert ref.im_type == target.im_type
        self.device = device
        self._intrinsics = ref.intrinsics[:, :3, :3].to(torch.float32)
        self._pose = relative_transformation(target.extrinsics.inverse(), ref.extrinsics.inverse()).to(torch.float32)
        self._ref = ref
        self._target = target
        self._f = ref.f
        self._depth_max = depth_max
        self._depth_min = depth_min
        self._shape = ref.im_calib.shape[-2:]
        self._name = name if name is not None else f'{ref.id}&{target.id}'

    def __call__(self, sample, *args, reverse=False, **kwargs):
        if not reverse:
            return {'sample': {'ref': sample[self.ref.id].clone(), 'target': sample[self.target.id].clone()},
                    'intrinsics': self.intrinsics,
                    'pose': self._pose,
                    'focal': self.ref.f,
                    'depth': True}
        else:
            res = {}
            res.update({self.ref.id: self.clip(sample['ref'])}) if 'ref' in sample.keys()\
                else res.update({self.target.id: self.clip(sample['target'])})
            # res.update({self.target.id: sample['target']}) if 'target' in sample.keys() else res.update({self.ref.id: sample['ref']})
            return res

    def clip(self, depth):
        depth = depth.clip(self.depth_min, self.depth_max)
        depth.max_value = self.depth_max
        depth.min_value = self.depth_min
        return depth

    @property
    def ref(self):
        return self._ref

    @property
    def depth_max(self):
        return self._depth_max

    @property
    def depth_min(self):
        return self._depth_min

    @property
    def target(self):
        return self._target

    @property
    def name(self):
        return self._name

    @property
    def pose(self):
        return self._pose

    @property
    def intrinsics(self):
        return self._intrinsics

    @property
    def f(self):
        return self._f

    @property
    def shape(self):
        return self._shape
