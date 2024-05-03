from __future__ import division

from torch import from_numpy, cat
from torch.cuda import FloatTensor
import numpy as np
from torchvision.transforms.functional import hflip
import torch.nn.functional as F1


class Compose(object):
    def __init__(self, transforms: list, device):
        self.transforms = transforms
        no_pad = True
        no_resize = True
        for t in self.transforms:
            if isinstance(t, Resize):
                no_resize = False
            if isinstance(t, Pad):
                no_pad = False
        if not no_pad and not no_resize:
            raise AttributeError("There cannot be a Resize AND a Padding for Pre-processing")
        self.device = device

    def __call__(self, sample, *args, **kwargs):
        for t in self.transforms:
            sample = t(sample, self.device, **kwargs)
        return sample


class Pad:
    def __init__(self, inference_size, keep_ratio=False):
        self.keep_ratio = keep_ratio
        self.pad = []
        self.inference_size = inference_size
        self.ori_size = []

    def __call__(self, sample, device, *args, **kwargs):
        for key in sample.keys():
            h, w = sample[key].shape[-2:]
            self.ori_size.append([h, w])
            if self.keep_ratio:
                while h > self.inference_size[0] or w > self.inference_size[1]:
                    sample[key] = sample[key].pyrDown()
                    h, w = sample[key].shape[-2:]
            else:
                if h > self.inference_size[0] or w > self.inference_size[1]:
                    if h / self.inference_size[0] >= w / self.inference_size[1]:
                        w = w * self.inference_size[0] / h
                        h = self.inference_size[0]
                    else:
                        h = h * self.inference_size[1] / w
                        w = self.inference_size[1]
                        sample[key] = F1.interpolate(sample[key], size=[int(h), int(w)],
                                                     mode='bilinear',
                                                     align_corners=True)
            self._pad(int(h), int(w))
            sample[key] = sample[key].pad(self.pad[-1])
        return sample

    @property
    def inference_size(self):
        return self._inference_size

    @inference_size.setter
    def inference_size(self, value):
        self._inference_size = value
        self.ori_size = []
        self.pad = []

    def _pad(self, h: int, w: int):
        """
        The pad method modify the parameter pad of the Pad object to put a list :
        [pad_left, pad_top, pad_right, pad_bottom]
        :param h: Current height of the image
        :param w: Current width of the image
        :return: Nothing, modify the attribute "pad" of the object
        """
        pad_h = (self.inference_size[0] - h) / 2
        t_pad = pad_h if pad_h % 1 == 0 else pad_h + 0.5
        b_pad = pad_h if pad_h % 1 == 0 else pad_h - 0.5
        pad_w = (self.inference_size[1] - w) / 2
        l_pad = pad_w if pad_w % 1 == 0 else pad_w + 0.5
        r_pad = pad_w if pad_w % 1 == 0 else pad_w - 0.5
        self.pad.append([int(l_pad), int(r_pad), int(t_pad), int(b_pad)])


class Unpad:
    def __init__(self, pad, ori_size):
        self.pad = pad
        self.ori_size = ori_size

    def __call__(self, disp, device, *args, size=0, **kwargs):
        h, w = disp.shape[-2:]
        size = size if size is not None else self.ori_size
        l_pad, r_pad, t_pad, b_pad = self.pad[0]
        disp = disp[:, :, :, l_pad:] if l_pad > 0 else disp
        disp = disp[:, :, :, :-r_pad] if r_pad > 0 else disp
        disp = disp[:, :, t_pad:, :] if t_pad > 0 else disp
        disp = disp[:, :, :-b_pad, :] if b_pad > 0 else disp
        self.pad.pop(0)
        if size == 0:
            pass
        elif h != size[0] or w != size[1]:
            disp = F1.interpolate(disp, size=[size[0], size[1]],
                                  mode='bilinear',
                                  align_corners=True)
            disp *= size[1] / w
        return disp


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean=None, std=None):
        if isinstance(mean, list):
            self.mean = mean
        else:
            self.mean = [0, 0, 0]
        if isinstance(mean, list):
            self.std = std
        else:
            self.std = [0, 0, 0]

    def __call__(self, sample, *args, **kwargs):

        for key in sample.keys():
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)
        return sample


class Resize(object):
    """Resize image, with type tensor"""

    def __init__(self, inference_size, padding_factor):
        self.padding_factor = padding_factor
        self.inference_size = inference_size
        self.ori_size = []

    def __call__(self, sample, *args, **kwargs):
        self.ori_size = []
        key_ref = list(sample.keys())[0]
        if self.inference_size is None:
            if self.padding_factor > 0:
                self.inference_size = [
                    int(np.ceil(sample[key_ref].size(-2) / self.padding_factor)) * self.padding_factor,
                    int(np.ceil(sample[key_ref].size(-1) / self.padding_factor)) * self.padding_factor]
            else:
                pass
            self.inference_size = self.ori_size

        else:
            for key in sample.keys():
                ori_size = sample[key].shape[-2:]
                self.ori_size.append(ori_size)
                if self.inference_size[0] != ori_size[0] or self.inference_size[1] != ori_size[1]:
                    sample[key] = F1.interpolate(sample[key], size=self.inference_size,
                                                 mode='bilinear',
                                                 align_corners=True)
        return sample


class ResizeDepth(object):
    """Resize Disparity image, with type tensor"""

    def __init__(self, size):
        if size:
            self.size = size
        else:
            self.size = 0

    def __call__(self, depth, device, *args, size=None, **kwargs):
        h, w = depth.shape[-2:]
        size = size if size is not None else self.size
        if size == 0:
            pass
        elif h != size[0] or w != size[1]:
            # resize back
            return F1.interpolate(depth, size=size,
                                  mode='bilinear',
                                  align_corners=True)  # [1, H, W]
        else:
            return depth


class ResizeDisp(object):
    """Resize Disparity image, with type tensor"""

    def __init__(self, size):
        if size:
            self.size = size
        else:
            self.size = 0

    def __call__(self, disp, device, *args, size=None, **kwargs):
        h, w = disp.shape[-2:]
        size = size if size is not None else self.size
        if size == 0:
            pass
        elif h != size[0] or w != size[1]:
            # resize back
            disp = F1.interpolate(disp, size=size,
                                  mode='bilinear',
                                  align_corners=True)  # [1, H, W]
            return disp * size[1] / float(w)
        else:
            return disp


class DispSide(object):
    """Transform an image to get the disparity on the good side, with type tensor"""

    def __init__(self, disp_right, disp_bidir):
        self.disp_right = disp_right
        self.disp_bidir = disp_bidir

    def __call__(self, sample, *args, **kwargs):
        if self.disp_right:
            sample["left"], sample["right"] = hflip(sample["right"]), hflip(sample["left"])
        elif self.disp_bidir:
            new_left, new_right = hflip(sample["right"]), hflip(sample["left"])
            sample["left"] = cat((sample["left"], new_left), dim=0)
            sample["right"] = cat((sample["right"], new_right), dim=0)
        return sample


class ToTensor(object):
    """Convert numpy array to torch tensor and load it on the specified device"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample, device, *args, **kwargs):
        if isinstance(sample, dict):
            for key in sample.keys():
                sample[key] = np.transpose(sample[key], (2, 0, 1))  # [C, H, W]
                if self.no_normalize:
                    sample[key] = from_numpy(sample[key])
                else:
                    sample[key] = from_numpy(sample[key]) / 255.
                sample[key] = sample[key].to(device).unsqueeze(0)
        else:
            sample = np.transpose(sample, (2, 0, 1))
            sample = from_numpy(sample) / 255.
            sample = sample.to(device).unsqueeze(0)
        return sample


class ToFloatTensor(object):
    """Convert numpy array to torch tensor and load it on the specified device"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample, device, *args, **kwargs):
        if isinstance(sample, dict):
            for key in sample.keys():
                sample[key] = sample[key].to(FloatTensor)
                # sample[key] = np.transpose(sample[key], (2, 0, 1))  # [C, H, W]
                # if self.no_normalize:
                #     sample[key] = torch.cuda.FloatTensor(sample[key])
                # else:
                #     sample[key] = torch.cuda.FloatTensor(sample[key] / 255.)
                # sample[key] = sample[key].to(device).unsqueeze(0)
        else:
            sample = sample.to(FloatTensor)
        return sample

# class PerspectiveTransform:
#     """Register an image to another one following the perspective Transform matrix"""
#     _matrix = None
#     _inverse_matrix = None
#     _shape = None
#     _center = None
#     _map = None
#     _inverse_map = None
#
#     def __init__(self, matrix, shape, device, center=()):
#         self.device = device
#         self.update(matrix, shape, center)
#
#     def __call__(self, image, image_ref, reverse=False):
#         pass
#
#     def __mul__(self, other):
#         matrix = other.inverse_matrix @ self.matrix
#         return PerspectiveTransform(matrix, self.shape, self.device, self.center)
#
#     def __truediv__(self, other):
#         matrix = other.matrix @ self.inverse_matrix
#         return PerspectiveTransform(matrix, self.shape, self.device, self.center)
#
#     def __add__(self, other):
#         return self * other
#
#     def __sub__(self, other):
#         return self / other
#
#     def update_matrix(self, matrix):
#         self._matrix = Tensor(matrix).to(self.device)  # [3, 3]
#         i_matrix = torch.linalg.inv(self.matrix)  # [3, 3]
#         self._inverse_matrix = Tensor(i_matrix).to(self.device)  # [3, 3]
#
#     def update_shape(self, shape):
#         self._shape = Tensor(shape).to(self.device)
#
#     def update_center(self, center):
#         c = center if center else (int(self.shape[0] / 2), int(self.shape[1] / 2))
#         self._center = Tensor(c).to(self.device)
#
#     def update(self, matrix, shape, center=None):
#         self.update_shape(shape)
#         self.update_matrix(matrix)
#         self.update_center(center)
#         self._init_map_()
#
#     def _init_map_(self):
#         height, width = self.shape[0], self.shape[1]
#         grid_reg = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False,
#                                                 device=self.device)  # [1 H W 2]
#         grid_reg[:, :, :, 0] = grid_reg[:, :, :, 0] - self.center[1]
#         grid_reg[:, :, :, 1] = grid_reg[:, :, :, 1] - self.center[0]
#         z = torch.ones_like(grid_reg[:, :, :, 0])
#         grid = torch.stack([grid_reg, z], dim=-1)  # [1 H W 3]
#
#         grid_transformed = (self.matrix @ grid.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # [1 H W 3]
#         alpha = grid_transformed[:, :, :, 2]
#         grid_transformed[:, :, :, 0] = 2 * (grid_transformed[:, :, :, 0] / alpha) / width - 1  # [1 H W 3]
#         grid_transformed[:, :, :, 1] = 2 * (grid_transformed[:, :, :, 1] / alpha) / height - 1  # [1 H W 3]
#         self._map = grid_transformed  # [1 H W 3]
#
#         grid_transformed_inv = (self.inverse_matrix @ grid.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # [1 H W 3]
#         grid_transformed_inv[:, :, :, 0] = 2 * (
#                 grid_transformed_inv[:, :, :, 0] / grid_transformed_inv[:, :, :, 2]) / width - 1  # [1 H W 3]
#         grid_transformed_inv[:, :, :, 1] = 2 * (
#                 grid_transformed_inv[:, :, :, 1] / grid_transformed_inv[:, :, :, 2]) / height - 1  # [1 H W 3]
#         self._inverse_map = grid_transformed_inv  # [1 H W 3]
#
#     @property
#     def map(self):
#         return self._map
#
#     # @map.setter
#     # def map(self, value):
#     #     """Return the calling function's name."""
#     #     # Ref: https://stackoverflow.com/a/57712700/
#     #     name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
#     #     if name == '_init_map_' or name == '__new__':
#     #         self.map = value
#     #
#     # @map.deleter
#     # def map(self):
#     #     warnings.warn("The attribute can't be deleted")
#
#     @property
#     def inverse_map(self):
#         return self._inverse_map
#
#     #
#     # @inverse_map.setter
#     # def inverse_map(self, value):
#     #     """Return the calling function's name."""
#     #     # Ref: https://stackoverflow.com/a/57712700/
#     #     name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
#     #     if name == '_init_map_' or name == '__new__':
#     #         self.inverse_map = value
#
#     @inverse_map.deleter
#     def inverse_map(self):
#         warnings.warn("The attribute can't be deleted")
#
#     @property
#     def matrix(self):
#         return self._matrix
#
#     # @matrix.setter
#     # def matrix(self, value):
#     #     """Return the calling function's name."""
#     #     # Ref: https://stackoverflow.com/a/57712700/
#     #     name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
#     #     if name == 'update_matrix' or name == '__new__':
#     #         self.matrix = value
#     #
#     # @matrix.deleter
#     # def matrix(self):
#     #     warnings.warn("The attribute can't be deleted")
#
#     @property
#     def inverse_matrix(self):
#         return self._matrix
#
#     # @inverse_matrix.setter
#     # def inverse_matrix(self, value):
#     #     """Return the calling function's name."""
#     #     # Ref: https://stackoverflow.com/a/57712700/
#     #     name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
#     #     if name == 'update_matrix' or name == '__new__':
#     #         self.inverse_matrix = value
#     #
#     # @inverse_matrix.deleter
#     # def inverse_matrix(self):
#     #     warnings.warn("The attribute can't be deleted")
#
#     @property
#     def shape(self):
#         return self._shape
#
#     #
#     # @shape.setter
#     # def shape(self, value):
#     #     """Return the calling function's name."""
#     #     # Ref: https://stackoverflow.com/a/57712700/
#     #     name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
#     #     if name == 'update_shape' or name == '__new__':
#     #         self.shape = value
#     #
#     # @shape.deleter
#     # def shape(self):
#     #     warnings.warn("The attribute can't be deleted")
#
#     @property
#     def center(self):
#         return self._center
#
#     # @center.setter
#     # def center(self, value):
#     #     """Return the calling function's name."""
#     #     # Ref: https://stackoverflow.com/a/57712700/
#     #     name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
#     #     if name == 'update_center' or name == '__new__':
#     #         self.center = value
#     #
#     # @center.deleter
#     # def center(self):
#     #     warnings.warn("The attribute can't be deleted")

#
#
# class RandomCrop(object):
#     def __init__(self, img_height, img_width):
#         self.img_height = img_height
#         self.img_width = img_width
#
#     def __call__(self, sample):
#         ori_height, ori_width = sample['left'].shape[:2]
#
#         # pad zero when crop size is larger than original image size
#         if self.img_height > ori_height or self.img_width > ori_width:
#
#             # can be used for only pad one side
#             top_pad = max(self.img_height - ori_height, 0)
#             right_pad = max(self.img_width - ori_width, 0)
#
#             # try edge padding
#             sample['left'] = np.lib.pad(sample['left'],
#                                         ((top_pad, 0), (0, right_pad), (0, 0)),
#                                         mode='edge')
#             sample['right'] = np.lib.pad(sample['right'],
#                                          ((top_pad, 0), (0, right_pad), (0, 0)),
#                                          mode='edge')
#
#             if 'disp' in sample.keys():
#                 sample['disp'] = np.lib.pad(sample['disp'],
#                                             ((top_pad, 0), (0, right_pad)),
#                                             mode='constant',
#                                             constant_values=0)
#
#             # update image resolution
#             ori_height, ori_width = sample['left'].shape[:2]
#
#         assert self.img_height <= ori_height and self.img_width <= ori_width
#
#         # Training: random crop
#         self.offset_x = np.random.randint(ori_width - self.img_width + 1)
#
#         start_height = 0
#         assert ori_height - start_height >= self.img_height
#
#         self.offset_y = np.random.randint(start_height, ori_height - self.img_height + 1)
#
#         sample['left'] = self.crop_img(sample['left'])
#         sample['right'] = self.crop_img(sample['right'])
#         if 'disp' in sample.keys():
#             sample['disp'] = self.crop_img(sample['disp'])
#
#         return sample
#
#     def crop_img(self, img):
#         return img[self.offset_y:self.offset_y + self.img_height,
#                self.offset_x:self.offset_x + self.img_width]
#
#
# class RandomVerticalFlip(object):
#     """Randomly vertically filps"""
#
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             sample['left'] = np.copy(np.flipud(sample['left']))
#             sample['right'] = np.copy(np.flipud(sample['right']))
#
#             sample['disp'] = np.copy(np.flipud(sample['disp']))
#
#         return sample
#
#
# class ToPILImage(object):
#
#     def __call__(self, sample):
#         sample['left'] = Image.fromarray(sample['left'].astype('uint8'))
#         sample['right'] = Image.fromarray(sample['right'].astype('uint8'))
#
#         return sample
#
#
# class ToNumpyArray(object):
#
#     def __call__(self, sample):
#         sample['left'] = np.array(sample['left']).astype(np.float32)
#         sample['right'] = np.array(sample['right']).astype(np.float32)
#
#         return sample
#
#
# # Random coloring
# class RandomContrast(object):
#     """Random contrast"""
#
#     def __init__(self,
#                  asymmetric_color_aug=True,
#                  ):
#
#         self.asymmetric_color_aug = asymmetric_color_aug
#
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             contrast_factor = np.random.uniform(0.8, 1.2)
#
#             sample['left'] = F.adjust_contrast(sample['left'], contrast_factor)
#
#             if self.asymmetric_color_aug and np.random.random() < 0.5:
#                 contrast_factor = np.random.uniform(0.8, 1.2)
#
#             sample['right'] = F.adjust_contrast(sample['right'], contrast_factor)
#
#         return sample
#
#
# class RandomGamma(object):
#
#     def __init__(self,
#                  asymmetric_color_aug=True,
#                  ):
#
#         self.asymmetric_color_aug = asymmetric_color_aug
#
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet
#
#             sample['left'] = F.adjust_gamma(sample['left'], gamma)
#
#             if self.asymmetric_color_aug and np.random.random() < 0.5:
#                 gamma = np.random.uniform(0.7, 1.5)  # adopted from FlowNet
#
#             sample['right'] = F.adjust_gamma(sample['right'], gamma)
#
#         return sample
#
#
# class RandomBrightness(object):
#
#     def __init__(self,
#                  asymmetric_color_aug=True,
#                  ):
#
#         self.asymmetric_color_aug = asymmetric_color_aug
#
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             brightness = np.random.uniform(0.5, 2.0)
#
#             sample['left'] = F.adjust_brightness(sample['left'], brightness)
#
#             if self.asymmetric_color_aug and np.random.random() < 0.5:
#                 brightness = np.random.uniform(0.5, 2.0)
#
#             sample['right'] = F.adjust_brightness(sample['right'], brightness)
#
#         return sample
#
#
# class RandomHue(object):
#
#     def __init__(self,
#                  asymmetric_color_aug=True,
#                  ):
#
#         self.asymmetric_color_aug = asymmetric_color_aug
#
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             hue = np.random.uniform(-0.1, 0.1)
#
#             sample['left'] = F.adjust_hue(sample['left'], hue)
#
#             if self.asymmetric_color_aug and np.random.random() < 0.5:
#                 hue = np.random.uniform(-0.1, 0.1)
#
#             sample['right'] = F.adjust_hue(sample['right'], hue)
#
#         return sample
#
#
# class RandomSaturation(object):
#
#     def __init__(self,
#                  asymmetric_color_aug=True,
#                  ):
#
#         self.asymmetric_color_aug = asymmetric_color_aug
#
#     def __call__(self, sample):
#         if np.random.random() < 0.5:
#             saturation = np.random.uniform(0.8, 1.2)
#
#             sample['left'] = F.adjust_saturation(sample['left'], saturation)
#
#             if self.asymmetric_color_aug and np.random.random() < 0.5:
#                 saturation = np.random.uniform(0.8, 1.2)
#
#             sample['right'] = F.adjust_saturation(sample['right'], saturation)
#
#         return sample
#
#
# class RandomColor(object):
#
#     def __init__(self,
#                  asymmetric_color_aug=True,
#                  ):
#
#         self.asymmetric_color_aug = asymmetric_color_aug
#
#     def __call__(self, sample):
#         transforms = [RandomContrast(asymmetric_color_aug=self.asymmetric_color_aug),
#                       RandomGamma(asymmetric_color_aug=self.asymmetric_color_aug),
#                       RandomBrightness(asymmetric_color_aug=self.asymmetric_color_aug),
#                       RandomHue(asymmetric_color_aug=self.asymmetric_color_aug),
#                       RandomSaturation(asymmetric_color_aug=self.asymmetric_color_aug)]
#
#         sample = ToPILImage()(sample)
#
#         if np.random.random() < 0.5:
#             # A single transform
#             t = random.choice(transforms)
#             sample = t(sample)
#         else:
#             # Combination of transforms
#             # Random order
#             random.shuffle(transforms)
#             for t in transforms:
#                 sample = t(sample)
#
#         sample = ToNumpyArray()(sample)
#
#         return sample
#
#
# class RandomScale(object):
#     def __init__(self,
#                  min_scale=-0.4,
#                  max_scale=0.4,
#                  crop_width=512,
#                  nearest_interp=False,  # for sparse gt
#                  ):
#         self.min_scale = min_scale
#         self.max_scale = max_scale
#         self.crop_width = crop_width
#         self.nearest_interp = nearest_interp
#
#     def __call__(self, sample):
#         if np.random.rand() < 0.5:
#             h, w = sample['disp'].shape
#
#             scale_x = 2 ** np.random.uniform(self.min_scale, self.max_scale)
#
#             scale_x = np.clip(scale_x, self.crop_width / float(w), None)
#
#             # only random scale x axis
#             sample['left'] = cv2.resize(sample['left'], None, fx=scale_x, fy=1., interpolation=cv2.INTER_LINEAR)
#             sample['right'] = cv2.resize(sample['right'], None, fx=scale_x, fy=1., interpolation=cv2.INTER_LINEAR)
#
#             sample['disp'] = cv2.resize(
#                 sample['disp'], None, fx=scale_x, fy=1.,
#                 interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
#             ) * scale_x
#
#             if 'pseudo_disp' in sample and sample['pseudo_disp'] is not None:
#                 sample['pseudo_disp'] = cv2.resize(sample['pseudo_disp'], None, fx=scale_x, fy=1.,
#                                                    interpolation=cv2.INTER_LINEAR) * scale_x
#
#         return sample
#
#
# class Resize(object):
#     def __init__(self,
#                  scale_x=1,
#                  scale_y=1,
#                  nearest_interp=True,  # for sparse gt
#                  ):
#         """
#         Resize low-resolution data to high-res for mixed dataset training
#         """
#         self.scale_x = scale_x
#         self.scale_y = scale_y
#         self.nearest_interp = nearest_interp
#
#     def __call__(self, sample):
#         scale_x = self.scale_x
#         scale_y = self.scale_y
#
#         sample['left'] = cv2.resize(sample['left'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#         sample['right'] = cv2.resize(sample['right'], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
#
#         sample['disp'] = cv2.resize(
#             sample['disp'], None, fx=scale_x, fy=scale_y,
#             interpolation=cv2.INTER_LINEAR if not self.nearest_interp else cv2.INTER_NEAREST
#         ) * scale_x
#
#         return sample
#
#
# class RandomGrayscale(object):
#     def __init__(self, p=0.2):
#         self.p = p
#
#     def __call__(self, sample):
#         if np.random.random() < self.p:
#             sample = ToPILImage()(sample)
#
#             # only supported in higher version pytorch
#             # default output channels is 1
#             sample['left'] = F.rgb_to_grayscale(sample['left'], num_output_channels=3)
#             sample['right'] = F.rgb_to_grayscale(sample['right'], num_output_channels=3)
#
#             sample = ToNumpyArray()(sample)
#
#         return sample
#
#
# class RandomRotateShiftRight(object):
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, sample):
#         if np.random.random() < self.p:
#             angle, pixel = 0.1, 2
#             px = np.random.uniform(-pixel, pixel)
#             ag = np.random.uniform(-angle, angle)
#
#             right_img = sample['right']
#
#             image_center = (
#                 np.random.uniform(0, right_img.shape[0]),
#                 np.random.uniform(0, right_img.shape[1])
#             )
#
#             rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
#             right_img = cv2.warpAffine(
#                 right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
#             )
#             trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
#             right_img = cv2.warpAffine(
#                 right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
#             )
#
#             sample['right'] = right_img
#
#         return sample
#
#
# class RandomOcclusion(object):
#     def __init__(self, p=0.5,
#                  occlusion_mask_zero=False):
#         self.p = p
#         self.occlusion_mask_zero = occlusion_mask_zero
#
#     def __call__(self, sample):
#         bounds = [50, 100]
#         if np.random.random() < self.p:
#             img2 = sample['right']
#             ht, wd = img2.shape[:2]
#
#             if self.occlusion_mask_zero:
#                 mean_color = 0
#             else:
#                 mean_color = np.mean(img2.reshape(-1, 3), axis=0)
#
#             x0 = np.random.randint(0, wd)
#             y0 = np.random.randint(0, ht)
#             dx = np.random.randint(bounds[0], bounds[1])
#             dy = np.random.randint(bounds[0], bounds[1])
#             img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
#
#             sample['right'] = img2
#
#         return sample
