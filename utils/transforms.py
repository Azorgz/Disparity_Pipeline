from __future__ import division
import torch
import numpy as np
import torchvision.transforms.functional as F
from torchvision.transforms.functional import hflip
import torch.nn.functional as F1

from utils.classes.Image import ImageCustom


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

    def __call__(self, sample):
        if not isinstance(sample, F.Tensor):
            sample_ = sample.copy()
        else:
            sample_ = torch.tensor(sample)
        for t in self.transforms:
            sample_ = t(sample_, self.device)
        return sample_


class ToTensor(object):
    """Convert numpy array to torch tensor and load it on the specified device"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample, device, *args):
        if isinstance(sample, dict):
            for key in sample.keys():
                sample[key] = np.transpose(sample[key], (2, 0, 1))  # [C, H, W]
                if self.no_normalize:
                    sample[key] = torch.from_numpy(sample[key])
                else:
                    sample[key] = torch.from_numpy(sample[key]) / 255.
                sample[key] = sample[key].to(device).unsqueeze(0)
        else:
            sample = np.transpose(sample, (2, 0, 1))
            sample = torch.from_numpy(sample) / 255.
            sample = sample.to(device).unsqueeze(0)
        return sample


class ToFloatTensor(object):
    """Convert numpy array to torch tensor and load it on the specified device"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample, device, *args):
        if isinstance(sample, dict):
            for key in sample.keys():
                sample[key] = np.transpose(sample[key], (2, 0, 1))  # [C, H, W]
                if self.no_normalize:
                    sample[key] = torch.cuda.FloatTensor(sample[key])
                else:
                    sample[key] = torch.cuda.FloatTensor(sample[key] / 255.)
                sample[key] = sample[key].to(device).unsqueeze(0)
        else:
            sample = np.transpose(sample, (2, 0, 1))
            sample = torch.cuda.FloatTensor(sample / 255.)
            sample = sample.to(device).unsqueeze(0)
        return sample


class ToNumpy(object):
    """Convert torch tensor to a numpy array loading it from the specified device"""

    def __init__(self, normalize=True, permute=True):
        self.normalize = normalize
        self.permute = permute

    def __call__(self, sample, *args):
        if isinstance(sample, dict):
            for key in sample.keys():
                if not isinstance(sample[key], np.ndarray or ImageCustom):
                    if self.permute and len(sample[key].shape) == 3:
                        sample[key] = sample[key].squeeze().permute(1, 2, 0)  # [C, H, W]
                    if not self.normalize:
                        sample[key] = sample[key].cpu().numpy()
                    else:
                        sample[key] = (sample[key].cpu().numpy() * 255).astype("uint8")
        else:
            if not isinstance(sample, np.ndarray or ImageCustom):
                sample = sample.squeeze()
                if self.permute and len(sample.shape) == 3:
                    sample = sample.squeeze().permute(1, 2, 0)
                sample = sample.cpu().numpy()
            if self.normalize:
                sample = (sample * 255).astype("uint8")
        return sample

class Pad:
    def __init__(self, shape, keep_ratio=False):
        self.keep_ratio = keep_ratio
        self.shape = shape
        self.pad = [0, 0, 0, 0]
        self.inference_size = [0, 0]
        self.ori_size = [0, 0]

    def __call__(self, sample, device, *args):
        _, _, h, w = sample['left'].shape
        self.ori_size = [h, w]
        if self.keep_ratio:
            while h > self.shape[0] or w > self.shape[1]:
                sample['left'] = F1.interpolate(sample['left'], size=[round(h * 0.5), round(w * 0.5)],
                                                mode='bilinear',
                                                align_corners=True)
                sample['right'] = F1.interpolate(sample['right'], size=[round(h * 0.5), round(w * 0.5)],
                                                 mode='bilinear',
                                                 align_corners=True)
                _, _, h, w = sample['left'].shape
        else:
            if h > self.shape[0] or w > self.shape[1]:
                if h / self.shape[0] >= w / self.shape[1]:
                    w = w * self.shape[0] / h
                    h = self.shape[0]
                else:
                    h = h * self.shape[1] / w
                    w = self.shape[1]
                sample['left'] = F1.interpolate(sample['left'], size=[int(h), int(w)],
                                                mode='bilinear',
                                                align_corners=True)
                sample['right'] = F1.interpolate(sample['right'], size=[int(h), int(w)],
                                                 mode='bilinear',
                                                 align_corners=True)
        self.__pad__(int(h), int(w))
        sample['left'] = F.pad(sample['left'], self.pad, fill=0, padding_mode='edge')
        sample['right'] = F.pad(sample['right'], self.pad, fill=0, padding_mode='edge')
        _, _, h, w = sample['left'].shape
        self.inference_size = [h, w]
        return sample

    def __pad__(self, h: int, w: int):
        """
        The pad method modify the parameter pad of the Pad object to put a list :
        [pad_left, pad_top, pad_right, pad_bottom]
        :param h: Current height of the image
        :param w: Current width of the image
        :return: Nothing, modify the attribute "pad" of the object
        """
        pad_h = (self.shape[0] - h) / 2
        t_pad = pad_h if pad_h % 1 == 0 else pad_h + 0.5
        b_pad = pad_h if pad_h % 1 == 0 else pad_h - 0.5
        pad_w = (self.shape[1] - w) / 2
        l_pad = pad_w if pad_w % 1 == 0 else pad_w + 0.5
        r_pad = pad_w if pad_w % 1 == 0 else pad_w - 0.5
        self.pad = [int(l_pad), int(t_pad), int(r_pad), int(b_pad)]


class Unpad:
    def __init__(self, pad, ori_size):
        self.pad = pad
        self.ori_size = ori_size

    def __call__(self, image, device, *args):
        _, h, w = image.shape
        image = F.crop(image,
                       self.pad[1], self.pad[0],
                       h - self.pad[1] - self.pad[3],
                       w - self.pad[0] - self.pad[2])

        im = F1.interpolate(image.unsqueeze(1), size=[self.ori_size[0], self.ori_size[1]],
                            mode='bilinear',
                            align_corners=True).squeeze(1)
        return im * self.ori_size[1] / float(w)


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

    def __call__(self, sample, *args):

        norm_keys = ['left', 'right', 'other']
        if self.mean == [0, 0, 0]:
            self.mean = torch.squeeze(sample['left']).mean(axis=[1, 2])
        if self.std == [0, 0, 0]:
            self.std = torch.squeeze(sample['left']).std(axis=[1, 2])
        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample


class Resize(object):
    """Resize image, with type tensor"""

    def __init__(self, inference_size, padding_factor):
        self.padding_factor = padding_factor
        self.inference_size = inference_size

    def __call__(self, sample, *args):
        if self.inference_size is None:
            if self.padding_factor > 0:
                self.inference_size = [
                    int(np.ceil(sample["left"].size(-2) / self.padding_factor)) * self.padding_factor,
                    int(np.ceil(sample["left"].size(-1) / self.padding_factor)) * self.padding_factor]
            else:
                pass
        self.ori_size = sample["left"].shape[-2:]
        if self.inference_size is not None:
            if self.inference_size[0] != self.ori_size[0] or self.inference_size[1] != self.ori_size[1]:
                for key in sample.keys():
                    sample[key] = F1.interpolate(sample[key], size=self.inference_size,
                                                mode='bilinear',
                                                align_corners=True)

        else:
            self.inference_size = self.ori_size
        return sample


class Resize_disp(object):
    """Resize Disparity image, with type tensor"""

    def __init__(self, size):
        self.size = size

    def __call__(self, disp, device, *args):
        _, h, w = disp.shape
        if h != self.size[0] or w != self.size[1]:
            # resize back
            disp = F1.interpolate(disp.unsqueeze(1), size=self.size,
                                  mode='bilinear',
                                  align_corners=True).squeeze(1)  # [1, H, W]
            return disp * self.size[1] / float(w)


class DispSide(object):
    """Transform an image to get the disparity on the good side, with type tensor"""

    def __init__(self, disp_right, disp_bidir):
        self.disp_right = disp_right
        self.disp_bidir = disp_bidir

    def __call__(self, sample, *args):
        if self.disp_right:
            sample["left"], sample["right"] = hflip(sample["right"]), hflip(sample["left"])
        elif self.disp_bidir:
            new_left, new_right = hflip(sample["right"]), hflip(sample["left"])
            sample["left"] = torch.cat((sample["left"], new_left), dim=0)
            sample["right"] = torch.cat((sample["right"], new_right), dim=0)
        return sample

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
