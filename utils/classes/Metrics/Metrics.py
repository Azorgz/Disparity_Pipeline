import torch
from torchmetrics import MeanSquaredError as MSE
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
import torch.nn.functional as F

######################### METRIC ##############################################
from utils.gradient_tools import grad_tensor


class BaseMetric_Tensor:
    ##
    # A class defining the general basic metric working with Tensor on GPU

    def __init__(self, device):
        self.device = device
        self.metric = "Base Metric"
        self.value = 0
        self.range_min = 0
        self.range_max = 1
        self.commentary = "Just a base"
        self.ratio_list = torch.tensor([1, 3 / 4, 2 / 3, 9 / 16, 9 / 21])
        self.ratio_dict = {1: [512, 512],
                           round(3 / 4, 3): [480, 640],
                           round(2 / 3, 3): [440, 660],
                           round(9 / 16, 3): [405, 720],
                           round(9 / 21, 3): [340, 800]}

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
        # Input array is a path to an image OR an already formed ndarray instance
        # assert im1.shape[-2:] == im2.shape[-2:], " The inputs are not the same size"
        c, h, w = im1.shape[-3:]
        c1 = im2.shape[1]

        if c == c1:
            self.image_true = im1.clone()
            self.image_test = im2.clone()
        elif c > 1:
            self.image_true = im1.GRAYSCALE()
            self.image_test = im2.clone()
        else:
            self.image_true = im1.clone()
            self.image_test = im2.GRAYSCALE()
        size = self._determine_size_from_ratio()
        self.image_true = F.interpolate(self.image_true, size=size,
                                        mode='bilinear',
                                        align_corners=True)
        self.image_test = F.interpolate(self.image_test, size=size,
                                        mode='bilinear',
                                        align_corners=True)
        if mask is not None:
            self.mask = F.interpolate(mask.to(torch.float32), size=size,
                                      mode='bilinear',
                                      align_corners=True).to(torch.bool)
        self.value = 0

    def _determine_size_from_ratio(self):
        ratio = self.image_true.shape[-2] / self.image_true.shape[-1]
        idx_ratio = round(float(self.ratio_list[torch.argmin((self.ratio_list - ratio) ** 2)]), 3)
        size = self.ratio_dict[float(idx_ratio)]
        return size

    def __add__(self, other):
        assert isinstance(other, BaseMetric_Tensor)
        if self.metric == other.metric:
            self.value = self.value + other.value
        else:
            self.value = self.scale + other.scale
        return self

    @property
    def scale(self):
        return self.value

    def __str__(self):
        ##
        # Redefine the way of printing
        return f"{self.metric} metric : {self.value} | between {self.range_min} and {self.range_max} | {self.commentary}"


class Metric_ssim_tensor(BaseMetric_Tensor):
    def __init__(self, device):
        super().__init__(device)
        self.ssim = SSIM(gaussian_kernel=True,
                         sigma=1.5,
                         kernel_size=11,
                         reduction=None,
                         data_range=None,
                         k1=0.01, k2=0.03,
                         return_full_image=True,
                         return_contrast_sensitivity=False).to(self.device)
        self.metric = "SSIM"
        self.commentary = "The higher, the better"

    def __call__(self, im1, im2, *args, mask=None, return_image=False, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        if mask is None:
            temp, image = self.ssim(self.image_test, self.image_true)
            self.value = torch.abs(image).mean()
            self.ssim.reset()
        else:
            temp, image = self.ssim(self.image_test, self.image_true)
            image = torch.abs(image)
            self.value = image[:, :, self.mask[0, 0, :, :]].mean()
            self.ssim.reset()
        del temp
        # self.value = self.ssim(self.image_test * mask, self.image_true * mask)
        if return_image:
            return image.GRAYSCALE().RGB('gray')
        else:
            return self.value

    def scale(self):
        self.range_max += self.range_max
        return self


class MultiScaleSSIM_tensor(BaseMetric_Tensor):
    def __init__(self, device):
        super().__init__(device)
        self.ms_ssim = MS_SSIM(gaussian_kernel=True,
                               sigma=1.5,
                               kernel_size=11,
                               reduction=None,
                               data_range=None,
                               k1=0.01, k2=0.03,
                               betas=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)).to(self.device)
        self.metric = "Multi Scale SSIM"
        self.commentary = "The higher, the better"

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        if mask is None:
            self.value = self.ms_ssim(self.image_test.to_tensor(), self.image_true.to_tensor())
        else:
            self.value = self.ms_ssim(self.image_test.to_tensor() * self.mask, self.image_true.to_tensor() * self.mask)
            nb_pixel_im = self.image_test.shape[-2] * self.image_test.shape[-1]
            nb_pixel_mask = (~self.mask).to(torch.float32).sum()
            # Remove the perfect SSIM given by the mask
            self.value = (self.value * nb_pixel_im - nb_pixel_mask) / (nb_pixel_im - nb_pixel_mask)
        return self.value

    def scale(self):
        return self.value


class Metric_mse_tensor(BaseMetric_Tensor):
    def __init__(self, device):
        super().__init__(device)
        self.mse = MSE(squared=True).to(self.device)
        self.metric = "MSE"
        self.range_max = 1
        self.commentary = "The lower, the better"

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        if mask is None:
            self.value = self.mse(self.image_true, self.image_test)
        else:
            self.value = self.mse(self.image_true[:, :, self.mask[0, 0, :, :]].flatten(),
                                  self.image_test[:, :, self.mask[0, 0, :, :]].flatten())
        return self.value

    def scale(self):
        self.range_max = 2
        return self.value


class Metric_rmse_tensor(BaseMetric_Tensor):
    def __init__(self, device):
        super().__init__(device)
        self.rmse = MSE(squared=False).to(self.device)
        self.metric = "RMSE"
        self.range_max = 1

    def __call__(self, im1, im2, *args, mask=None, return_image=False, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        if mask is None:
            self.value = self.rmse(self.image_true, self.image_test)
        else:
            self.value = self.rmse(self.image_true[:, :, self.mask[0, 0, :, :]].flatten(),
                                   self.image_test[:, :, self.mask[0, 0, :, :]].flatten())
        return self.value

    def scale(self):
        self.value = super().scale
        return self.value


class Metric_psnr_tensor(BaseMetric_Tensor):
    def __init__(self, device):
        super().__init__(device)
        self.psnr = PSNR(data_range=None, base=10.0, reduction=None, dim=None).to(self.device)
        self.metric = "Peak Signal Noise Ratio"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = "inf"

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        try:
            if mask is None:
                self.value = self.psnr(self.image_true, self.image_test)
            else:
                self.value = self.psnr(self.image_true * self.mask[0, 0, :, :],
                                       self.image_test * self.mask[0, 0, :, :])
        except RuntimeError:
            self.value = -1
        return self.value

    def __add__(self, other):
        assert isinstance(other, BaseMetric_Tensor)
        if self.metric == other.metric:
            self.value = self.value + other.value * self.range_max / other.range_max
            self.range_max += self.range_max
        else:
            self.value = self.scale() + other.scale()
        return self


class Metric_nec_tensor(BaseMetric_Tensor):
    def __init__(self, device):
        super().__init__(device)
        self.metric = "Edges Correlation"
        self.commentary = "The higher, the better"
        self.range_min = 0
        self.range_max = 1

    def __call__(self, im1, im2, *args, mask=None, return_image=False, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        ref_true = grad_tensor(self.image_true)
        ref_test = grad_tensor(self.image_test)
        if mask is not None:
            ref_true = ref_true * self.mask
            ref_test = ref_test * self.mask
        dot_prod = (torch.abs(torch.cos(ref_true[:, 1, :, :] - ref_test[:, 1, :, :])) *
                    ((ref_true[:, 1, :, :] != 0) + (ref_test[:, 1, :, :] != 0)))
        image_nec = (ref_true[:, 0, :, :] * ref_test[:, 0, :, :] * dot_prod)
        nec_ref = torch.sqrt(torch.sum(ref_true[:, 0, :, :] * ref_true[:, 0, :, :]) *
                             torch.sum(ref_test[:, 0, :, :] * ref_test[:, 0, :, :]))
        self.value = image_nec.sum() / nec_ref
        if return_image:
            return image_nec.unsqueeze(0).RGB('gray')
        else:
            return self.value
