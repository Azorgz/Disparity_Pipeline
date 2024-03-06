import torch
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import torch.nn.functional as F1
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
        self.image_true = F1.interpolate(self.image_true, size=size,
                                         mode='bilinear',
                                         align_corners=True)
        self.image_test = F1.interpolate(self.image_test, size=size,
                                         mode='bilinear',
                                         align_corners=True)
        if mask is not None:
            self.mask = F1.interpolate(mask.to(torch.float32), size=size,
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

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        if mask is None:
            self.value, temp = self.ssim(self.image_test, self.image_true)
            self.ssim.reset()
            del temp
        else:
            temp, image = self.ssim(self.image_test, self.image_true)
            self.value = image[:, :, self.mask[0, 0, :, :]].mean()
            self.ssim.reset()
            del temp
            # self.value = self.ssim(self.image_test * mask, self.image_true * mask)
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
            self.value = self.ms_ssim(self.image_test, self.image_true)
        else:
            self.value = self.ms_ssim(self.image_test * self.mask, self.image_true * self.mask)
            nb_pixel_im = self.image_test.shape[-2] * self.image_test.shape[-1]
            nb_pixel_mask = (~self.mask).to(torch.float32).sum()
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

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
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
                self.value = self.psnr(self.image_true[:, :, self.mask[0, 0, :, :]].flatten(),
                                       self.image_test[:, :, self.mask[0, 0, :, :]].flatten())
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

    def __call__(self, im1, im2, *args, mask=None, **kwargs):
        super().__call__(im1, im2, *args, mask=mask, **kwargs)
        ref_true = grad_tensor(self.image_true)
        ref_test = grad_tensor(self.image_test)
        if mask is not None:
            ref_true = ref_true.flatten(start_dim=2)[:, :, self.mask[0, 0, :, :].flatten()]
            ref_test = ref_test.flatten(start_dim=2)[:, :, self.mask[0, 0, :, :].flatten()]
            dot_prod = torch.abs(torch.cos(ref_true[0, 1, :] - ref_test[0, 1, :]))
            self.value = ((ref_true[:, 0, :]*ref_test[:, 0, :] * dot_prod).sum() /
                          torch.sqrt(torch.sum(ref_true[:, 0, :] * ref_true[:, 0, :]) * torch.sum(ref_test[:, 0, :] * ref_test[:, 0, :])))
        else:
            dot_prod = torch.abs(torch.cos(ref_true[:, 1, :, :] - ref_test[:, 1, :, :]))
            self.value = ((ref_true[:, 0, :, :]*ref_test[:, 0, :, :]*dot_prod).sum() /
                          torch.sqrt(torch.sum(ref_true[:, 0, :, :] * ref_true[:, 0, :, :]) * torch.sum(ref_test[:, 0, :, :] * ref_test[:, 0, :, :])))
        # self.value = torch.sum(ref_true * ref_test) / torch.sqrt(torch.sum(ref_true * ref_true) * torch.sum(ref_test * ref_test))
        return self.value

# class Metric_nec_tensor(BaseMetric_Tensor):
#     def __init__(self, device):
#         super().__init__(device)
#         self.metric = "Edges Correlation"
#         self.commentary = "The higher, the better"
#         self.range_min = 0
#         self.range_max = 1
#
#     def __call__(self, im1, im2, *args, mask=None, **kwargs):
#         super().__call__(im1, im2, *args, mask=mask, **kwargs)
#         ref_true = grad_tensor(self.image_true, self.device)
#         ref_test = grad_tensor(self.image_test, self.device)
#         ref_true = ref_true / ref_true.max()
#         ref_test = ref_test / ref_test.max()
#         if mask is not None:
#             ref_true = ref_true.flatten(start_dim=2)[:, :, self.mask[0, 0, :, :].flatten()]
#             ref_test = ref_test.flatten(start_dim=2)[:, :, self.mask[0, 0, :, :].flatten()]
#             # dot_prod = torch.abs(torch.cos(ref_true[:, 1:, :] - ref_test[:, 1:, :]))
#             # self.value = ((ref_true[:, :1, :]*ref_test[:, :1, :] * dot_prod).sum() /
#             #               torch.sqrt(torch.sum(ref_true[:, :1, :] * ref_true[:, :1, :]) * torch.sum(ref_test[:, :1, :] * ref_test[:, :1, :])))
#         # else:
#         #     dot_prod = torch.abs(torch.cos(ref_true[:, 1:, :, :] - ref_test[:, 1:, :, :]))
#         #     self.value = ((ref_true[:, :1, :, :]*ref_test[:, :1, :, :]*dot_prod).sum() /
#         #                   torch.sqrt(torch.sum(ref_true[:, :1, :, :] * ref_true[:, :1, :, :]) * torch.sum(ref_test[:, :1, :, :] * ref_test[:, :1, :, :])))
#         self.value = torch.sum(ref_true * ref_test) / torch.sqrt(torch.sum(ref_true * ref_true) * torch.sum(ref_test * ref_test))
#         return self.value

########
# class BaseMetric:
#     ##
#     # A class defining the general basic metric
#
#     def __init__(self, image_true, image_test, *args):
#         # Input array is a path to an image OR an already formed ndarray instance
#         assert isinstance(image_true, (np.ndarray, ImageCustom)) and isinstance(image_test, (np.ndarray, ImageCustom)), \
#             "A Metric is defined from an array or an ImageCustom, first Input got the wrong type"
#         assert image_true.shape[:2] == image_test.shape[:2], " The inputs are not the same size"
#         # if args:
#         #     assert isinstance(args[0], (np.ndarray, ImageCustom))
#         #     assert args[0].shape[:2] == image_test.shape[:2], " The inputs are not the same size"
#         #     if isinstance(args[0], ImageCustom):
#         #         image_fusion = args[0]
#         #     else:
#         #         image_fusion = ImageCustom(args[0])
#         # else:
#         #     image_fusion = None
#         if not isinstance(image_true, ImageCustom):
#             image_true = ImageCustom(image_true)
#         if not isinstance(image_test, ImageCustom):
#             image_test = ImageCustom(image_test)
#
#         if image_true.shape == image_test.shape:
#             self.image_true = image_true
#             self.image_test = image_test
#         elif len(image_true.shape) > 2:
#             self.image_true = image_true.LAB()[:, :, 0]
#             self.image_test = image_test
#         else:
#             self.image_true = image_true
#             self.image_test = image_test.LAB()[:, :, 0]
#         # if image_fusion is not None:
#         #     self.image_true = np.hstack([self.image_true, self.image_test])
#         #     self.image_test = np.hstack([image_fusion.LAB()[:, :, 0], image_fusion.LAB()[:, :, 0]])
#
#         self.metric = "Base Metric"
#         self.value = 0
#         self.range_min = 0
#         self.range_max = 1
#         self.range = (self.range_min, self.range_max)
#         self.commentary = "Just a base"
#
#     def __add__(self, other):
#         assert isinstance(other, BaseMetric)
#         if self.metric == other.metric:
#             self.range_max += self.range_max
#             self.value = self.value + other.value
#         else:
#             self.value = self.scale() + other.scale()
#         return self
#
#     def pass_attr(self, image):
#         self.__dict__ = image.__dict__.copy()
#
#     def scale(self):
#         return self.value
#
#     def __str__(self):
#         ##
#         # Redefine the way of printing
#         return f"{self.metric} metric : {self.value} | between {self.range_min} and {self.range_max} | {self.commentary}"
#
#
# class Metric_ssim(BaseMetric):
#     def __init__(self, image_true, image_test, *args):
#         super().__init__(image_true, image_test, *args)
#         layer = None
#         if len(self.image_true.shape) > 2:
#             layer = 2
#         self.value = structural_similarity(self.image_test, self.image_true, win_size=None, gradient=False,
#                                            data_range=None, channel_axis=layer, gaussian_weights=True, full=False)
#         self.metric = "SSIM"
#         self.commentary = "The higher, the better"
#
#     def scale(self):
#         self.range_max += self.range_max
#         return self
#
#
# class Metric_mse(BaseMetric):
#     def __init__(self, image_true, image_test, *args):
#         super().__init__(image_true, image_test, *args)
#         self.value = mean_squared_error(self.image_true, self.image_test)
#         self.metric = "MSE"
#         self.range_max = 255 ** 2
#         self.commentary = "The lower, the better"
#
#     def scale(self):
#         self.range_max = 2
#         return 1 - self.value / 128 ** 2
#
#
# class Metric_rmse(Metric_mse):
#     def __init__(self, image_true, image_test, *args):
#         super().__init__(image_true, image_test, *args)
#         self.value = np.sqrt(self.value)
#         self.metric = "RMSE"
#         self.range_max = 255
#
#     def scale(self):
#         self.value = super().scale()
#         return (255 - self.value) / 255
#
#
# class Metric_nrmse(BaseMetric):
#     def __init__(self, image_true, image_test, *args):
#         super().__init__(image_true, image_test, *args)
#         self.value = normalized_root_mse(self.image_true, self.image_test)
#         ref = normalized_root_mse(self.image_true, self.image_true - 128)
#         self.value = self.value / ref
#         self.metric = "Normalized RMSE"
#         self.commentary = "The lower, the better"
#
#     def scale(self):
#         self.value = super().scale()
#         return 1 - self.value
#
#
# class Metric_nmi(BaseMetric):
#     def __init__(self, image_true, image_test, *args):
#         super().__init__(image_true, image_test, *args)
#         self.value = normalized_mutual_information(self.image_true, self.image_test)
#         self.metric = "Normalized Mutual Information"
#         self.commentary = "The higher, the better"
#         self.range_min = 1
#         self.range_max = 2
#
#     def scale(self):
#         return 1 - -1 / (np.exp((self.value - 1) / (self.value - 2)))
#
#     def __add__(self, other):
#         assert isinstance(other, BaseMetric)
#         if self.metric == other.metric:
#             self.range_max *= self.range_max
#             self.value = self.value * other.value
#         else:
#             self.value = self.scale() + other.scale()
#         return self
#
#
# class Metric_psnr(BaseMetric):
#     def __init__(self, image_true, image_test, *args):
#         super().__init__(image_true, image_test, *args)
#         self.value = peak_signal_noise_ratio(self.image_true, self.image_test)
#         self.metric = "Peak Signal Noise Ratio"
#         self.commentary = "The higher, the better"
#         self.range_min = 0
#         ref = self.image_true.copy()
#         if len(ref.shape) > 2:
#             if ref[0, 0, 0] < 255:
#                 ref[0, 0, 0] += 1
#             else:
#                 ref[0, 0, 0] -= 1
#         else:
#             if ref[0, 0] < 255:
#                 ref[0, 0] += 1
#             else:
#                 ref[0, 0] -= 1
#         self.range_max = peak_signal_noise_ratio(self.image_true, ref)
#
#     def __add__(self, other):
#         assert isinstance(other, BaseMetric)
#         if self.metric == other.metric:
#             self.value = (self.value / self.range_max + other.value / other.range_max) / 2
#             self.range_max += other.range_max
#             self.value = self.value * self.range_max
#         else:
#             self.value = self.scale() + other.scale()
#         return self
#
#
# class Metric_nec(BaseMetric):
#     def __init__(self, image_true, image_test, *args, gradient=True):
#         super().__init__(image_true, image_test, *args)
#         self.metric = "Edges Correlation"
#         self.commentary = "The higher, the better"
#         self.range_min = 0
#         self.range_max = 1
#         if gradient:
#             ref_true = grad(self.image_true)
#             ref_test = grad(self.image_test)
#         else:
#             ref_true = self.image_true
#             ref_test = self.image_test
#         ref_true = ImageCustom(ref_true / ref_true.max() * 255)
#         ref_test = ImageCustom(ref_test / ref_test.max() * 255)
#
#         self.value = np.max(cv.matchTemplate(ref_true, ref_test, cv.TM_CCORR_NORMED))


#################### IMAGE FROM METRIC ###############################################
# class ImageMetric:
#     ##
#     # A class defining the general basic  Image metric
#
#     def __init__(self, image, *args):
#         # Input array is a path to an image OR an already formed ndarray instance
#         assert isinstance(image, (np.ndarray, ImageCustom)), \
#             "A Mask is defined from an array or an ImageCustom, first Input got the wrong type"
#         if isinstance(image, ImageCustom):
#             self.image = image
#         else:
#             self.image = ImageCustom(image)
#         if len(self.image.shape) > 2:
#             self.image = self.image.RGB()
#
#     def __add__(self, other):
#         assert isinstance(other, ImageMetric)
#         res = self.scale() + other.scale()
#         return res
#
#     def pass_attr(self, image):
#         self.__dict__ = image.__dict__.copy()
#
#     def scale(self):
#         pass
#
#
# class Image_ssim(ImageMetric):
#     def __init__(self, image, ref=None, win_size=3):
#         super().__init__(image)
#         if len(image.shape) > 2:
#             layer = 2
#         else:
#             layer = None
#         if ref is not None:
#             if len(image.shape) > 2 and len(ref.shape) == 2:
#                 image_ref = ref.RGB('gray')
#             else:
#                 image_ref = ref
#         else:
#             image_ref = np.ones_like(image) * image.max()
#         self.self_metric, self.image = structural_similarity(image, image_ref, win_size=win_size, gradient=False,
#                                                              data_range=None, channel_axis=layer,
#                                                              gaussian_weights=False, full=True)
#         self.image = - self.image
#
#         # self.image = 0.5 - self.image
#         self.image = ImageCustom(
#             (self.image - self.image.min()) / (self.image.max() - self.image.min()) * 255).GRAYSCALE()
#
#     def scale(self):
#         return self.image / 510
