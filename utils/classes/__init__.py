# from .Image import ImageCustomImageCustom
from .Image import ImageTensor
from .Metrics import Metric_ssim_tensor, MultiScaleSSIM_tensor, Metric_mse_tensor, Metric_rmse_tensor, \
    Metric_psnr_tensor, Metric_nec_tensor
import numpy as np
import torch

norms_dict = {'rmse': Metric_rmse_tensor,
              'psnr': Metric_psnr_tensor,
              'ssim': Metric_ssim_tensor,
              'ms_ssim': MultiScaleSSIM_tensor,
              'nec': Metric_nec_tensor}

stats_dict = {'mean': np.mean,
              'std': np.std}
