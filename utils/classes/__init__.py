from .Image import ImageCustom
from .Metrics import Metric_ssim, Metric_nmi, Metric_psnr, Metric_rmse, Metric_nec, Metric_ssim_tensor, MultiScaleSSIM_tensor, Metric_mse_tensor, Metric_rmse_tensor, Metric_psnr_tensor, Metric_nec_tensor
import numpy as np
import torch

norms_dict = {'rmse': Metric_rmse,
               'nmi': Metric_nmi,
               'psnr': Metric_psnr,
               'ssim': Metric_ssim,
               'nec': Metric_nec}

norms_dict_gpu = {'rmse': Metric_rmse_tensor,
                      'psnr': Metric_psnr_tensor,
                      'ssim': Metric_ssim_tensor,
                      'ms_ssim': MultiScaleSSIM_tensor,
                      'nec': Metric_nec_tensor}

stats_dict = {'mean': np.mean,
              'std': np.std}
