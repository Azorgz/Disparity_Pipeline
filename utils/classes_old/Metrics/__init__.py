from __future__ import annotations

from .Metrics import Metric_ssim_tensor, MultiScaleSSIM_tensor, Metric_mse_tensor, Metric_rmse_tensor, \
    Metric_psnr_tensor, Metric_nec_tensor
import numpy as np

# Version variable
__version__ = "1.0"

norms_dict = {'rmse': Metric_rmse_tensor,
              'psnr': Metric_psnr_tensor,
              'ssim': Metric_ssim_tensor,
              'ms_ssim': MultiScaleSSIM_tensor,
              'nec': Metric_nec_tensor}

stats_dict = {'mean': np.mean,
              'std': np.std}

__all__ = ["Metric_ssim_tensor",
           "MultiScaleSSIM_tensor",
           "Metric_mse_tensor",
           "Metric_rmse_tensor",
           "Metric_psnr_tensor",
           "Metric_nec_tensor",
           "norms_dict",
           "stats_dict"
           ]
