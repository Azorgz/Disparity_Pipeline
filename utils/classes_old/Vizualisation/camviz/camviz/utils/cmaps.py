# Copyright 2023 Toyota Research Institute.  All rights reserved.

import numpy as np
from matplotlib.cm import get_cmap

from camviz.utils.types import is_numpy, is_tensor


def jet(data, range=None, exp=1.0):
    """
    Creates a JET colormap from data

    Parameters
    ----------
    data : np.array [N,1]
        Data to be converted into a colormap
    range : tuple (min,max)
        Optional range value for the colormap (if None, use min and max from data)
    exp : float
        Exponential value to weight the color differently

    Returns
    -------
    colormap : np.array [N,3]
        Colormap obtained from data
    """
    # Return if data is not available
    if data is None or data.size == 0 or isinstance(data, tuple):
        return data
    else:
        # If data is a tensor, convert to numpy
        if is_tensor(data):
            data = data.detach().cpu().numpy()
        # If data is [N,1], remove second dimensions
        if len(data.shape) > 1:
            data = data.reshape(-1)
        # Determine range if not available
        if range is None:
            data = data.copy() - np.min(data)
            data = data / (np.max(data) + 1e-6)
        else:
            data = (data - range[0]) / (range[1] - range[0])
            data = np.maximum(np.minimum(data, 1.0), 0.0)
        # Use exponential if requested
        if exp != 1.0:
            data = data ** exp
        # Initialize colormap
        jet = np.ones((data.shape[0], 3), dtype=np.float32)
        # First stage
        idx = (data <= 0.33)
        jet[idx, 1] = data[idx] / 0.33
        jet[idx, 0] = 0.0
        # Second stage
        idx = (data > 0.33) & (data <= 0.67)
        jet[idx, 0] = (data[idx] - 0.33) / 0.33
        jet[idx, 2] = 1.0 - jet[idx, 0]
        # Third stage
        idx = data > 0.67
        jet[idx, 1] = 1.0 - (data[idx] - 0.67) / 0.33
        jet[idx, 2] = 0.0
        # Return colormap
        return jet


def depth_clr(data, range=None, exp=1.0, delta=1e-1):
    """
    Creates a depth_clr colormap from data

    Parameters
    ----------
    data : np.array [N,1]
        Data to be converted into a colormap
    range : tuple (min,max)
        Optional range value for the colormap (if None, use min and max from data)
    exp : float
        Exponential value to weight the color differently

    Returns
    -------
    colormap : np.array [N,3]
        Colormap obtained from data
    """
    # Return if data is not available
    if data is None or data.size == 0 or isinstance(data, tuple):
        return data
    else:
        # If data is a tensor, convert to numpy
        if is_tensor(data):
            data = data.detach().cpu().numpy()
        # If data is [N,1], remove second dimensions
        if len(data.shape) > 1:
            data = data.reshape(-1)
        # Determine range if not available
        if range is None:
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-6)
        else:
            data = (data - range[0]) / (range[1] - range[0])
            data = np.maximum(np.minimum(data, 1.0), 0.0)
        # Use exponential if requested

        if exp != 1.0:
            data = data ** exp
        # Initialize colormap
        depth_clr = np.ones((data.shape[0], 3), dtype=np.float32)
        # Red 1/exp2
        depth_clr[:, 0] = 1 / np.exp2(data)

        # Green
        depth_clr[:, 1] = np.sqrt(delta / (np.arctan(data) + delta))

        # Blue
        depth_clr[:, 2] = delta / (np.arctan(1 - data) + delta)

        # Return colormap
        return depth_clr
