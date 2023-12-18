import math
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup
from utils.manipulation_tools import random_noise, noise

cam = 'RGB'
setup = 'raw test'
name_path = f'Setup_Camera/intrinsics_{"ir" if cam == "IR" else "rgb"}{"_finer" if setup == "fine" else ""}'

perso = '/home/aurelien/Images/Images/'
pro = '/home/godeta/PycharmProjects/LYNRED/Images/'
p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'Day/master/visible'
path_RGB2 = p + 'Day/slave/visible'
path_IR = p + 'Day/master/infrared_corrected'

####### POSITION PARAMETERS ###################

d_calib = 5
center_x, center_y, center_z = 341 * 1e-03, 1, 0

######### INTRINSICS PARAMETERS  ##################
if setup == 'raw test':
    if cam == 'IR':
        f_, px_size_ = 14e-3, 16.4e-6
    else:
        f_, px_size_ = 6e-3, 3.45e-6
    f = (np.arange(0.75, 2, 0.25) * f_).tolist()
    px_size = (np.arange(0.75, 2, 0.25) * px_size_).tolist()
else:
    if cam == 'IR':
        f_, px_size_ = 14e-3, 16.4e-6
    else:
        f_, px_size_ = 6e-3, 3.45e-6
    f = (np.arange(0.75, 2, 0.05) * f_).tolist()
    px_size = (np.arange(0.75, 2, 0.05) * px_size_).tolist()

with tqdm(total=len(f) * len(px_size), desc='Setups saving') as bar:
    for i, f_ in enumerate(f):
        for j, px_size_ in enumerate(px_size):
            IR = IRCamera(None, None, path_IR, device=torch.device('cuda'), name='IR',
                          f=f_ if cam == 'IR' else 14e-3,
                          pixel_size=(px_size_ if cam == 'IR' else 16.4e-6, px_size_ if cam == 'IR' else 16.4e-6),
                          aperture=1.2)
            RGB = RGBCamera(None, None, path_RGB, device=torch.device('cuda'), name='RGB',
                            f=f_ if cam == 'RGB' else 6e-3,
                            pixel_size=(px_size_ if cam == 'RGB' else 3.45e-6, px_size_ if cam == 'RGB' else 3.45e-6),
                            aperture=1.4)
            RGB2 = RGBCamera(None, None, path_RGB2, device=torch.device('cuda'), name='RGB2',
                             f=f_ if cam == 'RGB' else 6e-3,
                             pixel_size=(px_size_ if cam == 'RGB' else 3.45e-6, px_size_ if cam == 'RGB' else 3.45e-6),
                             aperture=1.4)
            R = CameraSetup(RGB, IR, RGB2, print_info=False)

            x, y, z = 0.127, 0, 0
            rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
            ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
            rz = 0
            R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

            x, y, z = 0.127 + 0.214, 0, 0
            rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
            ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
            rz = 0
            R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

            R.calibration_for_stereo('RGB', 'RGB2')
            R.calibration_for_stereo('IR', 'RGB')
            R.calibration_for_depth('IR', 'RGB')
            R.calibration_for_depth('RGB', 'RGB2')
            name = f'_f_{i if i >= 10 else str(0) + str(i)}_px_{j if j >= 10 else str(0) + str(j)}_.yaml'
            perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
            pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
            p = pro if 'godeta' in os.getcwd() else perso
            path_result = p + name_path
            if not os.path.exists(path_result):
                os.makedirs(path_result, exist_ok=True)
                os.chmod(path_result, 0o777)

            R.save(path_result, name)
            bar.update(n=1)