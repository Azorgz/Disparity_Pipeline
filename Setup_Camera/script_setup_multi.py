import math

import numpy as np
import torch
from tqdm import tqdm

from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup
from utils.manipulation_tools import random_noise, noise

perso = '/home/aurelien/Images/Images/'
pro = '/home/godeta/PycharmProjects/LYNRED/Images/'
p = pro

path_RGB = p + 'Day/master/visible'
path_RGB2 = p + 'Day/slave/visible'
path_IR = p + 'Day/master/infrared_corrected'

IR = IRCamera(None, None, path_IR, device=torch.device('cuda'), name='IR', f=14e-3, pixel_size=(16.4e-6, 16.4e-6),
              aperture=1.2)
RGB = RGBCamera(None, None, path_RGB, device=torch.device('cuda'), name='RGB', f=6e-3, pixel_size=(3.45e-6, 3.45e-6),
                aperture=1.4)
RGB2 = RGBCamera(None, None, path_RGB2, device=torch.device('cuda'), name='RGB2', f=6e-3, pixel_size=(3.45e-6, 3.45e-6),
                 aperture=1.4)
R = CameraSetup(RGB, IR, RGB2, print_info=True)

d_calib = 5
center_x, center_y, center_z = 341 * 1e-03, 1, 0

vec_x = np.arange(0, 5 * 1e-2, 5*1e-3)
vec_x = (vec_x - vec_x.max()/2).tolist()
vec_z = np.arange(0, 5 * 1e-2, 1e-2)
vec_z = (vec_z - vec_z.max()/2).tolist()
vec_y = np.arange(0, 5 * 1e-2, 1e-2)
vec_y = (vec_y - vec_y.max()/2).tolist()
vec_alpha = np.arange(0, 2e-2, 4e-3)
vec_alpha = (vec_alpha - vec_alpha.max()/2).tolist()

with tqdm(total=len(vec_x)*len(vec_y)*len(vec_z)*len(vec_alpha), desc='Setups saving') as bar:
    for i, dx in enumerate(vec_x):
        for j, dy in enumerate(vec_y):
            for k, dz in enumerate(vec_z):
                for h, da in enumerate(vec_alpha):
                    x, y, z = 0.127, 0, 0
                    rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
                    ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
                    rz = 0
                    R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

                    x, y, z = 0.127 + 0.214 + dx, 0 + dy, 0 + dz
                    rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
                    ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib) + da
                    rz = 0
                    R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

                    R.calibration_for_stereo('RGB', 'RGB2')
                    R.calibration_for_stereo('IR', 'RGB')
                    R.calibration_for_depth('IR', 'RGB')
                    R.calibration_for_depth('RGB', 'RGB2')
                    name = f'dx_{i}_dy_{j}dz_{k}_da_{h}.yaml'
                    home = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
                    pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
                    R.save(pro + 'Setup_Camera/Setup_Camera_postion_rgb', name)
                    bar.update(n=1)
