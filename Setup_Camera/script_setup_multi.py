import math
import os

import numpy as np
import torch
from tqdm import tqdm

from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup
from utils.manipulation_tools import random_noise, noise

cam = 'RGB'
setup = 'raw'
name_path = f'Setup_Camera/position_{"ir" if cam=="IR" else "rgb"}{"_finer" if setup == "fine" else ""}'

perso = '/home/aurelien/Images/Images/'
pro = '/home/godeta/PycharmProjects/LYNRED/Images/'
p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'Day/master/visible'
path_RGB2 = p + 'Day/slave/visible'
path_IR = p + 'Day/master/infrared_corrected'

####### POSITION PARAMETERS ###################

d_calib = 5
center_x, center_y, center_z = 341 * 1e-03, 1, 0

if setup == 'raw':
    vec_x = np.arange(0, 8 * 1e-2, 1e-2).tolist()
    vec_z = np.arange(0, 6e-2, 1e-2).tolist()
    vec_y = np.arange(0, 6e-2, 1e-2).tolist()
    vec_alpha = (np.arange(0, 4, 1) / 180 * np.pi).tolist()

elif setup == 'fine':
    vec_x = np.arange(0, 9 * 1e-2, 1e-2)
    vec_x = (vec_x - vec_x.max() / 2).tolist()
    vec_z = np.arange(-6e-2, 1.5e-2, 5e-3)
    vec_z = (vec_z - vec_z.max() / 2).tolist()
    vec_y = np.arange(0, 1.5e-1, 2.5e-2)
    vec_y = (vec_y - vec_y.max() / 2).tolist()
    vec_alpha = (np.arange(0, 3, 1) / 180 * np.pi)
    vec_alpha = (vec_alpha - vec_alpha.max() / 2).tolist()

else:
    vec_x = [0]
    vec_z = [0]
    vec_y = [0]
    vec_alpha = [0]


IR = IRCamera(None, None, path_IR, device=torch.device('cuda'), name='IR', f=14e-3, pixel_size=(16.4e-6, 16.4e-6),
                  aperture=1.2)
RGB = RGBCamera(None, None, path_RGB, device=torch.device('cuda'), name='RGB', f=6e-3, pixel_size=(3.45e-6, 3.45e-6),
                    aperture=1.4)
RGB2 = RGBCamera(None, None, path_RGB2, device=torch.device('cuda'), name='RGB2', f=6e-3, pixel_size=(3.45e-6, 3.45e-6),
                     aperture=1.4)
R = CameraSetup(RGB, IR, RGB2, print_info=True)

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
if not os.path.exists(path_result):
    os.makedirs(path_result, exist_ok=True)
    os.chmod(path_result, 0o777)
if os.listdir(path_result):
    from utils.misc import clear_folder
    clear_folder(path_result)

with tqdm(total=len(vec_x) * len(vec_y) * len(vec_z) * len(vec_alpha), desc=f'Setups saving for {setup} {cam}') as bar:
    for i, dx in enumerate(vec_x):
        for j, dy in enumerate(vec_y):
            for k, dz in enumerate(vec_z):
                for h, da in enumerate(vec_alpha):
                    x, y, z = (0.127 + dx, 0 + dy, 0 + dz) if cam == 'IR' else (0.127, 0, 0)
                    rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
                    ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
                    rz = 0 + da
                    R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

                    x, y, z = (0.127 + 0.214 + dx, 0 + dy, 0 + dz) if cam == 'RGB' else (0.127 + 0.214, 0, 0)
                    rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
                    ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
                    rz = 0
                    R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

                    R.calibration_for_stereo('RGB', 'RGB2')
                    R.calibration_for_stereo('IR', 'RGB')
                    R.calibration_for_depth('IR', 'RGB')
                    R.calibration_for_depth('RGB', 'RGB2')
                    name = (f'_dx_{i if i >= 10 else str(0) + str(i)}_dy_{j if j >= 10 else str(0) + str(j)}'
                            f'_dz_{k if k >= 10 else str(0) + str(k)}_da_{h if h >= 10 else str(0) + str(h)}_.yaml')
                    R.save(path_result, name)
                    bar.update(n=1)
