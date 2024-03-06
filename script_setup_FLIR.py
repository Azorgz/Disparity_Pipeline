import math
import os

import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup

name_path = '/'
perso = '/home/aurelien/Images/Images/'
pro = '/home/godeta/PycharmProjects/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'FLIR_ADAS_v2/images_rgb_processed/'
path_IR = p + 'FLIR_ADAS_v2/images_thermal_processed/'

IR = IRCamera(path=path_IR, device=torch.device('cuda'), name='IR', f=13, HFOV=45, VFOV=37, aperture=1)

RGB = RGBCamera(path=path_RGB, device=torch.device('cuda'), name='RGB', f=8.5, HFOV=52.8, pixel_size=3.5, aperture=1.4,
                sensor_resolution=(2448, 2048))
R = CameraSetup(RGB, IR, print_info=True)

x, y, z = 0.001, 0, 0
rx = 0
ry = 0
rz = 0
R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

R.calibration_for_stereo('IR', 'RGB')
R.stereo_pair('RGB', 'IR').show_image_calibration()

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
R.save(path_result, 'Setup_Camera_FLIR.yaml')
