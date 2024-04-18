import math
import os

import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup


name_path = '/'
perso = '/home/aurelien/Images/Images_LLVIP/LLVIP_raw_images/train/'
pro = '/home/aurelien/Images/Images_LLVIP/LLVIP_raw_images/train/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'visible'
path_IR = p + 'infrared_corrected'

IR = IRCamera(path_IR, device=torch.device('cuda'), name='IR', f=75e-3, pixel_size=(17e-6, 17e-6),
              aperture=1.2)

RGB = RGBCamera(path_RGB, device=torch.device('cuda'), name='RGB', f=27e-3, pixel_size=(2.9e-6, 2.9e-6),
                aperture=1.4)

R = CameraSetup(RGB, IR, print_info=True)
R.update_camera_relative_position('IR', x=-5e-3, y=0, z=0, ry=0, rx=0, rz=0)

R.calibration_for_stereo('IR', 'RGB')
R.calibration_for_depth('IR', 'RGB')
# R.stereo_pair('RGB', 'IR').show_image_calibration()

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
R.save(path_result, 'Setup_Camera_LLVIP_raw.yaml')
