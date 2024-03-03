import math
import os

import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup


name_path = '/'
perso = '/home/aurelien/Images/Images_LLVIP/LLVIP/'
pro = '/home/aurelien/Images/Images_LLVIP/LLVIP/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'visible/test/'
path_IR = p + 'infrared/test/'

IR = IRCamera(None, None, path_IR, device=torch.device('cuda'), name='IR', aperture=1.2)

RGB = RGBCamera(None, None, path_RGB, device=torch.device('cuda'), name='RGB', aperture=1.4)

R = CameraSetup(RGB, IR, print_info=True)
R.update_camera_relative_position('IR', x=0, y=0, z=0, ry=0, rx=0, rz=0)

R.calibration_for_stereo('IR', 'RGB')
R.calibration_for_depth('IR', 'RGB')
# R.stereo_pair('RGB', 'IR').show_image_calibration()

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
R.save(path_result, 'Setup_Camera_LLVIP.yaml')
