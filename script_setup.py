import math
import os

import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup


name_path = '/'
perso = '/home/aurelien/Images/Images/'
pro = '/home/godeta/PycharmProjects/LYNRED/Images/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'Night/master/visible'
path_RGB2 = p + 'Night/slave/visible'
path_IR = p + 'Night/master/infrared_corrected'
path_IR2 = p + 'Night/slave/infrared_corrected'

IR = IRCamera(None, None, path_IR, device=torch.device('cuda'), name='IR', f=14e-3, pixel_size=(16.4e-6, 16.4e-6),
              aperture=1.2)
IR2 = IRCamera(None, None, path_IR2, device=torch.device('cuda'), name='IR2', f=14e-3, pixel_size=(16.4e-6, 16.4e-6),
               aperture=1.2)
RGB = RGBCamera(None, None, path_RGB, device=torch.device('cuda'), name='RGB', f=6e-3, pixel_size=(3.45e-6, 3.45e-6),
                aperture=1.4)
RGB2 = RGBCamera(None, None, path_RGB2, device=torch.device('cuda'), name='RGB2', f=6e-3, pixel_size=(3.45e-6, 3.45e-6),
                 aperture=1.4)
R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)

d_calib = 5
center_x, center_y, center_z = 341 * 1e-03, 1, 0

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

x, y, z = 0.127 + 0.214 + 0.127, 0, 0
rx = math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
rz = 0

R.update_camera_relative_position('IR2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
R.calibration_for_stereo('RGB', 'RGB2')
# R.stereo_pair('RGB', 'RGB2').show_image_calibration()

R.calibration_for_stereo('IR', 'RGB')
R.calibration_for_stereo('IR', 'RGB2')
# R.stereo_pair('RGB', 'IR').show_image_calibration()

R.calibration_for_stereo('IR', 'IR2')
# R.stereo_pair('IR', 'IR2').show_image_calibration()

R.calibration_for_depth('IR', 'IR2')
R.calibration_for_depth('RGB', 'RGB2')

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
R.save(path_result, 'Setup_Camera_night.yaml')
