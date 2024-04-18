import math
import os

import numpy as np
import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup

name_path = '/'
perso = '/home/aurelien/Images/Images_LYNRED/'
pro = '/home/godeta/PycharmProjects/Datasets/Lynred/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB = [p + 'Day/master/visible', p + 'Night/master/visible']
path_RGB2 = [p + 'Day/slave/visible', p + 'Night/slave/visible']
path_IR = [p + 'Day/master/infrared_corrected', p + 'Night/master/infrared_corrected']
path_IR2 = [p + 'Day/slave/infrared_corrected', p + 'Night/slave/infrared_corrected']

IR = IRCamera(path=path_IR, device=torch.device('cuda'), id='IR', name='SmartIR640', f=14, pixel_size=16.4,
              aperture=1.2)

IR2 = IRCamera(path=path_IR2, device=torch.device('cuda'), id='IR2', name='subIR', f=14, pixel_size=(16.4, 16.4),
               aperture=1.2)
RGB = RGBCamera(path=path_RGB, device=torch.device('cuda'), id='RGB', name='mainRGB', f=6, pixel_size=3.45,
                aperture=1.4)
RGB2 = RGBCamera(path=path_RGB2, device=torch.device('cuda'), id='RGB2', name='subRGB', f=6, pixel_size=(3.45, 3.45),
                 aperture=1.4)
R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)

d_calib = 5
center_x, center_y, center_z = 341 * 1e-03, 1, 0

x, y, z = 0.127, 0, -0.0413
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
rz = 0
# ext = torch.tensor(np.array([[9.99992049e-01, -3.97317796e-03, -3.40122104e-04, 3.14094925e-01],
#                              [3.97420714e-03, 9.99987357e-01, 3.08070689e-03, -5.06425823e-02],
#                              [3.27877607e-04, -3.08203411e-03, 9.99995197e-01, 2.15528077e-02],
#                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]))
# R.update_camera_relative_position('IR', extrinsics=ext)
R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

x, y, z = 0.127 + 0.214, 0, 0
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
rz = 0
R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

x, y, z = 0.127 + 0.214 + 0.127, 0, -0.0413
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
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

print(R)

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/Setup_Camera/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/Setup_Camera/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
R.save(path_result, 'Lynred_day&night.yaml')