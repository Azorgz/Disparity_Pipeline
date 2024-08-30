import math
import os

import torch
from utils.ImagesCameras import Camera
from utils.ImagesCameras import CameraSetup

name_path = '/'
perso = '/home/aurelien/Images/Images_LYNRED/'
pro = '/media/godeta/T5 EVO/Datasets/CATS_Sorted/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB_left = p + 'OUTDOORS/color/left'
path_RGB_right = p + 'OUTDOORS/color/right'
path_IR_left = p + 'OUTDOORS/thermal/left'
path_IR_right = p + 'OUTDOORS/thermal/right'

IR = Camera(path=path_IR_left, device=torch.device('cuda'), id='IR', f=12.1, name='SmartIR640',
            sensor_name='SmartIR640')
IR2 = Camera(path=path_IR_right, device=torch.device('cuda'), id='IR2', f=12.1, name='subIR', sensor_name='SmartIR640')
RGB = Camera(path=path_RGB_left, device=torch.device('cuda'), id='RGB', f=6, name='mainRGB', sensor_name='RGBLynred')
RGB2 = Camera(path=path_RGB_right, device=torch.device('cuda'), id='RGB2', f=6, name='subRGB', sensor_name='RGBLynred')

R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)

d_calib = 4.8
center_x, center_y, center_z = 200 * 1e-03, 1, 4.8  # 341 * 1e-03, 1, 0

x, y, z = -0.1, 0.168, -0.1  # -0.0413
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / center_z) - math.atan(center_x / center_z)
rz = 0
R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

x, y, z = -0.353, 0.11, -0.018
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / center_z) - math.atan(center_x / center_z)
rz = 0
R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

# x, y, z = 0.127 + 0.214 + 0.127, -0.008, -0.055
# rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
# ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
# rz = 0
# R.update_camera_relative_position('IR2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

R.calibration_for_stereo('RGB', 'RGB2')
R.stereo_pair('RGB', 'RGB2').show_image_calibration()

# R.calibration_for_stereo('IR', 'RGB')
# R.calibration_for_stereo('IR', 'RGB2')
# R.stereo_pair('RGB', 'IR').show_image_calibration()

# R.calibration_for_stereo('IR', 'IR2')
# R.stereo_pair('IR', 'IR2').show_image_calibration()

# R.calibration_for_depth('IR', 'IR2')
R.calibration_for_depth('RGB', 'RGB2')

print(R)

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/Setup_Camera/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/Setup_Camera/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path
R.save(path_result, 'Cats_out.yaml')
