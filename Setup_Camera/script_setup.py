import math
import os

import torch
from utils.ImagesCameras import Camera
from utils.ImagesCameras import CameraSetup

name_path = '/'
perso = '/home/aurelien/Images/Images_LYNRED/'
pro = '/media/godeta/T5 EVO/Datasets/Lynred/'

p = pro if 'godeta' in os.getcwd() else perso

dataset = "original"

if dataset == "original":
    # Original dataset
    path_RGB = p + 'Day/master/visible'
    path_RGB2 = p + 'Day/slave/visible'
    path_IR = p + 'Day/master/infrared_corrected'
    path_IR2 = p + 'Day/slave/infrared_corrected'
    file_name = "Lynred_test"

if dataset == "sequence":
    # New dataset sequence
    seq_num = 1
    path_RGB = p + f'/sequence_{seq_num}/visible_1/'
    path_RGB2 = p + f'/sequence_{seq_num}/visible_2/'
    path_IR = p + f'/sequence_{seq_num}/infrared_1/'
    path_IR2 = p + f'/sequence_{seq_num}/infrared_2/'
    file_name = f"Lynred_seq_{seq_num}"

IR = Camera(path=path_IR, device=torch.device('cuda'), id='IR', f=14, name='SmartIR640', sensor_name='SmartIR640')
IR2 = Camera(path=path_IR2, device=torch.device('cuda'), id='IR2', f=14, name='subIR', sensor_name='SmartIR640')
RGB = Camera(path=path_RGB, device=torch.device('cuda'), id='RGB', f=6, name='mainRGB', sensor_name='RGBLynred')
RGB2 = Camera(path=path_RGB2, device=torch.device('cuda'), id='RGB2', f=6, name='subRGB', sensor_name='RGBLynred')

R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)

d_calib = 5
center_x, center_y, center_z = 341 * 1e-03, 1, 0

x, y, z = 0.127, -0.008, -0.055  # -0.0413
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
rz = 0
R.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

x, y, z = 0.127 + 0.214, 0, 0
rx = 0  # math.atan((center_y - y) / d_calib) - math.atan(center_y / d_calib)
ry = math.atan((center_x - x) / d_calib) - math.atan(center_x / d_calib)
rz = 0
R.update_camera_relative_position('RGB2', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

x, y, z = 0.127 + 0.214 + 0.127, -0.008, -0.055
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
R.save(path_result, f'{file_name}.yaml')
