import math
import os

import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup

name_path = '/'
perso = '/home/aurelien/Images/'
pro = '/home/godeta/PycharmProjects/'

p = pro if 'godeta' in os.getcwd() else perso

path_RGB = p + 'Images_FLIR/FLIR_ADAS_1_3_full/RGB_processed/'
path_IR = p + 'Images_FLIR/FLIR_ADAS_1_3_full/IR_processed/'

IR720 = IRCamera(path=path_IR+'720x480/', device=torch.device('cuda'), id='IR', name='FLIR Tau2',
                 f=13, pixel_size=17, aperture=1, sensor_resolution=(640, 512))
IR1280 = IRCamera(path=path_IR+'1280x1024/', device=torch.device('cuda'), id='IR', name='FLIR Tau2',
                  f=13, pixel_size=17, aperture=1, sensor_resolution=(640, 512))
IR1800 = IRCamera(path=path_IR+'1800x1600/', device=torch.device('cuda'), id='IR', name='FLIR Tau2',
                  f=13, pixel_size=17, aperture=1, sensor_resolution=(640, 512))
IR2048 = IRCamera(path=path_IR+'2048x1536/', device=torch.device('cuda'), id='IR', name='FLIR Tau2',
                  f=13, pixel_size=17, aperture=1, sensor_resolution=(640, 512))

RGB720 = RGBCamera(path=path_RGB+'720x480/', device=torch.device('cuda'), id='RGB', name='FLIR BlackFly',
                   f=4, pixel_size=7.4, aperture=1.4, sensor_resolution=(720, 480))
RGB1280 = RGBCamera(path=path_RGB+'1280x1024/', device=torch.device('cuda'), id='RGB', name='FLIR BlackFly',
                    f=8.5, HFOV=52.8, pixel_size=4, aperture=1.4, sensor_resolution=(2048, 1536))
RGB1800 = RGBCamera(path=path_RGB+'1800x1600/', device=torch.device('cuda'), id='RGB', name='FLIR BlackFly',
                    f=8.5, HFOV=52.8, pixel_size=3, aperture=1.4, sensor_resolution=(2048, 1536))
RGB2048 = RGBCamera(path=path_RGB+'2048x1536/', device=torch.device('cuda'), id='RGB', name='FLIR BlackFly',
                    f=8.5, pixel_size=3.4, aperture=1.4, sensor_resolution=(2448, 2048))


R720 = CameraSetup(RGB720, IR720, print_info=True)
R1280 = CameraSetup(RGB1280, IR720, print_info=True)
R1800 = CameraSetup(RGB1800, IR1800, print_info=True)
R2048 = CameraSetup(RGB2048, IR2048,  print_info=True)

x, y, z = 0.04826, -0.001, 0
rx = 0
ry = 0
rz = 0

R720.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
R1280.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
R1800.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)
R2048.update_camera_relative_position('IR', x=x, y=y, z=z, ry=ry, rx=rx, rz=rz)

R720.calibration_for_stereo('IR', 'RGB')
# R720.stereo_pair('RGB', 'IR').show_image_calibration()

R1280.calibration_for_stereo('IR', 'RGB')
# R1280.stereo_pair('RGB', 'IR').show_image_calibration()

R1800.calibration_for_stereo('IR', 'RGB')
R1800.stereo_pair('RGB', 'IR').show_image_calibration()

R2048.calibration_for_stereo('IR', 'RGB')
R2048.stereo_pair('RGB', 'IR').show_image_calibration()

perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/'
pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/'
p = pro if 'godeta' in os.getcwd() else perso
path_result = p + name_path

R720.save(path_result, 'Setup_Camera_FLIR_720.yaml')
R1280.save(path_result, 'Setup_Camera_FLIR_1280.yaml')
R1800.save(path_result, 'Setup_Camera_FLIR_1800.yaml')
R2048.save(path_result, 'Setup_Camera_FLIR_2048.yaml')
