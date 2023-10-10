import math
import torch
from utils.classes.Cameras import RGBCamera, IRCamera
from module.SetupCameras import CameraSetup

path = '/home/godeta/PycharmProjects/LYNRED/Images/Day/master/visible'
path2 = '/home/godeta/PycharmProjects/LYNRED/Images/Day/slave/visible'
path1 = '/home/godeta/PycharmProjects/LYNRED/Images/Day/master/infrared_corrected'
path3 = '/home/godeta/PycharmProjects/LYNRED/Images/Day/slave/infrared_corrected'

IR = IRCamera(None, None, path1, device=torch.device('cuda'), name='IR', f=14e-3, pixel_size=(16.4e-6, 16.4e-6), aperture=1.2)
IR2 = IRCamera(None, None, path3, device=torch.device('cuda'), name='IR2', f=14e-3, pixel_size=(16.4e-6, 16.4e-6), aperture=1.2)
RGB = RGBCamera(None, None, path, device=torch.device('cuda'), name='RGB', f=6e-3, pixel_size=(3.45e-6, 3.45e-6), aperture=1.4)
RGB2 = RGBCamera(None, None, path2, device=torch.device('cuda'), name='RGB2', f=6e-3, pixel_size=(3.45e-6, 3.45e-6), aperture=1.4)
R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)

d_calib = 5
center_x, center_y, center_z = 341*1e-03, 1, 0
x, y, z = 0.127, 0, 0
R.update_camera_relative_position('IR', x=x, y=y, z=z,
                                  ry=math.atan((center_x - x)/d_calib)-math.atan(center_x/d_calib),
                                  rx=math.atan((center_y - y)/d_calib)-math.atan(center_y/d_calib),
                                  rz=0)
x, y, z = 0.127+0.214, 0, 0
R.update_camera_relative_position('RGB2', x=x, y=y, z=z,
                                  ry=math.atan((center_x - x) / d_calib) - math.atan(center_x/d_calib),
                                  rx=math.atan((center_y - y)/d_calib)-math.atan(center_y/d_calib))
x, y, z = 0.127+0.214+0.127, 0, 0
R.update_camera_relative_position('IR2', x=x, y=y, z=z,
                                  ry=math.atan((center_x - x) / d_calib) - math.atan(center_x/d_calib),
                                  rx=math.atan((center_y - y)/d_calib)-math.atan(center_y/d_calib))

R.calibration_for_stereo('RGB', 'RGB2')
# R.stereo_pair('RGB', 'RGB2').show_image_calibration()

R.calibration_for_stereo('IR', 'RGB')
R.calibration_for_stereo('IR', 'RGB2')
R.stereo_pair('RGB', 'IR').show_image_calibration()

R.calibration_for_stereo('IR', 'IR2')
# R.stereo_pair('IR', 'IR2').show_image_calibration()

R.calibration_for_depth('IR', 'IR2')
R.calibration_for_depth('RGB', 'RGB2')


R.save('/home/godeta/PycharmProjects/Disparity_Pipeline/')
