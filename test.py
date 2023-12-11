import os
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot
from pandas import DataFrame

# import torch
from Result_analysis.ResultFrame import ResultFrame
# from utils.classes import ImageTensor
# from utils.classes.Visualizer import Visualizer

# from utils.classes.Image import ImageTensor
import numpy as np

# import cv2 as cv
#
# from utils.registration_tools import SIFT
# from utils.visualization import result_visualizer, visual_control
base_path = os.getcwd() + "/results/"
res = ResultFrame(base_path + "camera_position_ir_finer/Depth-Disparity")
# delta = res.delta
z_coeff = 11
y_coeff = 3 * z_coeff
x_coeff = 11 * y_coeff


def cal_idx(variable):
    if variable == 'x':
        y, z, a = np.arange(0, 5, 1) * y_coeff, np.arange(0, 5, 1) * z_coeff, np.arange(0, 5, 1)
        x = np.arange(0, 11, 1) * x_coeff
        result = [[y_+z_+a_ for y_ in y for z_ in z for a_ in a]+x_ for x_ in x]
    elif variable == 'y':
        x, z, a = np.arange(0, 11, 1) * x_coeff, np.arange(0, 5, 1) * z_coeff, np.arange(0, 5, 1)
        y = np.arange(0, 5, 1) * y_coeff
        result = [[x_ + z_ + a_ for x_ in x for z_ in z for a_ in a] + y_ for y_ in y]
    elif variable == 'z':
        x, y, a = np.arange(0, 11, 1) * x_coeff, np.arange(0, 5, 1) * y_coeff, np.arange(0, 5, 1)
        z = np.arange(0, 5, 1) * z_coeff
        result = [[y_ + x_ + a_ for y_ in y for x_ in x for a_ in a] + z_ for z_ in z]
    elif variable == 'a':
        x, y, z = np.arange(0, 11, 1) * x_coeff, np.arange(0, 5, 1) * y_coeff, np.arange(0, 5, 1) * z_coeff
        a = np.arange(0, 5, 1)
        result = [[x_ + y_ + z_ for x_ in x for y_ in y for z_ in z] + a_ for a_ in a]
    else:
        result = np.array(0).tolist()
    return result


idx_x = cal_idx('x')
idx_y = cal_idx('y')
idx_z = cal_idx('z')
idx_a = cal_idx('a')

val = res.delta_full.combine_column('nec-rmse+psnr+ms_ssim+ssim', 'sum').values

v_x = val[idx_x].mean(1)

v_y = val[idx_y].mean(1)

v_z = val[idx_z].mean(1)

v_a = val[idx_a].mean(1)


plt.plot(np.arange(0, 11 * 1e-2, 1e-2), v_x)
plt.plot(np.arange(0, 1e-1, 2e-2), v_y)
plt.plot(np.arange(0, 1e-1, 2e-2), v_z)
plt.plot(np.arange(0, 10, 2)/180*np.pi, v_a)
plt.legend(['dx m', 'dy m', 'dz m', 'da rad'])
plt.show()


time.sleep(1)
#
# name = '/home/godeta/PycharmProjects/Disparity_Pipeline/results/Vis/Validation.yaml'
# with open(name, "r") as file:
#     val = yaml.safe_load(file)
#
# list_new = []
# list_ref = []
# diff_length = 3
# kernel = np.array([1 for i in range(diff_length*2+1)])
#
# stat = 'ms_ssim'
# name = f'{stat}_new/{stat}_ref'
# for new, ref in val['2. results'][name]:
#     list_new.append(new)
#     list_ref.append(ref)
# x = range(len(list_new))
# list_new = np.array(list_new)
# list_ref = np.array(list_ref)
#
# ax1 = plt.subplot(121)
# ax1.plot(x, list_new, label='new')
# ax1.plot(x, list_ref, label='ref')
# ax1.set_title(stat)
# ax2 = plt.subplot(122)
# diff = np.convolve(list_new-list_ref, kernel/(diff_length*2+1))
# ax2.plot(x, diff[diff_length:-diff_length], label='diff')
# ax2.plot(x, np.array(x)*0)
# ax2.set_title('Diff')
#
# plt.legend()
# plt.show()
# im = ImageTensor("/home/godeta/PycharmProjects/Disparity_Pipeline/results/Res/Disparity-HD/image_reg/RGB2_to_RGB/RGB2_000.png",
#                  device=torch.cuda.device(0))
# i = im.RGB('gray')
# i.show()

# path = "/home/godeta/PycharmProjects/Disparity_Pipeline/results/Res/Disparity-HD"
# Visualizer(path).run()

# R2 = CameraSetup(from_file="/home/godeta/PycharmProjects/Disparity_Pipeline/Setup_Camera.yaml")
#


# img_src = R2.cameras['RGB'].im_calib
# img_dst = R2.cameras['IR'].im_calib
#
# DETECTOR = {'SIFT_SCALE': SIFTFeatureScaleSpace, 'SIFT': SIFTFeature,}
# MATCHER = {'NN': match_nn, 'MNN': match_mnn, 'SNN': match_snn, 'SMNN': match_smnn, 'FGINN': match_fginn, 'ADALAM': match_adalam}
# for key in DETECTOR.keys():
#     for k in MATCHER.keys():
#         k_gen = KeypointsGenerator(device=torch.device('cuda'), detector=key, matcher=k)
#         k_gen(img_src, img_dst, min_kpt=8, th=0.85, draw_result=True)
#         print(key, k)


# R2.cameras['RGB'].__getitem__(333).show()
#
# R2.calibration_for_stereo('RGB', 'RGB2')
# R2.stereo_pair['RGB&RGB2'].show_image_calibration()

# R2.calibration_for_stereo('IR', 'RGB')
# R2.stereo_pair['RGB&IR'].show_image_calibration()
#
# R2.calibration_for_stereo('IR', 'IR2')
# # R2.stereo_pair['IR&IR2'].show_image_calibration()

# for key in R2.stereo_pair:
#     if key != 'left' and key != 'right' and key != 'name':
#         R2.stereo_pair[key].show_image_calibration()

#
# RGB2 = R2('RGB2', print_info=True)
# print(R2)
# print(R2.pos)

# R2.update_camera_ref('IR')
#
# R2.save('/home/godeta/PycharmProjects/Disparity_Pipeline/')
# R2 = CameraSetup(from_file="/home/godeta/PycharmProjects/Disparity_Pipeline/Setup_Camera.yaml")
# R2.update_camera_ref('IR')
# R2.recover_pose_from_keypoints('RGB2', t=[0.340, 0, 0])
# R2.calibration_for_stereo('RGB', 'RGB2')
# R2.calibration_for_stereo('IR', 'RGB')
# R2.calibration_for_stereo('IR', 'IR2')
# print(R2.disparity_ready('RGB2', 'RGB'))
# R2.stereo_pair('RGB', 'IR').show_image_calibration()
# R2.update_camera_ref('RGB2')
# print(R2.pos)
# print(R2)

# R2.calibration_for_stereo('RGB', 'RGB2')
# R2.stereo_pair('RGB', 'RGB2').show_image_calibration()


# R2.save('/home/godeta/PycharmProjects/Disparity_Pipeline/')
# print("####################")
# print(R2.disparity_ready('RGB', 'IR'))
# print("####################")
# print(R2.disparity_ready())
# R2.register_camera('IR')
# print(R2.cameras['IR'].__dict__)
# R2.save('/home/godeta/PycharmProjects/Disparity_Pipeline/')
# breakpoint()
#
# #
# path = '/home/godeta/PycharmProjects/Disparity_Pipeline/dataset/vis_day/left/calibration_image.png'
# path2 = '/home/godeta/PycharmProjects/Disparity_Pipeline/dataset/vis_day/other/calibration_image.png'
#
# im = ImageTensor(path)
# im2 = im[:,None, None]
#
# print(isinstance(im2, ImageTensor))
# print(im2.channel_pos)
# #
# im = im.put_channel_at(2)
# print(im.__dict__, im.shape)
# print(im2.__dict__, im2.shape)
# im = im.match_shape(im2)
# print(im.__dict__, im.shape)
# print(im2.__dict__, im2.shape)
