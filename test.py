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
res = ResultFrame(base_path + "camera_position_ir_finer/Depth-Depth")
cam = 'IR'
setup = 'fine test'

if setup == 'raw test':
    vec_x = np.arange(0, 8 * 1e-2, 1e-2)
    vec_z = np.arange(0, 6e-2, 1e-2)
    vec_y = np.arange(0, 6e-2, 1e-2)
    vec_alpha = (np.arange(0, 4, 1) / 180 * np.pi)
    if cam == 'IR':
        f_, px_size_ = 14e-3, 16.4e-6
    else:
        f_, px_size_ = 6e-3, 3.45e-6
    vec_f = np.arange(0.75, 2, 0.25) * f_ * 1e3
    vec_px = np.arange(0.75, 2, 0.25) * px_size_ * 1e6

elif setup == 'fine test':
    vec_x = np.arange(0, 9 * 1e-2, 1e-2)
    vec_x = (vec_x - vec_x.max() / 2)
    vec_z = np.arange(-6e-2, 1.5e-2, 5e-3)
    vec_z = (vec_z - vec_z.max() / 2)
    vec_y = np.arange(0, 1.5e-1, 2.5e-2)
    vec_y = (vec_y - vec_y.max() / 2)
    vec_alpha = (np.arange(0, 3, 1) / 180 * np.pi)
    vec_alpha = (vec_alpha - vec_alpha.max() / 2)
    if cam == 'IR':
        f_, px_size_ = 14e-3, 16.4e-6
    else:
        f_, px_size_ = 6e-3, 3.45e-6
    vec_f = np.arange(0.75, 2, 0.05) * f_ * 1e3
    vec_px = np.arange(0.75, 2, 0.05) * px_size_ * 1e6

else:
    vec_x = np.array([0])
    vec_z = np.array([0])
    vec_y = np.array([0])
    vec_alpha = np.array([0])
    vec_f = np.array([0])
    vec_px = np.array([0])


a = np.arange(0, len(vec_alpha), 1)
z_coeff = len(vec_alpha)
z = np.arange(0, len(vec_z), 1)

y_coeff = len(vec_z) * z_coeff
y = np.arange(0, len(vec_y), 1)

x_coeff = len(vec_y) * y_coeff
x = np.arange(0, len(vec_x), 1)

f_coeff = len(vec_px)
f = np.arange(0, len(vec_f), 1)
px = np.arange(0, len(vec_px), 1)


def cal_idx(variable):
    if variable == 'x':
        x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
        result = [[y__ + z__ + a_ for y__ in y_ for z__ in z_ for a_ in a] + x__ for x__ in x_]
    elif variable == 'y':
        x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
        result = [[x__ + z__ + a_ for x__ in x_ for z__ in z_ for a_ in a] + y__ for y__ in y_]
    elif variable == 'z':
        x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
        result = [[y__ + x__ + a_ for y__ in y_ for x__ in x_ for a_ in a] + z__ for z__ in z_]
    elif variable == 'a':
        x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
        result = [[x__ + y__ + z__ for x__ in x_ for y__ in y_ for z__ in z_] + a_ for a_ in a]
    elif variable == 'f':
        f1 = f * f_coeff
        result = [[px_ for px_ in px] + f__ for f__ in f1]
    elif variable == 'px':
        f1 = f * f_coeff
        result = [[f__ for f__ in f1] + px_ for px_ in px]
    else:
        result = np.array(0).tolist()
    return result


idx_x = cal_idx('x')
idx_y = cal_idx('y')
idx_z = cal_idx('z')
idx_a = cal_idx('a')
idx_f = cal_idx('f')
idx_px = cal_idx('px')

val = res.delta_full.combine_column('nec-rmse+psnr+ms_ssim+ssim').values

v_x = val[idx_x].mean(1)
v_y = val[idx_y].mean(1)
v_z = val[idx_z].mean(1)
v_a = val[idx_a].mean(1)
m = np.array([v_x.min(), v_y.min(), v_z.min(), v_a.min()]).min()
M = np.array([v_x.max(), v_y.max(), v_z.max(), v_a.max()]).max()

plt.plot(vec_x, v_x)
plt.plot(vec_y, v_y)
plt.plot(vec_z, v_z)
plt.plot(vec_alpha, v_a)
plt.vlines(0, m, M, colors='k', linestyles='solid', label='delta = 0')
plt.legend(['dx m', 'dy m', 'dz m', 'da rad'])
#
# v_f = val[idx_f].mean(1)
# v_px = val[idx_px].mean(1)
#
# m = np.array([v_f.min(), v_px.min()]).min()
# M = np.array([v_f.max(), v_px.max()]).max()
#
# plt.plot(vec_f, v_f, color='b')
# plt.plot(vec_px, v_px, color='r')
#
# plt.vlines(f_ * 1e3, m, M, colors='b', linestyles='solid', label='delta = 0')
# plt.vlines(px_size_ * 1e6, m, M, colors='r', linestyles='solid', label='delta = 0')
#
# plt.legend(['df mm', 'dpx um'])

plt.show()


time.sleep(1)

# Best Depth 1186 / dz=5, dx=4, dy=2, da=1
# Best Depth 1984 / dz=1, dx=7, dy=2, da=1










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
