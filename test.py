import os
import time
import open3d as o3d
import numpy as np
import torch
from kornia.geometry import depth_to_3d
from kornia.morphology import dilation
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot
from open3d.examples.pipelines.icp_registration import draw_registration_result
from pandas import DataFrame
from torch import Tensor

from Networks.KenburnDepth.KenburnDepth import KenburnDepth
# import torch
from Result_analysis.ResultFrame import ResultFrame
# from utils.classes import ImageTensor
# from utils.classes.Visualizer import Visualizer
# from utils.classes.Image import ImageTensor
import numpy as np

from module.SetupCameras import CameraSetup
from utils.classes import ImageTensor, Metric_nec_tensor
from utils.classes.Image import DepthTensor
from utils.gradient_tools import grad_tensor
from utils.misc import time_fct

# import cv2 as cv
# from utils.registration_tools import SIFT
# from utils.visualization import result_visualizer, visual_control
from utils.gradient_tools import grad_tensor


device = torch.device('cuda')
R = CameraSetup(from_file="/home/aurelien/PycharmProjects/Disparity_Pipeline/Setup_Camera.yaml")
NN = KenburnDepth(path_ckpt=os.getcwd() + "/Networks/KenburnDepth/pretrained",
                  semantic_adjustment=False,
                  device=device)
# Depth
src = 'IR'
dst = 'RGB'

# im_dst, idx = R.cameras[dst].random_image()
im_dst = R.cameras['RGB'].__getitem__(0)
im_src = R.cameras[src].__getitem__(0).RGB('gray')
im_src_new = ImageTensor("/home/aurelien/PycharmProjects/Disparity_Pipeline/results/methods_comparison/Depth-Depth/image_reg/IR_to_RGB/IR_000-Setup_Camera.png")
matrix_dst = R.cameras[dst].intrinsics[:, :3, :3]
f_dst = R.cameras[dst].f
var_dst = {'focal': f_dst, 'intrinsics': matrix_dst}
matrix_src = R.cameras[src].intrinsics[:, :3, :3]
f_src = R.cameras[src].f
var_src = {'focal': f_src, 'intrinsics': matrix_src}


# DepthTensor(NN(Tensor(im_dst.pyrDown().pyrDown()), **var_dst)[0].clip(0, 200)).show()
# DepthTensor(NN(Tensor(im_src.pyrDown()), **var_src)[0].clip(0, 200)).show()
#
# DepthTensor(NN(Tensor(im_dst.pyrDown()), **var_dst)[0].clip(0, 200)).show()
# DepthTensor(NN(Tensor(im_src), **var_src)[0].clip(0, 200)).show()
#
# DepthTensor(NN(Tensor(im_dst), **var_dst)[0].clip(0, 200)).show()
# DepthTensor(NN(Tensor(im_src.pyrUp()), **var_src)[0].clip(0, 200)).show()

metric = Metric_nec_tensor(device)

old = metric(im_dst, im_src)
new = metric(im_dst, im_src_new)

time.time()



# base_path = os.getcwd() + "/results/"
# res = ResultFrame(base_path + "Test/Depth-Disparity")
# # res1 = ResultFrame(base_path + "camera_position_ir_finer/Depth-Depth")
# cam = 'IR'
# setup = 'raw test'
#
# if setup == 'raw test':
#     vec_x = np.arange(0, 8 * 1e-2, 1e-2)
#     vec_z = np.arange(0, 6e-2, 1e-2)
#     vec_y = np.arange(0, 6e-2, 1e-2)
#     vec_alpha = (np.arange(0, 4, 1) / 180 * np.pi)
#     if cam == 'IR':
#         f_, px_size_ = 14e-3, 16.4e-6
#     else:
#         f_, px_size_ = 6e-3, 3.45e-6
#     vec_f = np.arange(0.75, 2, 0.25) * f_ * 1e3
#     vec_px = np.arange(0.75, 2, 0.25) * px_size_ * 1e6
#
# elif setup == 'fine test':
#     vec_x = np.arange(0, 9 * 1e-2, 9e-2)
#     vec_x = (vec_x - vec_x.max() / 2)
#     vec_z = np.arange(-6e-2, 1.5e-2, 5e-4)
#     vec_z = (vec_z - vec_z.max() / 2)
#     vec_y = np.arange(0, 1.25e-1, 2.5e-1)
#     vec_y = (vec_y - vec_y.max() / 2)
#     vec_alpha = (np.arange(0, 3, 4) / 180 * np.pi)
#     vec_alpha = (vec_alpha - vec_alpha.max() / 2)
#     if cam == 'IR':
#         f_, px_size_ = 14e-3, 16.4e-6
#     else:
#         f_, px_size_ = 6e-3, 3.45e-6
#     vec_f = np.arange(0.75, 2, 0.05) * f_ * 1e3
#     vec_px = np.arange(0.75, 2, 0.05) * px_size_ * 1e6
#
# else:
#     vec_x = np.array([0])
#     vec_z = np.array([0])
#     vec_y = np.array([0])
#     vec_alpha = np.array([0])
#     vec_f = np.array([0])
#     vec_px = np.array([0])
#
#
# a = np.arange(0, len(vec_alpha), 1)
# z_coeff = len(vec_alpha)
# z = np.arange(0, len(vec_z), 1)
#
# y_coeff = len(vec_z) * z_coeff
# y = np.arange(0, len(vec_y), 1)
#
# x_coeff = len(vec_y) * y_coeff
# x = np.arange(0, len(vec_x), 1)
#
# f_coeff = len(vec_px)
# f = np.arange(0, len(vec_f), 1)
# px = np.arange(0, len(vec_px), 1)
#
#
# def cal_idx(variable):
#     if variable == 'x':
#         x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
#         result = [[y__ + z__ + a_ for y__ in y_ for z__ in z_ for a_ in a] + x__ for x__ in x_]
#     elif variable == 'y':
#         x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
#         result = [[x__ + z__ + a_ for x__ in x_ for z__ in z_ for a_ in a] + y__ for y__ in y_]
#     elif variable == 'z':
#         x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
#         result = [[y__ + x__ + a_ for y__ in y_ for x__ in x_ for a_ in a] + z__ for z__ in z_]
#     elif variable == 'a':
#         x_, y_, z_ = x * x_coeff, y * y_coeff, z * z_coeff
#         result = [[x__ + y__ + z__ for x__ in x_ for y__ in y_ for z__ in z_] + a_ for a_ in a]
#     elif variable == 'f':
#         f1 = f * f_coeff
#         result = [[px_ for px_ in px] + f__ for f__ in f1]
#     elif variable == 'px':
#         f1 = f * f_coeff
#         result = [[f__ for f__ in f1] + px_ for px_ in px]
#     else:
#         result = np.array(0).tolist()
#     return result
#
#
# idx_x = cal_idx('x')
# idx_y = cal_idx('y')
# idx_z = cal_idx('z')
# idx_a = cal_idx('a')
# idx_f = cal_idx('f')
# idx_px = cal_idx('px')
#
#
# val = res.delta_full.combine_column('nec-rmse+psnr+ms_ssim+ssim').values
# # val1 = res1.delta_full.combine_column('nec-rmse+psnr+ms_ssim+ssim').values
#
# v_x = val[idx_x].mean(1)
# v_y = val[idx_y].mean(1)
# v_z = val[idx_z].mean(1)
# v_a = val[idx_a].mean(1)
# m = np.array([v_x.min(), v_y.min(), v_z.min(), v_a.min()]).min()
# M = np.array([v_x.max(), v_y.max(), v_z.max(), v_a.max()]).max()
#
# plt.plot(vec_x, v_x)
# plt.plot(vec_y, v_y)
# plt.plot(vec_z, v_z)
# plt.plot(vec_alpha, v_a)
# plt.vlines(0, m, M, colors='k', linestyles='solid', label='delta = 0')
# # plt.vlines(vec_z[(val).argmax()], m, M, colors='k', linestyles='solid', label=f'delta = {vec_z[(val).argmax()]}')
# # plt.legend(['dz m'])
# plt.legend(['dx m', 'dy m', 'dz m', 'da rad'])
#
# #
# # v_f = val[idx_f].mean(1)
# # v_px = val[idx_px].mean(1)
# #
# # m = np.array([v_f.min(), v_px.min()]).min()
# # M = np.array([v_f.max(), v_px.max()]).max()
# #
# # plt.plot(vec_f, v_f, color='b')
# # plt.plot(vec_px, v_px, color='r')
# #
# # plt.vlines(f_ * 1e3, m, M, colors='b', linestyles='solid', label='delta = 0')
# # plt.vlines(px_size_ * 1e6, m, M, colors='r', linestyles='solid', label='delta = 0')
# #
# # plt.legend(['df mm', 'dpx um'])
#
# plt.show()
#

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
