import os
import time
import pandas as pd
from matplotlib import pyplot as plt
from Result_analysis.ResultFrame import ResultFrame

#     # Without Occlusion
#     frame_mean = pd.DataFrame(
#         {f"{f}": res.delta.mean().values},
#         index=res.delta.mean().index)
#     frames_mean = frame_mean.join(frames_mean)
#     frame_std = pd.DataFrame(
#         {f"{f}": res.delta.std().values},
#         index=res.delta.mean().index)
#     frames_std = frame_std.join(frames_std)
#     data = ((res.delta > 0).sum(0) / len(res.delta)).values
#     data[0] = 1 - data[0]
#     frame_pos = pd.DataFrame(
#         {f"{f}": data},
#         index=res.delta.mean().index)
#     frames_pos = frame_pos.join(frames_pos)
#
#     # With Occlusion
#     frame_mean_occ = pd.DataFrame(
#         {f"{f}": res.delta_occ.mean().values},
#         index=res.delta_occ.mean().index)
#     frames_mean_occ = frame_mean_occ.join(frames_mean_occ)
#     frame_std_occ = pd.DataFrame(
#         {f"{f}": res.delta_occ.std().values},
#         index=res.delta_occ.mean().index)
#     frames_std_occ = frame_std_occ.join(frames_std_occ)
#     data = ((res.delta_occ > 0).sum(0) / len(res.delta_occ)).values
#     data[0] = 1 - data[0]
#     frame_pos_occ = pd.DataFrame(
#         {f"{f}": data},
#         index=res.delta_occ.mean().index)
#     frames_pos_occ = frame_pos_occ.join(frames_pos_occ)
#
# with pd.ExcelWriter(f'{folder}.xlsx') as writer:
#     frames_mean.to_excel(writer, sheet_name='Delta moyen')
#     frames_mean_occ.to_excel(writer, sheet_name='Delta occ moyen')
#     frames_std.to_excel(writer, sheet_name='Delta std')
#     frames_std_occ.to_excel(writer, sheet_name='Delta occ std')
#     frames_pos.to_excel(writer, sheet_name='Delta positif')
#     frames_pos_occ.to_excel(writer, sheet_name='Delta occ positif')
#
# time.sleep(1)
### MULTI SETUP CONFIG ############

folder = "camera_intrinsic_ir"
base_path = os.getcwd() + "/../results/"

frames_mean = pd.DataFrame()
frames_std = pd.DataFrame()
frames_pos = pd.DataFrame()
frames_mean_occ = pd.DataFrame()
frames_std_occ = pd.DataFrame()
frames_pos_occ = pd.DataFrame()

for f in reversed(os.listdir(base_path + folder)):
    res = ResultFrame(base_path + folder + '/' + f)
# res1 = ResultFrame(base_path + "camera_position_ir_finer/Depth-Depth")
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
