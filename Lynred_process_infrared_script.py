import os

import numpy as np
import torch
from kornia.enhance import sharpness
from kornia.filters import bilateral_blur
from kornia.morphology import opening, closing
from tqdm import tqdm

from utils.ImagesCameras import ImageTensor
from utils.misc import name_generator


## SEQUENCE PART #################################

for i in [1, 2, 3]:
    path_infrared_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/infrared_1_corrected/"
    new_path_infrared_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/sorted/infrared_1/"
    path_infrared_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/infrared_2_corrected/"
    new_path_infrared_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/sorted/infrared_2/"
    path_visible_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/visible_1/"
    new_path_visible_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/sorted/visible_1/"
    path_visible_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/visible_2/"
    new_path_visible_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{1 + i}/sorted/visible_2/"
    os.makedirs(new_path_infrared_1, exist_ok=True)
    os.makedirs(new_path_infrared_2, exist_ok=True)
    os.makedirs(new_path_visible_1, exist_ok=True)
    os.makedirs(new_path_visible_2, exist_ok=True)

    histo = None
    base_name = os.listdir(path_infrared_1)[0].split('_')[0] + '_' + os.listdir(path_infrared_1)[0].split('_')[1] + '_'
    first_idx = np.min([int(x.replace(base_name, '').replace('.png', '')) for x in os.listdir(path_infrared_1)])
    for idx in tqdm(range(len(os.listdir(path_infrared_1)))):
        im = base_name + f'{first_idx + idx}.png'
        ir = ImageTensor(path_infrared_1 + im)
        if histo is None:
            histo = ir.hist()
        else:
            histo = histo.__add__(ir.hist())
    mini, maxi = histo.clip()
    for idx in tqdm(range(len(os.listdir(path_infrared_1)))):
        im = base_name + f'{first_idx + idx}.png'
        ir = ImageTensor(path_infrared_1 + im)
        rgb = ImageTensor(path_visible_1 + im)
        ir = ir.histo_equalization(mini=mini, maxi=maxi)
        ir.name = f"IR1_{name_generator(idx, max_number=len(os.listdir(path_infrared_1)))}"
        ir.save(new_path_infrared_1)
        rgb.name = f"RGB1_{name_generator(idx, max_number=len(os.listdir(path_visible_1)))}"
        rgb.save(new_path_visible_1)

    histo = None
    base_name = os.listdir(path_infrared_2)[0].split('_')[0] + '_' + os.listdir(path_infrared_2)[0].split('_')[1] + '_'
    first_idx = np.min([int(x.replace(base_name, '').replace('.png', '')) for x in os.listdir(path_infrared_2)])
    for idx in tqdm(range(len(os.listdir(path_infrared_2)))):
        im = base_name + f'{first_idx + idx}.png'
        ir = ImageTensor(path_infrared_2 + im)
        if histo is None:
            histo = ir.hist()
        else:
            histo = histo.__add__(ir.hist())

    mini, maxi = histo.clip()
    for idx in tqdm(range(len(os.listdir(path_infrared_2)))):
        im = base_name + f'{first_idx + idx}.png'
        ir = ImageTensor(path_infrared_2 + im)
        rgb = ImageTensor(path_visible_2 + im)
        ir = ir.histo_equalization(mini=mini, maxi=maxi)
        ir.name = f"IR2_{name_generator(idx, max_number=len(os.listdir(path_infrared_2)))}"
        ir.save(new_path_infrared_2)
        rgb.name = f"RGB2_{name_generator(idx, max_number=len(os.listdir(path_visible_2)))}"
        rgb.save(new_path_visible_2)
## CALIB PART #################################
# Processing fct
# def proc_inf1(img, cam=1):
#     img = img.pyrUp()
#     if cam == 1:
#         img.data = img.histo_equalization(filtering=True).clip(0, 0.55)
#         img.normalize(in_place=True)
#         img.data = img ** 1.5
#         img.data = bilateral_blur(img, (3, 3), 0.1, (1.5, 1.5))
#         img.data = opening(img, torch.ones(3, 3))
#         img.data = closing(img, torch.ones(3, 3))
#     else:
#         img.data = img.histo_equalization(filtering=False).clip(0, 0.35)
#         img.normalize(in_place=True)
#         img.data = img ** 1.5
#         img.data = bilateral_blur(img, (3, 3), 0.1, (1.5, 1.5))
#         img.data = opening(img, torch.ones(5, 5))
#     return img.pyrDown()
#
#
# def proc_inf3(img, cam=1):
#     img = img.pyrUp()
#     img.data = img.histo_equalization(filtering=False).clip(0, 0.35)
#     img.normalize(in_place=True)
#     img.data = img ** 1.5
#     img.data = sharpness(img, 2)
#     return img.pyrDown()
#
#
# def proc_inf4(img, cam=1):
#     img = img.pyrUp()
#     if cam == 1:
#         img.data = img.histo_equalization(filtering=True).clip(0, 0.55)
#         img.normalize(in_place=True)
#         img.data = img ** 1.5
#         img.data = bilateral_blur(img, (3, 3), 0.1, (1.5, 1.5))
#         img.data = opening(img, torch.ones(3, 3))
#         img.data = closing(img, torch.ones(3, 3))
#     else:
#         img.data = img.histo_equalization(filtering=False).clip(0.08, 0.15)
#         img.normalize(in_place=True)
#         img.data = img ** 1.5
#         img.data = bilateral_blur(img, (3, 3), 0.1, (1.5, 1.5))
#         img.data = opening(img, torch.ones(7, 7))
#     return img.pyrDown()
#
#
# process_fct = {1: proc_inf1, 3: proc_inf3, 4: proc_inf4}
#
# for i in [1, 3, 4]:
#     path_infrared_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/infrared_1_calibration/"
#     new_path_infrared_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/infrared_1/"
#
#     path_infrared_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/infrared_2_calibration/"
#     new_path_infrared_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/infrared_2/"
#
#     path_visible_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/visible_1_calibration/"
#     new_path_visible_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/visible_1/"
#
#     path_visible_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/visible_2_calibration/"
#     new_path_visible_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/visible_2/"
#
#     os.makedirs(new_path_infrared_1, exist_ok=True)
#     os.makedirs(new_path_infrared_2, exist_ok=True)
#     os.makedirs(new_path_visible_1, exist_ok=True)
#     os.makedirs(new_path_visible_2, exist_ok=True)
#
#     histo = None
#     base_name = os.listdir(path_infrared_1)[0].split('_')[0] + '_' + os.listdir(path_infrared_1)[0].split('_')[1] + '_'
#     first_idx = np.min([int(x.replace(base_name, '').replace('.png', '')) for x in os.listdir(path_infrared_1)])
#
#     for idx in tqdm(range(len(os.listdir(path_infrared_1)))):
#         im = base_name + f'{first_idx + idx}.png'
#         ir = ImageTensor(path_infrared_1 + im)
#         if histo is None:
#             histo = ir.hist()
#         else:
#             histo = histo.__add__(ir.hist())
#     mini, maxi = histo.clip()
#     for idx in tqdm(range(len(os.listdir(path_infrared_1)))):
#         im = base_name + f'{first_idx + idx}.png'
#         ir = ImageTensor(path_infrared_1 + im)
#         rgb = ImageTensor(path_visible_1 + im)
#         ir = process_fct[i](ir, cam=1)
#         ir.name = f"calib_infrared_1_{name_generator(idx, max_number=len(os.listdir(path_infrared_1)))}"
#         rgb.name = f"calib_visible_1_{name_generator(idx, max_number=len(os.listdir(path_infrared_1)))}"
#         ir.save(new_path_infrared_1)
#         rgb.save(new_path_visible_1)
#
#     histo = None
#     base_name = os.listdir(path_infrared_2)[0].split('_')[0] + '_' + os.listdir(path_infrared_2)[0].split('_')[1] + '_'
#     first_idx = np.min([int(x.replace(base_name, '').replace('.png', '')) for x in os.listdir(path_infrared_2)])
#     for idx in tqdm(range(len(os.listdir(path_infrared_2)))):
#         im = base_name + f'{first_idx + idx}.png'
#         ir = ImageTensor(path_infrared_2 + im)
#         if histo is None:
#             histo = ir.hist()
#         else:
#             histo = histo.__add__(ir.hist())
#     mini, maxi = histo.clip()
#     for idx in tqdm(range(len(os.listdir(path_infrared_2)))):
#         im = base_name + f'{first_idx + idx}.png'
#         ir = ImageTensor(path_infrared_2 + im)
#         rgb = ImageTensor(path_visible_2 + im)
#         ir = process_fct[i](ir, cam=2)
#         ir.name = f"calib_infrared_1_{name_generator(idx, max_number=len(os.listdir(path_infrared_2)))}"
#         rgb.name = f"calib_visible_1_{name_generator(idx, max_number=len(os.listdir(path_infrared_2)))}"
#         ir.save(new_path_infrared_2)
#         rgb.save(new_path_visible_2)
