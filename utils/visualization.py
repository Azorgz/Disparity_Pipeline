import os
from pathlib import Path

import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils
import cv2
import yaml
from matplotlib.cm import get_cmap
import matplotlib as mpl
import matplotlib.cm as cm
import cv2 as cv

from utils.classes import ImageCustom

convert_alpha_number = {55: 7, 56: 8, 57: 9, 52: 4, 53: 5, 54: 6, 49: 1, 50: 2, 51: 3, 48: 0}


def result_visualizer(path: str or Path, target: str, ref: str, start_idx=-1):
    """
    :param path: Path to the result folder
    :param target: str: other, left or right being the targeted image
    :param ref: other, left or right being the projected image
    :param start_idx: int, start idx of the visualisation
    :return: None
    To navigate use the arrows or +/- or specify an index using the num pad and validate with Enter.
    To quit press Escape
    To show/hide the current index press i
    To show/hide the overlay of disparity press d
    To show/hide the validation indexes (only available with the validation done) press v
    """
    target_path, _, target_list = os.walk(f'{path}/input/{target}').__next__()
    new_path, _, new_list = os.walk(f'{path}/reg_images').__next__()
    ref_path, _, ref_list = os.walk(f'{path}/input/{ref}').__next__()
    target_disp_path, _, target_disp_list = os.walk(f'{path}/disp_{target}').__next__()
    ref_disp_path, _, ref_disp_list = os.walk(f'{path}/disp_{ref}').__next__()
    target_list, new_list, ref_list = sorted(target_list), sorted(new_list), sorted(ref_list)
    target_disp_list, ref_disp_list = sorted(target_disp_list), sorted(ref_disp_list)
    if os.path.exists(f'{path}/Validation.yaml'):
        validation_available = True
        with open(f'{path}/Validation.yaml', "r") as file:
            val = yaml.safe_load(file)
    else:
        validation_available = False
    show_validation = False
    show_grad_im = False
    show_disp_overlay = True
    show_idx = True
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (255, 255, 255)
    org_idx = (10, 15)
    thickness = 1
    fontScale = 0.5
    idx = start_idx if 0 <= start_idx <= len(new_list) else 0
    key = 0
    idx_max = len(new_list)
    while key != 27:
        target_im = ImageCustom(f'{target_path}/{target_list[idx]}').BGR()
        new_im = ImageCustom(f'{new_path}/{new_list[idx]}').BGR()
        ref_im = ImageCustom(f'{ref_path}/{ref_list[idx]}').BGR()
        visu = np.hstack([target_im / 510 + new_im / 510, target_im / 510 + ref_im / 510])
        if visu.shape[1] > 1920 or visu.shape[0] > 1080:
            visu = cv.pyrDown(visu)

        if show_grad_im:
            pass
        if show_disp_overlay:
            disp_target = ImageCustom(f'{target_disp_path}/{target_disp_list[idx]}').GRAYSCALE().expand_dims()
            disp_ref = ImageCustom(f'{ref_disp_path}/{ref_disp_list[idx]}').GRAYSCALE().expand_dims()
            disp_overlay_ref = (ref_im / 255) * (disp_ref / 255)
            disp_overlay_target = (target_im / 255) * (disp_target / 255)
            visu = np.vstack([visu, np.hstack(
                [disp_overlay_ref / disp_overlay_ref.max(), disp_overlay_target / disp_overlay_target.max()])])

        if show_idx:
            visu = cv.putText(visu, f'idx : {idx}', org_idx, font, fontScale, color, thickness, cv2.LINE_AA)
        if show_validation and validation_available:
            org_val = int(visu.shape[1]/2+10), visu.shape[0] - 65
            for key, value in val['2. results'].items():
                stats = f'{key} : {value[idx][0]} / {value[idx][1]}'
                color_val = (0, 1, 0) if value[idx][0] >= value[idx][1] else (0, 0, 1)
                visu = cv.putText(visu, stats, org_val, font, fontScale, color_val, thickness, cv2.LINE_AA)
                org_val = org_val[0], org_val[1] + 15

        cv.imshow('Result visionner', visu)
        key = cv.waitKey(0)
        i = 0
        while key in [55, 56, 57, 52, 53, 54, 49, 50, 51, 48]:
            i = i * 10 + convert_alpha_number[key]
            key = cv.waitKey(0)
            if key == 13:
                idx = i
        if key == 81 or key == 82 or key == 45:  # left or up or +
            idx -= 1
        if key == 83 or key == 84 or key == 43:  # right or down or -
            idx += 1
        if key == 100:  # d
            show_disp_overlay = not show_disp_overlay
        if key == 105:  # i
            show_idx = not show_idx
        if key == 118:  # v
            show_validation = not show_validation
        idx = idx % idx_max
        print(key)


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def tensor2numpy(var_dict):
    for key, vars in var_dict.items():
        if isinstance(vars, np.ndarray):
            var_dict[key] = vars
        elif isinstance(vars, torch.Tensor):
            var_dict[key] = vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    return var_dict


def viz_depth_tensor(disp, return_numpy=False, colormap='plasma'):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.cpu().numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz


def visual_control(*args):
    for idx, im in enumerate(args):
        if isinstance(im, torch.Tensor):
            im_show = im.squeeze().cpu().numpy()
        else:
            im_show = im.copy()
        if im.ndim == 3:
            im_show = im_show / im_show.max()
        else:
            im_show = vis_disparity(im_show)
        cv.imshow(f'Image {idx}', im_show)
    cv.waitKey(0)
    cv.destroyAllWindows()