import os
from pathlib import Path
from typing import Union

import cv2 as cv
import numpy as np
import torch
import yaml
from kornia.utils import get_cuda_device_if_available

from utils.classes import ImageTensor
from utils.gradient_tools import grad_tensor, grad
from utils.transforms import ToTensor


class Visualizer:
    show_validation = False
    show_grad_im = False
    show_occlusion = False
    show_disp_overlay = False
    show_idx = True
    font = cv.FONT_HERSHEY_TRIPLEX
    color = (255, 255, 255)
    org_idx = (10, 20)
    thickness = 1
    fontScale = 0.5
    idx = 0
    key = 0
    convert_alpha_number = {55: 7, 56: 8, 57: 9, 52: 4, 53: 5, 54: 6, 49: 1, 50: 2, 51: 3, 48: 0}
    tensor = False
    window = 0

    def __init__(self, path: Union[str, Path, list] = None, search_exp=False):
        """
        :param path: Path to the result folder
        :return: None
        path ___|input|target
                |reg_images
                |disp_target
                |disp_ref
                |occlusion
                |Validation.yaml
        To navigate use the arrows or +/- or specify an index using the num pad and validate with Enter.
                |     |ref
        To quit press Escape
        To show/hide the current index press i
        To show/hide the overlay of disparity press d
        To show/hide the validation indexes (only available with the validation done) press v
        """
        if path is None or search_exp:
            if path is None:
                p = "/home/godeta/PycharmProjects/Disparity_Pipeline/results"
            else:
                p = path
            path = [p + f'/{d}' for d in os.listdir(p) if os.path.isdir(p + f'/{d}')]
        if isinstance(path, list):
            self.exp_list = []
            self.path = []
            for pa in path:
                self.exp_list.append(*os.listdir(pa + '/image_reg'))
                self.path.append(pa)
        else:
            self.exp_list = os.listdir(path + '/image_reg')
        self.experiment = {}
        for idx, (p, P) in enumerate(zip(self.exp_list, self.path)):
            ref, target = p.split('_to_')
            exp_name = os.path.split(P)[1]
            p = f'{exp_name} - {p}'
            self.exp_list[idx] = p
            self.experiment[p] = {}
            new_path, _, new_list = os.walk(f'{P}/image_reg/{ref}_to_{target}').__next__()
            if os.path.exists(f'{P}/inputs'):
                target_path, _, target_list = os.walk(f'{P}/inputs/{target}').__next__()
                ref_path, _, ref_list = os.walk(f'{P}/inputs/{ref}').__next__()

            elif os.path.exists(f'{P}/dataset.yaml'):
                with open(f'{P}/dataset.yaml', "r") as file:
                    dataset = yaml.safe_load(file)
                target_path, target_list = '', dataset['Files'][target]
                ref_path, ref_list = '', dataset['Files'][ref]
            else:
                target_path, ref_path = None, None
            try:
                self.experiment[p]['occlusion_path'], _, occlusion_list = (
                    os.walk(f'{P}/occlusion/{ref}_to_{target}').__next__())
                self.experiment[p]['occlusion_mask'] = sorted(occlusion_list)
                self.experiment[p]['occlusion_ok'] = True
            except StopIteration:
                self.experiment[p]['occlusion_ok'] = False
                print(f'Occlusion masks wont be available for the {p} couple')
            if target_path is not None:
                self.experiment[p]['target_list'] = [target_path + '/' + n for n in target_list]
                self.experiment[p]['ref_list'] = [ref_path + '/' + n for n in ref_list]
                self.experiment[p]['inputs_available'] = True
            else:
                self.experiment[p]['target_list'], self.experiment[p]['ref_list'] = None, None
                print(f'Inputs images wont be available for experiment {p}')
                self.experiment[p]['inputs_available'] = False
            self.experiment[p]['new_list'] = [new_path + '/' + n for n in sorted(new_list)]
            try:
                self.experiment[p]['target_disp_path'], _, target_disp_list = os.walk(
                    f'{P}/pred_disp/{target}').__next__()
                self.experiment[p]['ref_disp_path'], _, ref_disp_list = os.walk(f'{P}/disp_reg/{ref}').__next__()
                self.experiment[p]['target_disp_list'], self.experiment[p]['ref_disp_list'] = \
                    sorted(target_disp_list), sorted(ref_disp_list)
                self.experiment[p]['disp_ok'] = True
            except StopIteration:
                self.experiment[p]['disp_ok'] = False
                print(f'Disparity images wont be available for the {p} couple')
            try:
                self.experiment[p]['target_depth_path'], _, target_depth_list = os.walk(
                    f'{P}/pred_depth/{target}').__next__()
                self.experiment[p]['ref_depth_path'], _, ref_depth_list = os.walk(f'{P}/depth_reg/{ref}').__next__()
                self.experiment[p]['target_depth_list'], self.experiment[p]['ref_depth_list'] = \
                    sorted(target_depth_list), sorted(ref_depth_list)
                self.experiment[p]['depth_ok'] = True
            except StopIteration:
                self.experiment[p]['depth_ok'] = False
                print(f'Depth images wont be available for the {p} couple')

            if os.path.exists(f'{P}/Validation.yaml'):
                self.experiment[p]['validation_available'] = True
                with open(f'{P}/Validation.yaml', "r") as file:
                    self.experiment[p]['val'] = yaml.safe_load(file)
            else:
                self.experiment[p]['validation_available'] = False
            self.experiment[p]['idx_max'] = len(new_list)
        self.device = get_cuda_device_if_available()
        self.tensor = True
        self.idx = 0
        self.show_disp_overlay = 0

    def run(self):
        exp = self.exp_list[0]
        exp_idx = 0
        experiment = self.experiment[exp]
        while self.key != 27:
            new_im = ImageTensor(f'{experiment["new_list"][self.idx]}').RGB()
            if experiment["inputs_available"]:
                target_im = ImageTensor(f'{experiment["target_list"][self.idx]}').RGB()
                ref_im = ImageTensor(f'{experiment["ref_list"][self.idx]}').RGB().match_shape(
                target_im)
            else:
                target_im = new_im.clone()
                ref_im = new_im.clone()
            if self.show_occlusion:
                mask = 1 - ImageTensor(f'{experiment["occlusion_path"]}/{experiment["occlusion_mask"][self.idx]}').match_shape(
                target_im)
            else:
                mask = 1
            visu = (target_im / 2 + new_im * mask / 2).vstack(target_im / 2 + ref_im / 2)

            if self.show_grad_im:
                grad_im = self._create_grad_im(new_im, ref_im, target_im, mask)
                visu = visu.hstack(grad_im)

            if self.show_disp_overlay:
                if self.show_disp_overlay == 1:
                    disp_overlay = self._create_disp_overlay(experiment, ref_im, target_im, mask)
                    visu = visu.hstack(disp_overlay)
                elif self.show_disp_overlay >= 2:
                    depth_overlay = self._create_depth_overlay(experiment, ref_im, target_im, mask)
                    visu = visu.hstack(depth_overlay)

            h, w = ref_im.shape[-2:]

            while visu.shape[3] > 1920 or visu.shape[2] > 1080:
                visu = visu.pyrDown()
                h, w = h // 2, w // 2

            visu = visu.opencv()
            if self.show_idx:
                visu = cv.putText(visu, f'idx : {self.idx}', self.org_idx, self.font, self.fontScale, self.color,
                                  self.thickness, cv.LINE_AA)
            if self.show_grad_im:
                org = self.org_idx[0] + w, self.org_idx[1]
                visu = cv.putText(visu, f'Image grad : {"with tensor" if self.tensor else "with numpy"}', org,
                                  self.font,
                                  self.fontScale, self.color,
                                  self.thickness, cv.LINE_AA)
            if self.show_validation and experiment['validation_available']:
                org_val = 10, visu.shape[0] - 65
                for key, value in experiment['val']['2. results'].items():
                    if key in exp:
                        k = 2 if self.show_occlusion else 0
                        for key_stat, stat in value.items():
                            stats = f'{key_stat} : {stat[self.idx][0+k]} / {stat[self.idx][1]}'
                            if key_stat == 'rmse':
                                color_val = (0, 0, 255) if stat[self.idx][0+k] >= stat[self.idx][1] else (0, 255, 0)
                            else:
                                color_val = (0, 255, 0) if stat[self.idx][0+k] >= stat[self.idx][1] else (0, 0, 255)
                            visu = cv.putText(visu, stats, org_val, self.font, self.fontScale, color_val,
                                              self.thickness,
                                              cv.LINE_AA)
                            org_val = org_val[0], org_val[1] + 15

            cv.imshow(f'Experience {exp}', visu)
            self.key = cv.waitKey(0)
            i = 0
            while self.key in [55, 56, 57, 52, 53, 54, 49, 50, 51, 48]:
                i = i * 10 + self.convert_alpha_number[self.key]
                self.key = cv.waitKey(0)
                if self.key == 13:
                    self.idx = i
            if self.key == 81 or self.key == 82 or self.key == 45:  # left or up or +
                self.idx -= 1
            if self.key == 83 or self.key == 84 or self.key == 43:  # right or down or -
                self.idx += 1
            if self.key == 100:  # d
                self.show_disp_overlay += 1
                if self.show_disp_overlay == 1:
                    if not experiment['disp_ok']:
                        self.show_disp_overlay += 1
                if self.show_disp_overlay >= 2:
                    if not experiment['depth_ok']:
                        self.show_disp_overlay = 0
            if self.key == 105:  # i
                self.show_idx = not self.show_idx
            if self.key == 118:  # v
                self.show_validation = not self.show_validation
            if self.key == 103:  # g
                self.show_grad_im = not self.show_grad_im
            if self.key == 116 and self.device:  # t
                self.tensor = not self.tensor
            if self.key == 111 and experiment['occlusion_ok']:  # t
                self.show_occlusion = not self.show_occlusion
            if self.key == 9:
                exp_idx += 1
                exp_idx = exp_idx % len(self.exp_list)
                exp = self.exp_list[exp_idx]
                experiment = self.experiment[exp]
                if self.show_disp_overlay == 1:
                    if not experiment['disp_ok']:
                        self.show_disp_overlay += 1
                if self.show_disp_overlay >= 2:
                    if not experiment['depth_ok']:
                        self.show_disp_overlay = 0
            self.idx = self.idx % self.experiment[exp]['idx_max']
            # print(self.key)

    def _create_grad_im(self, new_im, ref_im, target_im, mask):
        if self.tensor:
            grad_new = grad_tensor(new_im, self.device)
            grad_ref = grad_tensor(ref_im, self.device)
            grad_target = grad_tensor(target_im, self.device)
        else:
            grad_new = grad(new_im)
            grad_ref = grad(ref_im)
            grad_target = grad(target_im)
        im = (grad_new*mask / 2 + grad_target*mask / 2).vstack(grad_ref / 2 + grad_target / 2)

        return im

    def _create_disp_overlay(self, experiment, ref_im, target_im, mask):
        disp_target = ImageTensor(
            f'{experiment["target_disp_path"]}/{experiment["target_disp_list"][self.idx]}').RGB()
        disp_ref = ImageTensor(f'{experiment["ref_disp_path"]}/{experiment["ref_disp_list"][self.idx]}').RGB()
        disp_overlay_ref = disp_ref.match_shape(ref_im)
        disp_overlay_target = disp_target.match_shape(ref_im) * mask
        return (disp_overlay_ref / disp_overlay_ref.max()).vstack(disp_overlay_target / disp_overlay_target.max())

    def _create_depth_overlay(self, experiment, ref_im, target_im, mask):
        depth_target = ImageTensor(
            f'{experiment["target_depth_path"]}/{experiment["target_depth_list"][self.idx]}').RGB()
        depth_ref = ImageTensor(f'{experiment["ref_depth_path"]}/{experiment["ref_depth_list"][self.idx]}').RGB()
        depth_overlay_ref = depth_ref.match_shape(ref_im)
        depth_overlay_target = depth_target * mask
        return (depth_overlay_ref / depth_overlay_ref.max()).vstack(depth_overlay_target / depth_overlay_target.max())


if __name__ == '__main__':
    path = "/home/godeta/PycharmProjects/Disparity_Pipeline/results/Camera_position"
    Visualizer(path, search_exp=True).run()
