import os
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
import yaml

from utils.classes import ImageCustom
from utils.gradient_tools import grad_tensor, grad
from utils.transforms import ToTensor, ToNumpy


class Visualizer:
    show_validation = False
    show_grad_im = False
    show_disp_overlay = True
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

    def __init__(self, path: str or Path, target: str, ref: str):
        """
        :param path: Path to the result folder
        :param target: str: other, left or right being the targeted image
        :param ref: other, left or right being the projected image
        :return: None
        path ___|input|target
                |reg_images
                |disp_target
                |disp_ref
                |Validation.yaml
        To navigate use the arrows or +/- or specify an index using the num pad and validate with Enter.
                |     |ref
        To quit press Escape
        To show/hide the current index press i
        To show/hide the overlay of disparity press d
        To show/hide the validation indexes (only available with the validation done) press v
        """
        try:
            self.target_path, _, target_list = os.walk(f'{path}/input/{target}').__next__()
            self.new_path, _, new_list = os.walk(f'{path}/reg_images').__next__()
            self.ref_path, _, ref_list = os.walk(f'{path}/input/{ref}').__next__()
            self.target_disp_path, _, target_disp_list = os.walk(f'{path}/disp_{target}').__next__()
            self.ref_disp_path, _, ref_disp_list = os.walk(f'{path}/disp_{ref}').__next__()
        except FileNotFoundError:
            print('The given directory does not contain the needed folders')
        self.target_list, self.new_list, self.ref_list = sorted(target_list), sorted(new_list), sorted(ref_list)
        self.target_disp_list, self.ref_disp_list = sorted(target_disp_list), sorted(ref_disp_list)
        if os.path.exists(f'{path}/Validation.yaml'):
            self.validation_available = True
            with open(f'{path}/Validation.yaml', "r") as file:
                self.val = yaml.safe_load(file)
        else:
            self.validation_available = False
        self.idx_max = len(new_list)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.tensor = True
        else:
            self.device = None

    def run(self):
        while self.key != 27:
            target_im = ImageCustom(f'{self.target_path}/{self.target_list[self.idx]}').BGR()
            new_im = ImageCustom(f'{self.new_path}/{self.new_list[self.idx]}').BGR()
            ref_im = ImageCustom(f'{self.ref_path}/{self.ref_list[self.idx]}').BGR()
            visu = np.vstack([target_im / 510 + new_im / 510, target_im / 510 + ref_im / 510])
            h, w = ref_im.shape[:2]

            if self.show_grad_im:
                grad_im = self._create_grad_im(new_im, ref_im, target_im)
                visu = np.hstack([visu, grad_im])

            if self.show_disp_overlay:
                disp_overlay = self._create_disp_overlay(ref_im, target_im)
                visu = np.hstack([visu, disp_overlay])

            if visu.shape[1] > 1920 or visu.shape[0] > 1080:
                visu = cv.pyrDown(visu)
                h, w = int(h/2), int(w/2)
            if self.show_idx:
                visu = cv.putText(visu, f'idx : {self.idx}', self.org_idx, self.font, self.fontScale, self.color,
                                  self.thickness, cv.LINE_AA)
            if self.show_grad_im:
                org = self.org_idx[0] + w, self.org_idx[1]
                visu = cv.putText(visu, f'Image grad : {"with tensor" if self.tensor else "with numpy"}', org, self.font,
                           self.fontScale, self.color,
                           self.thickness, cv.LINE_AA)
            if self.show_validation and self.validation_available:
                org_val = 10, visu.shape[0] - 65
                for key, value in self.val['2. results'].items():
                    stats = f'{key} : {value[self.idx][0]} / {value[self.idx][1]}'
                    color_val = (0, 1, 0) if value[self.idx][0] >= value[self.idx][1] else (0, 0, 1)
                    visu = cv.putText(visu, stats, org_val, self.font, self.fontScale, color_val, self.thickness,
                                      cv.LINE_AA)
                    org_val = org_val[0], org_val[1] + 15

            cv.imshow('Result visionner', visu)
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
                self.show_disp_overlay = not self.show_disp_overlay
            if self.key == 105:  # i
                self.show_idx = not self.show_idx
            if self.key == 118:  # v
                self.show_validation = not self.show_validation
            if self.key == 103:  # g
                self.show_grad_im = not self.show_grad_im
            if self.key == 116 and self.device:  # t
                self.tensor = not self.tensor
            self.idx = self.idx % self.idx_max
            print(key)

    def _create_grad_im(self, new_im, ref_im, target_im):
        if self.tensor:
            to_tensor = ToTensor()
            to_numpy = ToNumpy()
            grad_new = ImageCustom(to_numpy(grad_tensor(to_tensor(new_im, self.device), self.device))).BGR()
            grad_ref = ImageCustom(to_numpy(grad_tensor(to_tensor(ref_im, self.device), self.device))).BGR()
            grad_target = ImageCustom(to_numpy(grad_tensor(to_tensor(target_im, self.device), self.device))).BGR()
        else:
            grad_new = grad(new_im)
            grad_ref = grad(ref_im)
            grad_target = grad(target_im)
        # im = np.vstack([grad_new / 256, grad_target / 256])
        im = np.vstack([grad_new / 510 + grad_target / 510, grad_ref / 510 + grad_target / 510])

        return im

    def _create_disp_overlay(self, ref_im, target_im):
        disp_target = ImageCustom(
            f'{self.target_disp_path}/{self.target_disp_list[self.idx]}').GRAYSCALE().expand_dims()
        disp_ref = ImageCustom(f'{self.ref_disp_path}/{self.ref_disp_list[self.idx]}').GRAYSCALE().expand_dims()
        disp_overlay_ref = (ref_im / 255) * (disp_ref / 255)
        disp_overlay_target = (target_im / 255) * (disp_target / 255)
        return np.vstack([disp_overlay_ref / disp_overlay_ref.max(), disp_overlay_target / disp_overlay_target.max()])
