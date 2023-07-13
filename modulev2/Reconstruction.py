import kornia
from kornia.geometry import StereoCamera, project_points
from kornia.morphology import dilation, closing
from matplotlib import pyplot as plt
from torch.nn.functional import grid_sample
import torch
import warnings
import numpy as np

from Config.Config import ConfigPipe
from module import Projection_process
from module.BaseModule import BaseModule
from utils.disparity_tools import reconstruction_from_disparity, disparity_post_process, find_occluded_pixel
from utils.misc import timeit
from utils.visualization import viz_depth_tensor, visual_control

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
import cv2 as cv


class Reconstruction(BaseModule):
    """
    This block implement the reconstruction methods following the reconstruction options
    """

    def __init__(self, config: ConfigPipe):
        super(Reconstruction, self).__init__(config)
        # If the DisparityPipe need to measure the execution time of each block, this parameter will be set to True.

    def _update_conf(self, config):
        self.device = config["device"]["device"]
        self.save_disp = config["save_disp"]
        self.dist_left_right = config['dataset']["position_setup"][0]
        self.dist_left_other = config['dataset']["position_setup"][1]
        self.data_type = config['dataset']["type"]
        self.setup = {"position_setup": config['dataset']["position_setup"],
                      "pred_bidir_disp": config['dataset']["pred_bidir_disp"],
                      "proj_right": config['dataset']["proj_right"],
                      "pred_right_disp": config['dataset']["pred_right_disp"]}
        if self.setup["proj_right"]:
            self.dist_other_closer = self.dist_left_other - self.dist_left_right
        else:
            self.dist_other_closer = self.dist_left_other
        if config["reconstruction"]["method"] == "pytorch":
            self.method = "pytorch"
            self.function = self._reconstruction
        elif config["reconstruction"]["method"] == "fullOpenCv":
            self.function = self._reconstruction
            self.method = "fullOpenCv"
            self.interpolation = config["reconstruction"]["interpolation"]
            self.border = 0 if self.setup['pred_bidir_disp'] else config["reconstruction"]["border"]
            if not isinstance(self.interpolation, int) and isinstance(self.interpolation, str):
                self.interpolation = INTERPOLATION_FLAG[self.interpolation]
            if not isinstance(self.border, int) and isinstance(self.border, str):
                self.border = BORDER_FLAG[self.border]
        elif config["reconstruction"]["method"] == "algo":
            self.function = self._algo_reconstruction
            self.method = "algo"
            self.inpainting = config["reconstruction"]["inpainting"]

    def __str__(self):
        string = super().__str__()
        string += f'Reconstruction method : {self.config["reconstruction"]["method"]}'
        if self.config['dataset']["pred_bidir_disp"]:
            string += f'\nThe projection of disparity will use the both left and right disparity images'
        if self.config['dataset']["type"] == '2vis':
            string += f"\nThe infrared Image will be projected to the " \
                      f"{'right' if self.config['dataset']['proj_right'] else 'left'} RGB image"
        else:
            string += f"\nThe {'right' if self.config['dataset']['proj_right'] else 'left'} " \
                      f"infrared image will be projected to the RGB image"
        string += f"\nThe current cameras configuration is the following : " \
                  f"{Projection_process[self.config['dataset']['projection_process']['step2']][-1]} " \
                  f"with 'R' and 'L' as right and left Stereo and 'IR' or 'VIS' the Other modality\n"
        return string

    def __call__(self, disp, sample, *args, **kwargs):
        # if self.method != "pytorch":
        #     sample["other"] = sample["other"].squeeze().permute(1, 2, 0).cpu().numpy()
        return self.function(disp, sample)

    # def _grid_sample(self, image, disp, bins=None, padding_mode='zeros'):
    #     h, w = disp.shape[-2:]
    #     bins = [0, 10, 15, 20, 25, 30, 35, 40, 200]
    #     if self.method == "pytorch":
    #         grid_reg = kornia.utils.create_meshgrid(h, w, device=self.device).to(image.dtype)
    #         temp = torch.zeros_like(image, device=self.device)
    #         kernel = torch.ones(5, 5).to(self.device)
    #         for idx, b in enumerate(bins[1:]):
    #             temp_disp = torch.zeros_like(disp, device=self.device)
    #             temp_im = torch.zeros_like(image, device=self.device)
    #             mask = (abs(disp) >= bins[idx]) * (abs(disp) <= b)
    #             temp_disp[mask] = disp[mask]
    #             temp_im[mask] = image[mask]
    #             dilated_disp = dilation(temp_disp.unsqueeze(0).unsqueeze(0), kernel).squeeze() / disp.shape[1] * 2
    #             cv.imshow('temp before', (temp/temp_im.max()).cpu().numpy())
    #             cv.waitKey(0)
    #             grid_reg[0, :, :, 0] -= dilated_disp
    #             while len(temp_im.shape) < 4:
    #                 temp_im = temp_im.unsqueeze(0)
    #             a = grid_sample(temp_im, grid_reg, padding_mode=padding_mode).squeeze()
    #             temp[a > 0] = a[a > 0]
    #             cv.imshow('temp after', (temp/temp.max()).cpu().numpy())
    #             cv.waitKey(0)
    #         while len(image.shape) < 4:
    #             image = image.unsqueeze(0)
    #         return grid_sample(image, grid_reg, padding_mode=padding_mode)
    #     else:
    #         x = np.linspace(0, w, w)
    #         y = np.linspace(0, h, h)
    #         mapX, mapY = np.meshgrid(x, y)
    #         mapX, mapY = cv.convertMaps(mapX.astype(np.float32), mapY.astype(np.float32), cv.CV_32FC1)
    #         return cv.remap(image, mapX, mapY, self.interpolation, None, self.border)

    def _grid_sample(self, image, disp, padding_mode='zeros'):
        h, w = disp.shape[-2:]
        if self.method == "pytorch":
            # mask = find_occluded_pixel(disp, self.device, upsample_factor=1/4)
            grid_reg = kornia.utils.create_meshgrid(h, w, device=self.device).to(image.dtype)  # [1 H W 2]
            disp = disp / disp.shape[1] * 2

            kernel = torch.ones(15, 15).to(self.device)
            dilated_disp = closing(disp.unsqueeze(0).unsqueeze(0), kernel).squeeze()
            # dilated_disp[mask == 0] = -10
            grid_reg[0, :, :, 0] -= dilated_disp
            while len(image.shape) < 4:
                image = image.unsqueeze(0)
            return grid_sample(image, grid_reg, padding_mode=padding_mode)
        else:
            x = np.linspace(0, w, w)
            y = np.linspace(0, h, h)
            mapX, mapY = np.meshgrid(x, y)
            mapX, mapY = cv.convertMaps(mapX.astype(np.float32), mapY.astype(np.float32), cv.CV_32FC1)
            return cv.remap(image, mapX, mapY, self.interpolation, None, self.border)

    @timeit
    def _reconstruction(self, disp, sample):
        h, w = disp.shape[-2:]
        if self.method == "fullOpenCv":
            disp = disp.cpu().numpy()
        else:
            disp = torch.tensor(disp, device=self.device)
        if self.dist_other_closer == 0:
            # Case where one of the Stereo Camera is in the same axis as the 'other' modality
            if self.method == "fullOpencv":
                new_disp = np.zeros([h, w])
            else:
                new_disp = torch.zeros(h, w)
            if self.data_type == '2ir':
                image_reg = sample['right'] if self.setup["proj_right"] else sample['left']
            else:
                image_reg = sample["other"]
        else:
            if self.setup["pred_bidir_disp"]:
                # Case where the 'other' modality is between the stereo pair and both disparity images
                # are used to project the new disparity
                factor = abs(self.dist_other_closer / self.dist_left_right)
                if self.setup["proj_right"]:
                    disp_left = -disp[0] * (1 - factor)  # (h, w)
                    disp_right = disp[1] * factor  # (h, w)
                else:
                    disp_left = -disp[0] * factor  # (h, w)
                    disp_right = disp[1] * (1 - factor)  # (h, w)
                image_grid_left = disp[0] * factor  # (h, w)
                image_grid_right = disp[1] * factor  # (h, w)
                image_grid_left = self._grid_sample(image_grid_left, disp_left).squeeze()  # (h, w)
                image_grid_right = self._grid_sample(image_grid_right, disp_right).squeeze()  # (h, w)

                if self.setup["proj_right"]:
                    new_disp = image_grid_right.clone()
                    mask = self.mask_bidir(new_disp)
                    new_disp[mask] = image_grid_left[mask]  # (h, w)
                else:
                    new_disp = image_grid_left
                    mask = self.mask_bidir(new_disp)
                    new_disp[mask] = image_grid_right[mask]  # (h, w)
                visual_control(disp[0], disp[1], new_disp)
            elif self.save_disp or not (self.data_type == '2ir'):
                # Case where the 'other' modality is NOT between the stereo pair and the closest disparity image
                # is used to project the new disparity
                sign = -abs(self.dist_other_closer) / self.dist_other_closer
                factor = abs(self.dist_other_closer / self.dist_left_right)
                # If sign > 0 : positive value of disp ==>projection image left 2 right (camera right 2 left)
                disp_temp = disp[0] * sign * factor  # (h, w)
                new_disp = disp[0] * factor  # (h, w)
                new_disp = self._grid_sample(new_disp, disp_temp, padding_mode='border').squeeze()  # (h, w)
            else:
                new_disp = None
            if self.data_type == '2ir':
                # Case where the disparity map doesn't need to be projected
                disp_reg = -self.dist_other_closer / self.dist_left_right * disp[0]
                image = sample["right"] if self.setup["proj_right"] else sample["left"]
            else:
                sign = abs(self.dist_other_closer) / self.dist_other_closer
                disp_reg = new_disp * sign
                image = sample["other"]  # (1, c, h, w)
            image_reg = self._grid_sample(image, disp_reg, padding_mode='border').squeeze()  # (c, h, w)
            # if len(image_reg.shape) == 3 and self.method == "pytorch":
            #     image_reg = image_reg.permute(1, 2, 0)
        return image_reg, new_disp

    @timeit
    def _algo_reconstruction(self, disp, sample, verbose=False):
        """
        Reconstruction function from disparity image using numpy and python basic tools.
        Verbose can be set to True to have a visual control over the function process
        :return: reconstruct image & projected disparity
        """
        # Case where the "other" camera is at the same position as one of the stereo camera
        if self.dist_other_closer == 0:
            new_disp = disp
            if self.data_type == '2ir':
                image_reg = sample['right'] if self.setup["proj_right"] else sample['left']
            else:
                image_reg = sample["other"]
        else:
            disp = disp.cpu().numpy()
            if self.setup["pred_bidir_disp"]:
                ## Case where the "other" is between left and right
                if 0 < self.dist_left_other < self.dist_left_right:
                    disp_left = -disp[0]
                    disp_right = disp[1]  # (h, w)
                ## Case where the "other" is at the left of the left stereo camera
                elif self.dist_left_other < 0:
                    disp_left = disp[0]
                    disp_right = disp[1]  # (h, w)
                ## Case where the "other" is at the right of the right stereo camera
                else:
                    disp_left = -disp[0]
                    disp_right = -disp[1]  # (h, w)
                if self.setup["proj_right"]:
                    translation_ratio_right = abs(self.dist_other_closer / self.dist_left_right)
                    translation_ratio_left = abs(self.dist_left_right + self.dist_other_closer) / self.dist_left_right
                    disp_image_left = abs(disp_left * translation_ratio_right)
                    disp_image_right = abs(disp_right * translation_ratio_right)

                else:
                    translation_ratio_right = abs(self.dist_left_right - self.dist_other_closer) / self.dist_left_right
                    translation_ratio_left = abs(self.dist_other_closer / self.dist_left_right)
                    disp_image_left = abs(disp_left * translation_ratio_left)
                    disp_image_right = abs(disp_right * translation_ratio_left)
                disp_right = disp_right * translation_ratio_right
                disp_left = disp_left * translation_ratio_left
                # Projection of the disparity maps on the "other" modality
                new_disp_left = reconstruction_from_disparity(disp_image_left, disp_left, verbose=False, median=True)
                new_disp_right = reconstruction_from_disparity(disp_image_right, disp_right, verbose=False, median=True)
                if self.setup["proj_right"]:
                    new_disp = new_disp_right
                    mask = self.mask_bidir(new_disp)
                    new_disp[mask] = new_disp_left[mask]
                else:
                    new_disp = new_disp_left
                    mask = self.mask_bidir(new_disp)
                    new_disp[mask] = new_disp_right[mask]
            elif self.save_disp or not (self.data_type == '2ir'):
                # Case where the 'other' modality is NOT between the stereo pair and the closest disparity image
                # is used to project the new disparity
                sign = -self.dist_other_closer / abs(self.dist_other_closer)
                translation_ratio = abs(self.dist_other_closer / self.dist_left_right)
                disp_reg = sign * disp[0] * translation_ratio  # (h, w)
                disp_image = disp[0] * translation_ratio  # (h, w)
                new_disp = reconstruction_from_disparity(disp_image, disp_reg, verbose=False, median=True)  # (h, w)
            else:
                new_disp = None
            new_disp = disparity_post_process(new_disp, 0, 0, 200, verbose=False)
            if self.data_type == '2ir':
                # Projection of the infrared image, the closest to the RGB image
                translation_ratio = self.dist_other_closer / abs(self.dist_left_right)
                disp_reg = -disp[0] * translation_ratio
                image = sample["right"].squeeze() if self.setup["proj_right"] else sample["left"].squeeze()
                image_reg = reconstruction_from_disparity(image, disp_reg, verbose=verbose, inpainting=self.inpainting)
            else:
                # Projection of the "other" image, to the closest image
                image = sample["other"].squeeze()
                sign = abs(self.dist_other_closer) / self.dist_other_closer
                disp_reg = sign * new_disp
                image_reg = reconstruction_from_disparity(image, disp_reg, verbose=verbose, inpainting=self.inpainting)
        return image_reg, new_disp

    def mask_bidir(self, image):
        if self.method == "pytorch":
            kernel = torch.ones(3, 3).to(self.device)
            mask = image <= 0.
            mask = dilation(mask.unsqueeze(0).unsqueeze(0), kernel).squeeze() > 0
        else:
            mask = np.uint8((image <= 0.) * 255)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            mask = cv.dilate(mask, kernel, iterations=1) > 0
        return mask


BORDER_FLAG = {"BORDER_CONSTANT": 0,
               "BORDER_REPLICATE": 1,
               "BORDER_REFLECT": 2,
               "BORDER_WRAP": 3,
               "BORDER_REFLECT_101": 4,
               "BORDER_TRANSPARENT": 5,
               "BORDER_ISOLATED": 16}

INTERPOLATION_FLAG = {"INTER_NEAREST": 0,
                      "INTER_LINEAR": 1,
                      "INTER_CUBIC": 2,
                      "INTER_AREA": 3,
                      "INTER_LANCZOS4": 4,
                      "INTER_LINEAR_EXACT": 5,
                      "INTER_NEAREST_EXACT": 6,
                      "INTER_MAX": 7,
                      "WARP_FILL_OUTLIERS": 8,
                      "WARP_INVERSE_MAP": 16}
