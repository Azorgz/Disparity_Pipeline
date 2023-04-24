import kornia
from torch.nn.functional import grid_sample
import torch
import warnings
import numpy as np

from utils.disparity_tools import reprojection_disparity, reconstruction_from_disparity
from utils.misc import timeit

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
import cv2 as cv


class ReconstructionBlock:
    """
    This block implement the reconstruction methods following the reconstruction options
    """

    def __init__(self, config, setup):
        # If the DisparityPipe need to measure the execution time of each block, this parameter will be set to True.
        if config["timeit"]:
            self.timeit = []
        self.device = config["device"]["device"]
        self.always_proj_infrared = config["always_proj_infrared"]
        self.save_disp = config["save_disp"]
        self.setup = setup
        self.dist_left_right = self.setup["position_setup"][0]
        self.dist_left_other = self.setup["position_setup"][1]
        self.data_type = config['dataset']["type"]
        if self.setup["proj_right"]:
            self.dist_other_closer = self.dist_left_other - self.dist_left_right
        else:
            self.dist_other_closer = self.dist_left_other
        if config["reconstruction"]["method"] == "pytorch":
            self.method = "pytorch"
            self.function = self.torch_reconstruction
        elif config["reconstruction"]["method"] == "fullOpenCv":
            self.function = self.opencv_reconstruction
            self.method = "fullOpenCv"
            self.interpolation = config["reconstruction"]["interpolation"]
            self.border = config["reconstruction"]["border"]
            if not isinstance(self.interpolation, int) and isinstance(self.interpolation, str):
                self.interpolation = INTERPOLATION_FLAG[self.interpolation]
            if not isinstance(self.border, int) and isinstance(self.border, str):
                self.border = BORDER_FLAG[self.border]
        else:
            self.function = self.algo_reconstruction
            self.method = "algo"
            self.inpainting = config["reconstruction"]["inpainting"]
        self.grid_sampling = self.chosen_grid_sample

    def __call__(self, disp, sample, *args, **kwargs):
        if self.method != "pytorch":
            sample["other"] = sample["other"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        return self.function(disp, sample)

    # def improved_grid_sample(self, image, disp, padding_mode='zeros', align_corners=True):
    #     h, w = disp.shape
    #     # cv.imshow("originale", image.cpu().numpy())
    #     image = image.squeeze(0)  # (c, h, w)
    #     image_color = len(image.shape) == 3 and image.shape[0] == 3
    #     if self.step == 0:
    #         self.step = 10
    #     inv = False
    #     if disp.max() <= 0:
    #         inv = True
    #         disp = -disp
    #     step = disp.max() / self.step
    #     values = torch.arange(start=disp.min() + step, end=disp.max() + step, step=step, device=self.device)
    #     if inv:
    #         disp = -disp
    #         values = -values
    #     new_image = torch.zeros_like(image)
    #     while len(new_image.shape) < 4:
    #         new_image = new_image.unsqueeze(0)
    #     inf = 0
    #     grid_reg = kornia.utils.create_meshgrid(h, w, device=self.device).to(image.dtype)
    #     grid_reg[0, :, :, 0] += disp
    #     for v in values:
    #         # breakpoint()
    #         mask = abs(inf) < abs(disp)
    #         mask = mask * (abs(disp) <= abs(v))
    #         temp = image.clone().detach()
    #         if image_color:
    #             temp[0, :, :] = temp[0, :, :] * mask
    #             temp[1, :, :] = temp[1, :, :] * mask
    #             temp[2, :, :] = temp[2, :, :] * mask
    #             temp = temp.unsqueeze(0)
    #         else:
    #             temp = (temp * mask).unsqueeze(0).unsqueeze(0)
    #         inf = v
    #         temp = grid_sample(temp, grid_reg, padding_mode=padding_mode, align_corners=align_corners)
    #         new_image[temp > 0] = temp[temp > 0]
    #         # demo = new_image.squeeze(0)
    #         # if len(demo.shape) == 3:
    #         #     demo = demo.permute(1, 2, 0)
    #         # print(v, abs(disp).max(), abs(inf))
    #         # cv.imshow("test", demo.cpu().numpy())
    #         # cv.waitKey(0)
    #     return new_image

    def chosen_grid_sample(self, image, disp, padding_mode='zeros', align_corners=True):
        h, w = disp.shape
        if self.method == "pytorch":
            grid_reg = kornia.utils.create_meshgrid(h, w, device=self.device).to(image.dtype)
            grid_reg[0, :, :, 0] += disp
            while len(image.shape) < 4:
                image = image.unsqueeze(0)
            return grid_sample(image, grid_reg, padding_mode=padding_mode, align_corners=align_corners)
        else:
            x = np.linspace(0, w, w)
            y = np.linspace(0, h, h)
            mapX, mapY = np.meshgrid(x, y)
            mapX += disp
            mapX, mapY = cv.convertMaps(mapX.astype(np.float32), mapY.astype(np.float32), cv.CV_32FC1)
            return cv.remap(image, mapX, mapY, self.interpolation, None, self.border)

    @timeit
    def torch_reconstruction(self, disp, sample):
        _, h, w = disp.shape
        if self.dist_other_closer == 0:
            # Case where one of the Stereo Camera is in the same axis as the 'other' modality
            new_disp = torch.zeros(h, w)
            if self.always_proj_infrared and self.data_type == '2ir':
                image_reg = sample['right'] if self.setup["proj_right"] else sample['left']
            else:
                image_reg = sample["other"]
        else:
            if self.setup["pred_bidir_disp"]:
                # Case where the 'other' modality is between the stereo pair and both disparity images
                # are used to project the new disparity
                if self.setup["proj_right"]:
                    disp_left = disp[0] / disp[0].shape[1] * 2 * \
                                (self.dist_left_right + self.dist_other_closer) / self.dist_left_right  # (h, w)
                    disp_right = disp[1] / disp[1].shape[1] * 2 * \
                                 self.dist_other_closer / self.dist_left_right  # (h, w)
                else:
                    disp_left = disp[0] / disp[0].shape[1] * 2 * \
                                self.dist_other_closer / self.dist_left_right  # (h, w)
                    disp_right = -disp[1] / disp[1].shape[1] * 2 * \
                                 (self.dist_left_right - self.dist_other_closer) / self.dist_left_right  # (h, w)
                image_grid_left = disp[0] * abs(self.dist_other_closer / self.dist_left_right)  # (h, w)
                image_grid_right = disp[1] * abs(self.dist_other_closer / self.dist_left_right)  # (h, w)
                image_grid_left = self.grid_sampling(image_grid_left, disp_left,
                                                     padding_mode='zeros', align_corners=True).squeeze()  # (h, w)
                image_grid_right = self.grid_sampling(image_grid_right, disp_right,
                                                      padding_mode='zeros', align_corners=True).squeeze()  # (h, w)
                if self.setup["proj_right"]:
                    new_disp = image_grid_right
                    new_disp[new_disp == 0.] = image_grid_left[new_disp == 0.]  # (h, w)
                else:
                    new_disp = image_grid_left
                    new_disp[new_disp == 0.] = image_grid_right[new_disp == 0.]  # (h, w)
            elif self.save_disp or not(self.always_proj_infrared and self.data_type == '2ir'):
                # Case where the 'other' modality is NOT between the stereo pair and the closest disparity image
                # is used to project the new disparity
                disp_temp = disp[0] / disp[0].shape[1] * 2 * \
                            self.dist_other_closer / self.dist_left_right  # (h, w)
                new_disp = disp[0] * abs(self.dist_other_closer / self.dist_left_right)  # (h, w)
                new_disp = self.grid_sampling(new_disp, disp_temp, padding_mode='border',
                                              align_corners=True).squeeze()  # (h, w)
            else:
                new_disp = None
            if self.always_proj_infrared and self.data_type == '2ir':
                #### Case where the disparity map doesn't need to be projected
                sign = self.dist_other_closer / abs(self.dist_other_closer)
                disp_reg = sign * disp[0] / disp[0].shape[1] * 2
            elif (self.setup["pred_bidir_disp"] and self.setup["proj_right"]) or \
                    (not self.setup["pred_bidir_disp"] and not self.setup["proj_right"]):
                disp_reg = new_disp / new_disp.shape[1] * 2
            else:
                disp_reg = -new_disp / new_disp.shape[1] * 2
            other = sample["other"]  # (1, c, h, w)
            image_reg = self.grid_sampling(other, disp_reg, padding_mode='border',
                                           align_corners=True).squeeze()  # (c, h, w)
            if len(image_reg.shape) == 3:
                image_reg = image_reg.permute(1, 2, 0)

        return image_reg, new_disp

    @timeit
    def algo_reconstruction(self, disp, sample, verbose=False):
        """
        Reconstruction function from disparity image using numpy and python basic tools.
        Verbose can be set to True to have a visual control over the function process
        :return: reconstruct image & projected disparity
        """
        # Case where the "other" camera is at the same position as one of the stereo camera
        if self.dist_other_closer == 0:
            new_disp = disp
            if self.always_proj_infrared and self.data_type == '2ir':
                image_reg = sample['right'] if self.setup["proj_right"] else sample['left']
            else:
                image_reg = sample["other"]
        else:
            disp = disp.cpu().numpy()
            if self.setup["pred_bidir_disp"]:
                ## Case where the "other" is between left and right
                if 0 < self.dist_left_other < self.dist_left_right:
                    disp_left = -disp[0]
                    disp_right = disp[1] # (h, w)
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
                    disp_left = disp_left * (1 - translation_ratio_right) / (1 - translation_ratio_left)
                else:
                    translation_ratio_right = abs(self.dist_left_right - self.dist_other_closer) / self.dist_left_right
                    translation_ratio_left = abs(self.dist_other_closer / self.dist_left_right)
                    disp_right = disp_right * (1 - translation_ratio_left) / (1 - translation_ratio_right)
                new_disp_left = abs(reprojection_disparity(disp_left, translation_ratio_left, verbose=verbose))
                new_disp_right = abs(reprojection_disparity(disp_right, translation_ratio_right, verbose=verbose))
                # ## Case where the "other" is between left and right
                # if 0 < self.dist_left_other < self.dist_left_right:
                #     new_disp_left = -new_disp_left
                # ## Case where the "other" is at the left of the left stereo camera
                # elif self.dist_left_other < 0:
                #     pass
                # ## Case where the "other" is at the right of the right stereo camera
                # else:
                #     new_disp_left = -new_disp_left
                #     new_disp_right = -new_disp_right  # (h, w)
                if self.setup["proj_right"]:
                    new_disp = new_disp_right
                    new_disp_left[new_disp_left <= 5] = new_disp_right[new_disp_left <= 5]
                    new_disp[new_disp <= 5] = new_disp_left[new_disp <= 5]
                else:
                    new_disp = new_disp_left
                    new_disp_right[new_disp_right <= 5] = new_disp_left[new_disp_right <= 5]
                    new_disp[new_disp <= 5] = new_disp_right[new_disp <= 5]
            elif self.save_disp or not (self.always_proj_infrared and self.data_type == '2ir'):
                # Case where the 'other' modality is NOT between the stereo pair and the closest disparity image
                # is used to project the new disparity
                sign = -self.dist_other_closer / abs(self.dist_other_closer)
                disp = sign * disp[0]
                translation_ratio = abs(self.dist_other_closer / self.dist_left_right)
                new_disp = abs(reprojection_disparity(disp, translation_ratio, verbose=verbose))
            else:
                new_disp = None
            if self.always_proj_infrared and self.data_type == '2ir':
                translation_ratio = -self.dist_other_closer / self.dist_left_right
                disp_reg = disp * translation_ratio
                if self.setup["proj_right"]:
                    imageR = sample["other"].squeeze()
                    imageL = None
                    image_reg = reconstruction_from_disparity(imageL, imageR, -disp_reg,
                                                              verbose=verbose, inpainting=self.inpainting)
                else:
                    imageL = sample["other"].squeeze()
                    imageR = None
                    image_reg = reconstruction_from_disparity(imageL, imageR, disp_reg,
                                                              verbose=verbose, inpainting=self.inpainting)
            else:
                disp_reg = new_disp
                if self.dist_other_closer < 0:
                    imageL = sample["other"].squeeze()
                    imageR = None
                    image_reg = reconstruction_from_disparity(imageL, imageR, -new_disp,
                                                              verbose=verbose, inpainting=self.inpainting)
                elif self.dist_other_closer > 0:
                    imageR = sample["other"].squeeze()
                    imageL = None
                    image_reg = reconstruction_from_disparity(imageL, imageR, new_disp,
                                                              verbose=verbose, inpainting=self.inpainting)
                else:
                    image_reg = sample["other"]
        return image_reg, new_disp

    @timeit
    def opencv_reconstruction(self, disp, sample):
        """
        Reconstruction function from disparity image.
        :return: reconstruct image.
        """
        _, h, w = disp.shape
        disp = disp.cpu().numpy()
        if self.dist_other_closer == 0:
            # Case where one of the Stereo Camera is in the same axis as the 'other' modality
            new_disp = np.zeros([h, w])
            if self.always_proj_infrared and self.data_type == '2ir':
                image_reg = sample['right'] if self.setup["proj_right"] else sample['left']
            else:
                image_reg = sample["other"]
        else:
            if self.setup["pred_bidir_disp"]:
                # Case where the 'other' modality is between the stereo pair and both disparity images
                # are used to project the new disparity
                if self.setup["proj_right"]:
                    disp_left = disp[0]  # (h, w)
                    disp_right = disp[1] # (h, w)
                else:
                    disp_left = disp[0]  # (h, w)
                    disp_right = -disp[1]  # (h, w)
                image_grid_left = disp_left * abs(self.dist_other_closer / self.dist_left_right)  # (h, w)
                image_grid_right = disp_right * abs(self.dist_other_closer / self.dist_left_right)  # (h, w)
                image_grid_left = self.grid_sampling(image_grid_left, disp_left).squeeze()  # (h, w)
                image_grid_right = self.grid_sampling(image_grid_right, disp_right).squeeze()  # (h, w)
                if self.setup["proj_right"]:
                    new_disp = image_grid_right
                    new_disp[new_disp == 0.] = image_grid_left[new_disp == 0.]  # (h, w)
                else:
                    new_disp = image_grid_left
                    new_disp[new_disp == 0.] = -image_grid_right[new_disp == 0.]  # (h, w)
            elif self.save_disp or not (self.always_proj_infrared and self.data_type == '2ir'):
            # Case where the 'other' modality is NOT between the stereo pair and the closest disparity image
            # is used to project the new disparity
                disp_temp = disp[0]  # (h, w)
                new_disp = disp_temp * -self.dist_other_closer / self.dist_left_right  # (h, w)
                new_disp = abs(self.grid_sampling(new_disp, disp_temp).squeeze())  # (h, w)
            else:
                new_disp = None
            if self.always_proj_infrared and self.data_type == '2ir':
                #### Case where the disparity map doesn't need to be projected
                sign = self.dist_other_closer / abs(self.dist_other_closer)
                disp_reg = sign * disp[0]
            elif (self.setup["pred_bidir_disp"] and self.setup["proj_right"]) or \
                    (not self.setup["pred_bidir_disp"] and not self.setup["proj_right"]):
                disp_reg = new_disp
            else:
                disp_reg = -new_disp
            other = sample["other"]
            image_reg = self.grid_sampling(other, disp_reg)  # (h, w, c)
            return image_reg, new_disp


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
