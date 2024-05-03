import time

import cv2 as cv
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import cm
from os.path import *
from pathlib import Path
from scipy.ndimage import median_filter


class Mask(np.ndarray):
    ##
    # A class defining the general basic framework of a mask

    def __new__(cls, RGB, *args):
        # Input array is a path to an image OR an already formed ndarray instance
        assert isinstance(RGB, (np.ndarray, ImageCustom)), \
            "A Mask is defined from an array or an ImageCustom, First Input got the wrong type"
        mask_generator = np.zeros_like(RGB[:, :, 0]).view(cls)
        if RGB.dtype == np.float64:
            mask_generator.RGB = ImageCustom(np.uint8(RGB * 255))
        else:
            mask_generator.RGB = ImageCustom(RGB)
        if len(args) > 0:
            IR = args[0]
            assert isinstance(IR, (np.ndarray, ImageCustom)), \
                "A Mask is defined from an array or an ImageCustom, Second Input got the wrong type"
            if IR.dtype == np.float64:
                mask_generator.IR = ImageCustom(np.uint8(IR * 255))
            else:
                mask_generator.IR = ImageCustom(IR)
            if len(mask_generator.IR.shape) == 2:
                mask_generator.DIFF = mask_generator.RGB.GRAYSCALE().diff(mask_generator.IR)
            else:
                mask_generator.DIFF = mask_generator.RGB.diff(mask_generator.IR)
        else:
            mask_generator.IR = None
            mask_generator.DIFF = None

        return mask_generator

    def __add__(self, other):
        res = np.zeros_like(self)
        res[self > 0] = other[self > 0]
        # res[other >= self] = other[other >= self] - self[other >= self]
        return res

    def __str__(self):
        ##
        # Redefine the way of printing
        if len(self.shape) == 0:
            return str(self.view(np.ndarray))
        else:
            return f"Resolution : {self.shape[1]}x{self.shape[0]}px\n"

    def __array_finalize__(self, mask):
        # see InfoArray.__array_finalize__ for comments
        if mask is None:
            return
        self.RGB = getattr(mask, 'RGB', None)
        self.IR = getattr(mask, 'Vis', None)
        self.DIFF = getattr(mask, 'DIFF', None)

    def pass_attr(self, image):
        self.__dict__ = image.__dict__.copy()

    def low(self, low_threshold=-1, diff=True, gaussian=3):
        '''
        :param low_threshold: If -1, means adaptive threshold
        :param diff: use the differential image with the Vis source if possible
        :param gaussian: size of the gaussian kernel for blurring
        :return: Mask for the low values of luminance over the RGB image
        '''
        mask = np.zeros_like(self.RGB.GRAYSCALE())
        print(mask.shape)
        if low_threshold >= 0 and low_threshold <= 255:
            if not self.IR is None and diff:
                mask[self.RGB.GRAYSCALE() <= low_threshold] = self.DIFF[self.RGB.GRAYSCALE() <= low_threshold]
            else:
                mask[self.RGB <= low_threshold] = 255
            mask = cv.GaussianBlur(mask, (gaussian, gaussian), 0)
        else:
            gray = self.RGB.LAB()[:, :, 0]
            blurred = cv.GaussianBlur(gray, (gaussian, gaussian), 0)
            mask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 10)
        mask = median_filter(mask, 3)
        return mask / mask.max()

    def high(self, high_threshold=-1, diff=True, gaussian=3, dilation_size=2):
        '''
        :param high_threshold: If -1, means adaptive threshold
        :param diff: use the differential image with the Vis source if possible
        :param gaussian: size of the gaussian kernel for blurring
        :param dilation_size: Half-size of the dilation kernel
        :return: Mask for the low values of luminance over the RGB image
        '''
        mask = np.zeros_like(self.RGB.GRAYSCALE())
        if high_threshold >= 0 and high_threshold <= 255:
            if not self.IR is None and diff:
                mask[self.RGB.GRAYSCALE() >= high_threshold] = (self.IR[self.RGB.GRAYSCALE() >= high_threshold] / 255 *
                                                                self.DIFF[
                                                                    self.RGB.GRAYSCALE() >= high_threshold] / 255) * 255
            else:
                mask[self.RGB >= high_threshold] = 255
            mask = cv.GaussianBlur(mask, (gaussian, gaussian), 0)
        else:
            gray = self.RGB.LAB()[:, :, 0]
            blurred = cv.GaussianBlur(gray, (gaussian, gaussian), 0)
            mask = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 15)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1))
        mask = cv.dilate(mask, kernel) / 255  # * self.DIFF/255
        return mask / mask.max()

    def diff(self, threshold=20, gaussian=3):
        assert self.DIFF is not None, "This mask generator doesn't have an Vis image"
        mask = np.zeros_like(self.RGB.GRAYSCALE())
        mask[self.DIFF > threshold] = 255 - self.DIFF[self.DIFF > threshold]
        mask[self.DIFF <= threshold] = 255
        mask = cv.GaussianBlur(mask, (gaussian, gaussian), 0)
        return mask / mask.max()

    @function_timer(False)
    def ssim(self, weightRGB=1.5, win_size=3, RGB=None, IR=None):
        assert self.IR is not None, "This mask generator doesn't have an Vis image"
        if RGB is None or IR is None:
            RGB = self.RGB
            IR = self.IR
        win_size = win_size * 2 + 1
        mask_diff = Image_ssim(RGB, IR, win_size=win_size)
        im = mask_diff.image.mean_shift() / 255

        mask_rgb = Image_ssim(RGB, win_size=3).image/255 * RGB.GRAYSCALE()/255
        # mask_rgb = (mask_rgb
        mask_ir = Image_ssim(IR, win_size=3).image/255 * IR/255
        # mask_ir = (mask_ir).mean_shift()
        # temp = mask_rgb-mask_ir
        # temp = (temp - temp.min()) / (temp.max() - temp.min())
        # cv.imshow('mask diff', im)
        cv.imshow('mask rgb', mask_rgb)
        cv.imshow('mask ir', mask_ir)
        # cv.imshow('diff', temp)
        # cv.imshow('mask ir', mask_ir)
        cv.waitKey(0)
        cv.destroyAllWindows()

        mask_rgb = (Image_ssim(RGB, win_size=win_size).image.mean_shift(1 - 1 / (weightRGB + 1)) / 255 - \
                    Image_ssim(IR, win_size=win_size).image.mean_shift(1 / (weightRGB + 1)) / 255) * im
        mask_ssim = ImageCustom((mask_rgb - mask_rgb.min()) / (mask_rgb.max() - mask_rgb.min())) \
            .mean_shift(weightRGB / (weightRGB + 1)).gaussian_filter(3)
        return mask_ssim

    def saliency_ssim(self, verbose=False, level_max=1, weightRGB=1):
        assert self.IR is not None, "This mask generator doesn't have an Vis image"
        scale = [0, 1, 2, 3]
        scale_pyr_RGB = self.RGB.pyr_scale(octave=max(scale), verbose=False)
        scale_pyr_IR = self.IR.pyr_scale(octave=max(scale), verbose=False)
        pyr_ssim = {}
        for c in scale:
            pyr_ssim[c] = {}
            for i in range(level_max):
                win_size = i + 1
                pyr_ssim[c][i] = self.ssim(weightRGB=weightRGB, RGB=scale_pyr_RGB[c],
                                           IR=scale_pyr_IR[c], win_size=win_size)
                if verbose:
                    cv.imshow(f'SSIM mask lv {i}, scale {c}', pyr_ssim[c][i])
                    cv.waitKey(0)
                    cv.destroyAllWindows()

        ssim_mask = scaled_fusion(pyr_ssim, mean, mean, first_level_scaled=False).gaussian_filter(3)
        if verbose:
            cv.imshow(f'SSIM mask final', ssim_mask)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return ImageCustom(ssim_mask, scale_pyr_RGB[0]).unpad()

    def saliency_gaussian(self, verbose=False, weightRGB=1.5, intensityRBG=1, intensityIR=1, colorRGB=1, edgeRGB=3,
                          edgeIR=1):
        assert self.IR is not None, "This mask generator doesn't have an Vis image"
        scale = [1, 2]
        delta = [3, 4]
        sigma0 = 3
        gaussian_pyr_RGB = self.RGB.pyr_gauss(octave=1, interval=max(delta) + max(scale), sigma0=sigma0,
                                              verbose=False)
        gaussian_pyr_IR = self.IR.pyr_gauss(octave=1, interval=max(delta) + max(scale), sigma0=sigma0)
        ## Saliency of Intensity
        maps_Int_RGB = {}
        maps_Int_IR = {}
        if verbose:
            visu1 = np.hstack([gaussian_pyr_RGB[0].GRAYSCALE(), gaussian_pyr_IR[0]])
        for c in scale:
            maps_Int_RGB[c] = {}
            maps_Int_IR[c] = {}
            for d in delta:
                maps_Int_RGB[c][d] = gaussian_pyr_RGB[1][c].GRAYSCALE().diff(gaussian_pyr_RGB[1][c + d].GRAYSCALE())
                maps_Int_IR[c][d] = gaussian_pyr_IR[1][c].diff(gaussian_pyr_IR[1][c + d])
                maps_Int_RGB[c][d] = normalization_maps(maps_Int_RGB[c][d])
                maps_Int_IR[c][d] = normalization_maps(maps_Int_IR[c][d])
                if verbose:
                    temp = np.hstack([maps_Int_RGB[c][d], maps_Int_IR[c][d]])
                    temp = np.hstack([temp, np.uint8(np.zeros([temp.shape[0], visu1.shape[1] - temp.shape[1]]))])
                    visu1 = np.vstack([visu1, temp])

        ## Saliency of Color
        RGBY = {}
        if verbose:
            visu2 = gaussian_pyr_RGB[0].BGR()
        for c in scale:
            RGBY[c] = {}
            for d in delta:
                blur = gaussian_pyr_RGB[1][c + d] / 255
                ref = gaussian_pyr_RGB[1][c] / 255
                Rc = ref[:, :, 0] - (ref[:, :, 1] + ref[:, :, 2]) / 2
                Rs = blur[:, :, 0] - (blur[:, :, 1] + blur[:, :, 2]) / 2
                Gc = ref[:, :, 1] - (ref[:, :, 0] + ref[:, :, 2]) / 2
                Bc = ref[:, :, 2] - (ref[:, :, 0] + ref[:, :, 1]) / 2
                Yc = (ref[:, :, 0] + ref[:, :, 1]) / 2 - abs(ref[:, :, 0] - (ref[:, :, 1])) / 2 - ref[:, :, 2]
                RG = abs((Rc - Gc) - (Gc - Rs))
                BY = abs((Bc - Yc) - (Bc - Rs))
                RGBY[c][d] = (RG / 2).add(BY / 2)
                RGBY[c][d] = ImageCustom(RGBY[c][d] * 255).gaussian_filter(3)
                if verbose:
                    temp = cv.pyrDown(np.hstack([ImageCustom(RGBY[c][d] * 255).RGB(colormap='gray'),
                                                 ImageCustom(RGBY[c][d] * 255).RGB(colormap='gray')]))
                    temp = np.hstack([temp, np.uint8(np.zeros([temp.shape[0], visu2.shape[1] - temp.shape[1], 3]))])
                    visu2 = np.vstack([visu2, temp])

        ## Saliency of Orientation
        O_RGB = {}
        O_IR = {}
        theta = [0, 45, 90, 135]
        filters = create_gaborfilter(num_filters=4, ksize=11, sigma=1.0)
        if verbose and self.IR is not None:
            visu3 = np.hstack([gaussian_pyr_RGB[0].GRAYSCALE(), gaussian_pyr_IR[0]])
        for c in scale:
            O_RGB[c] = {}
            Oc_RGB = gaussian_pyr_RGB[1][c].GRAYSCALE()
            O_IR[c] = {}
            Oc_IR = gaussian_pyr_IR[1][c]
            for d in delta:
                Os_RGB = gaussian_pyr_RGB[1][c + d].GRAYSCALE()
                O_RGB[c][d] = np.zeros_like(Os_RGB)
                Os_IR = gaussian_pyr_IR[1][c + d]
                O_IR[c][d] = np.zeros_like(Os_IR)
                for idx, kernel in enumerate(filters):
                    O_RGB_c = cv.filter2D(Oc_RGB / 255, -1, kernel)
                    O_RGB_s = cv.filter2D(Os_RGB / 255, -1, kernel)
                    O_RGB[c][d] = np.maximum(abs(O_RGB_c - O_RGB_s), O_RGB[c][d])
                    O_IR_c = abs(cv.filter2D(Oc_IR / 255, -1, kernel))
                    O_IR_s = abs(cv.filter2D(Os_IR / 255, -1, kernel))
                    O_IR[c][d] = np.maximum(abs(O_IR_c - O_IR_s), O_IR[c][d])
                    # if verbose:
                    #     cv.imshow(f'Orientation RGB', normalization_maps(O_RGB[c][d]))
                    #     cv.imshow(f'Orientation Vis', normalization_maps(O_IR[c][d]))
                    #     cv.waitKey(0)
                if verbose:
                    temp = normalization_maps(np.hstack([O_RGB[c][d], O_IR[c][d]])).gaussian_filter(3)
                    temp = np.hstack([temp, np.uint8(np.zeros([temp.shape[0], visu3.shape[1] - temp.shape[1]]))])
                    visu3 = np.vstack([visu3, temp])
                # O_RGB[c][d] = O_RGB[c][d]
                # O_IR[c][d] = normalization_maps(O_IR[c][d].gaussian_filter(3)

        # Intensity_maps_RGB, Intensity_maps_IR = normalization_maps(
        #     scaled_fusion(maps_Int_RGB, np.maximum, np.maximum, first_level_scaled=True),
        #     scaled_fusion(maps_Int_IR, np.maximum, np.maximum, first_level_scaled=True))
        Intensity_maps_RGB, Intensity_maps_IR = normalization_maps(
            scaled_fusion(maps_Int_RGB, np.maximum, np.maximum, first_level_scaled=True).gaussian_filter(3),
            scaled_fusion(maps_Int_IR, np.maximum, np.maximum, first_level_scaled=True).gaussian_filter(3))
        if verbose:
            cv.imshow('Intensity map Vis', Intensity_maps_IR)
            cv.imshow('Intensity map RGB', Intensity_maps_RGB)
            cv.waitKey(0)
            # cv.destroyAllWindows()

        Color_maps_RGB = normalization_maps(scaled_fusion(RGBY, np.maximum, np.maximum, first_level_scaled=True))
        if verbose:
            cv.imshow('Color Saliency RGB', Color_maps_RGB)
            cv.waitKey(0)
            # cv.destroyAllWindows()

        Orientation_maps_RGB, Orientation_maps_IR = normalization_maps(
            scaled_fusion(O_RGB, np.maximum, np.maximum, first_level_scaled=True),
            scaled_fusion(O_IR, np.maximum, np.maximum, first_level_scaled=True))
        if verbose:
            cv.imshow('Orientation map Vis', Orientation_maps_IR)
            cv.imshow('Orientation map RGB', Orientation_maps_RGB)
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.imshow('Saliency of Intensity', cv.pyrDown(visu1))
            cv.imshow('Saliency of Color', visu2)
            cv.imshow('Saliency of Orientation', cv.pyrDown(visu3))
            cv.waitKey(0)
            cv.destroyAllWindows()
        totRBG = intensityRBG + colorRGB + edgeRGB
        totIR = edgeIR + intensityIR
        intensityRBG, colorRGB, edgeRBG = intensityRBG / totRBG, colorRGB / totRBG, edgeRGB / totRBG
        edgeIR, intensityIR = edgeIR / totIR, intensityIR / totIR
        rgb_mask = ImageCustom((Orientation_maps_RGB * edgeRBG
                                + Color_maps_RGB * colorRGB
                                + Intensity_maps_RGB * intensityRBG) * 15 / 16 + 16, gaussian_pyr_RGB[0]).mean_shift(
            1 - 1 / (weightRGB + 1))
        ir_mask = ImageCustom((Orientation_maps_IR * edgeIR
                               + Intensity_maps_IR * intensityIR) * 15 / 16 + 16, gaussian_pyr_IR[0]).mean_shift(
            1 / (weightRGB + 1))
        tot = rgb_mask / 1.0 + ir_mask / 1.0
        diff = rgb_mask.diff(ir_mask)
        if verbose:
            cv.imshow('rgb mask', rgb_mask)
            cv.imshow('Vis mask', ir_mask)
            cv.waitKey(0)
        # print(f'tot max/min : {tot.max()}, {tot.min()}, rgb max/min : {rgb_mask.max()}, {rgb_mask.min()}, '
        #       f'ir max/min : {ir_mask.max()}, {ir_mask.min()}, diff max/min : {diff.max()}, {diff.min()}')

        res = cv.pyrDown(cv.pyrUp(rgb_mask / tot))
        return ImageCustom(res, rgb_mask).unpad()

    def saliency_scale(self, verbose=True, weightRGB=1, intensityRBG=1, intensityIR=1, colorRGB=1, edgeRGB=1, edgeIR=1):
        assert self.IR is not None, "This mask generator doesn't have an Vis image"
        scale = [0, 1, 2]
        delta = [3, 4]
        gaussian_pyr_RGB = self.RGB.pyr_scale(octave=max(scale) + max(delta), verbose=False)
        gaussian_pyr_IR = self.IR.pyr_scale(octave=max(scale) + max(delta), verbose=False)
        ## Saliency of Intensity
        maps_Int_RGB = {}
        maps_Int_IR = {}
        if verbose:
            visu1 = np.hstack([gaussian_pyr_RGB[0].GRAYSCALE(), gaussian_pyr_IR[0]])
        for c in scale:
            maps_Int_RGB[c] = {}
            maps_Int_IR[c] = {}
            for d in delta:
                temp = gaussian_pyr_RGB[c + d].GRAYSCALE()
                temp2 = gaussian_pyr_IR[c + d]
                for i in range(d):
                    temp = cv.pyrUp(temp)
                    temp2 = cv.pyrUp(temp2)
                maps_Int_RGB[c][d] = gaussian_pyr_RGB[c].GRAYSCALE().diff(temp)
                maps_Int_IR[c][d] = gaussian_pyr_IR[c].diff(temp2)
                # for i in range(c):
                #     maps_Int_RGB[k] = cv.pyrUp(maps_Int_RGB[k])
                # maps_Int_RGB[c][d] = normalization_maps(maps_Int_RGB[c][d])
                # for i in range(c):
                #     maps_Int_IR[k] = cv.pyrUp(maps_Int_IR[k])
                # maps_Int_IR[c][d] = normalization_maps(maps_Int_IR[c][d])
                if verbose:
                    temp = np.hstack([maps_Int_RGB[c][d], maps_Int_IR[c][d]])
                    temp = np.hstack([temp, np.uint8(np.zeros([temp.shape[0], visu1.shape[1] - temp.shape[1]]))])
                    visu1 = np.vstack([visu1, temp])

        Intensity_maps_RGB, Intensity_maps_IR = normalization_maps(
            scaled_fusion(maps_Int_RGB, np.maximum, np.maximum).gaussian_filter(3),
            scaled_fusion(maps_Int_IR, np.maximum, np.maximum).gaussian_filter(3))
        if verbose:
            cv.imshow('Intensity map Vis', Intensity_maps_IR)
            cv.imshow('Intensity map RGB', Intensity_maps_RGB)
            cv.waitKey(0)
            cv.destroyAllWindows()

        ## Saliency of Color
        RGBY = {}
        if verbose:
            visu2 = gaussian_pyr_RGB[0].GRAYSCALE()
        for c in scale:
            RGBY[c] = {}
            for d in delta:
                temp = gaussian_pyr_RGB[c + d]
                for i in range(d):
                    temp = cv.pyrUp(temp)
                blur = temp / 255
                ref = gaussian_pyr_RGB[c] / 255
                Rc = ref[:, :, 0] - (ref[:, :, 1] + ref[:, :, 2]) / 2
                Rs = blur[:, :, 0] - (blur[:, :, 1] + blur[:, :, 2]) / 2
                Gc = ref[:, :, 1] - (ref[:, :, 0] + ref[:, :, 2]) / 2
                Gs = blur[:, :, 1] - (blur[:, :, 0] + blur[:, :, 2]) / 2
                Bc = ref[:, :, 2] - (ref[:, :, 0] + ref[:, :, 1]) / 2
                Bs = blur[:, :, 2] - (blur[:, :, 0] + blur[:, :, 1]) / 2
                Yc = (ref[:, :, 0] + ref[:, :, 1]) / 2 - abs(ref[:, :, 0] - (ref[:, :, 1])) / 2 - ref[:, :, 2]
                RG = abs((Rc - Gc) - (Gs - Rs))
                BY = abs((Bc - Yc) - (Bs - Rs))
                RGBY[c][d] = (RG / 2).add(BY / 2)
                if verbose:
                    temp = cv.pyrDown(np.hstack([RG[c][d], BY[c][d]]))
                    temp = np.hstack([temp, np.uint8(np.zeros([temp.shape[0], visu2.shape[1] - temp.shape[1]]))])
                    visu2 = np.vstack([visu2, temp])
        Color_maps_RGB = normalization_maps(scaled_fusion(RGBY, np.maximum, np.maximum).gaussian_filter(3))
        if verbose:
            cv.imshow('Color map RGB', Color_maps_RGB)
            cv.waitKey(0)
            cv.destroyAllWindows()

        ## Saliency of Orientation
        O_RGB = {}
        O_IR = {}
        theta = [0, 45, 90, 135]
        filters = create_gaborfilter(num_filters=4, ksize=11, sigma=1.0)
        if verbose and self.IR is not None:
            visu3 = np.hstack([gaussian_pyr_RGB[0].GRAYSCALE(), gaussian_pyr_IR[0]])
        for c in scale:
            O_RGB[c] = {}
            Oc_RGB = gaussian_pyr_RGB[c].GRAYSCALE()
            O_IR[c] = {}
            Oc_IR = gaussian_pyr_IR[c]
            for d in delta:
                Os_RGB = gaussian_pyr_RGB[c + d].GRAYSCALE()
                Os_IR = gaussian_pyr_IR[c + d]
                for i in range(d):
                    Os_RGB = cv.pyrUp(Os_RGB)
                    Os_IR = cv.pyrUp(Os_IR)
                O_RGB[c][d] = np.zeros_like(Os_RGB)
                O_IR[c][d] = np.zeros_like(Os_IR)
                for idx, kernel in enumerate(filters):
                    O_RGB_c = cv.filter2D(Oc_RGB / 255, -1, kernel)
                    O_RGB_s = cv.filter2D(Os_RGB / 255., -1, kernel)
                    O_RGB[c][d] = np.maximum(abs(O_RGB_c - O_RGB_s), O_RGB[c][d])
                    O_IR_c = abs(cv.filter2D(Oc_IR / 255, -1, kernel))
                    O_IR_s = abs(cv.filter2D(Os_IR / 255, -1, kernel))
                    O_IR[c][d] = np.maximum(abs(O_IR_c - O_IR_s), O_IR[c][d])
                    if verbose:
                        cv.imshow(f'Orientation RGB', O_RGB[c][d])
                        cv.imshow(f'Orientation Vis', O_IR[c][d])
                        cv.waitKey(0)
                if verbose:
                    temp = normalization_maps(np.hstack([O_RGB[c][d], O_IR[c][d]]))
                    temp = np.hstack([temp, np.uint8(np.zeros([temp.shape[0], visu3.shape[1] - temp.shape[1]]))])
                    visu3 = np.vstack([visu3, temp])

        Orientation_maps_RGB, Orientation_maps_IR = normalization_maps(scaled_fusion(O_RGB, np.maximum, np.maximum),
                                                                       scaled_fusion(O_IR, np.maximum, np.maximum))
        if verbose:
            cv.imshow('Orientation map Vis', Orientation_maps_IR)
            cv.imshow('Orientation map RGB', Orientation_maps_RGB)
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.imshow('Saliency of Intensity', cv.pyrDown(visu1))
            cv.imshow('Saliency of Color', visu2)
            cv.imshow('Saliency of Orientation', cv.pyrDown(visu3))
            cv.waitKey(0)
            cv.destroyAllWindows()
        totRBG = intensityRBG + colorRGB + edgeRGB
        totIR = edgeIR + intensityIR
        intensityRBG, colorRGB, edgeRBG = intensityRBG / totRBG, colorRGB / totRBG, edgeRGB / totRBG
        edgeIR, intensityIR = edgeIR / totIR, intensityIR / totIR
        rgb_mask = ImageCustom((Orientation_maps_RGB * edgeRBG
                                + Color_maps_RGB * colorRGB
                                + Intensity_maps_RGB * intensityRBG) * 7 / 8 + 32, gaussian_pyr_RGB[0])
        ir_mask = ImageCustom((Orientation_maps_IR * edgeIR
                               + Intensity_maps_IR * intensityIR) * 7 / 8 + 32, gaussian_pyr_IR[0])
        tot = rgb_mask / 1.0 + ir_mask / 1.0
        diff = rgb_mask.diff(ir_mask)
        if verbose:
            cv.imshow('rgb mask', rgb_mask)
            cv.imshow('Vis mask', ir_mask)
            cv.waitKey(0)
        # print(f'tot max/min : {tot.max()}, {tot.min()}, rgb max/min : {rgb_mask.max()}, {rgb_mask.min()}, '
        #       f'ir max/min : {ir_mask.max()}, {ir_mask.min()}, diff max/min : {diff.max()}, {diff.min()}')

        res = (rgb_mask / tot).mean_shift(1 - 1 / (weightRGB + 1))
        return ImageCustom(res, rgb_mask).unpad()

    def static_saliency(self, weightRGB=1, verbose=False):
        # initialize OpenCV's static saliency spectral residual detector and
        # compute the saliency map
        saliency = cv.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap_rgb) = saliency.computeSaliency(self.RGB)
        (success, saliencyMap_ir) = saliency.computeSaliency(self.IR)
        saliencyMap_rgb = ImageCustom(saliencyMap_rgb * 255)
        saliencyMap_ir = ImageCustom(saliencyMap_ir * 255)
        saliencyMap_rgb, saliencyMap_ir = normalization_maps(saliencyMap_rgb, saliencyMap_ir)
        saliencyMap_rgb[saliencyMap_rgb < 16] = 16
        saliencyMap_ir[saliencyMap_ir < 16] = 16
        saliency_map = ImageCustom((saliencyMap_rgb / (saliencyMap_rgb/255 + saliencyMap_ir/255)))/255
        res = saliency_map.mean_shift(1 - 1 / (weightRGB + 1)).gaussian_filter(3)
        if verbose:
            cv.imshow("Image", self.RGB)
            cv.imshow("Output", res)
            cv.imshow("Output ir", saliencyMap_ir)
            cv.imshow("Output rgb", saliencyMap_rgb)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return res
