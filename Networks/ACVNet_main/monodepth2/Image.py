import cv2 as cv
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import cm
from os.path import *
from pathlib import Path
from scipy.ndimage import median_filter
import tifffile


class ImageCustom(np.ndarray):
    ##
    # A class defining the general basic framework of an image

    def __new__(cls, inp, *args, name='new_image'):
        # Input array is a path to an image OR an already formed ndarray instance
        if isinstance(inp, str):
            name = basename(inp)
            inp = io.imread(inp)
            if len(inp.shape) > 2:
                if np.sum(inp[:, :, 0] - inp[:, :, 1]) == 0:
                    inp = inp[:, :, 0]
        image = np.uint8(np.asarray(inp).view(cls))
        # image.current_value = np.uint8(np.asarray(inp))
        image.name = name
        # add the new attributes to the created instance of Image
        if len(image.shape) == 2:
            image.info = 'IR'
            image.cmap = 'GRAYSCALE'
        elif len(image.shape) == 3:
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            image.info = 'Visible'
            image.cmap = 'RGB'
        # image.origin = inp
        if image.name[-4] == '.':
            image.name = image.name[:-4]
        elif image.name[-5] == '.':
            image.name = image.name[:-5]
        if len(args) > 0:
            if isinstance(args[0], ImageCustom):
                image.pass_attr(args[0])
                # image.current_value = np.uint8(np.asarray(inp))
            elif isinstance(args[0], dict):
                image.__dict__.update(args[0])
                # image.current_value = np.uint8(np.asarray(inp))
        # Finally, we must return the newly created object:
        # image.lab = image.LAB()
        return image

    def __str__(self):
        ##
        # Redefine the way of printing
        if len(self.shape) == 0:
            return str(self.view(np.ndarray))
        else:
            return f"Resolution : {self.shape[1]}x{self.shape[0]}px\n" \
                   f"Type of image : {self.info}\n" \
                   f"Current Color Map : {self.cmap}\n"

    def __array_finalize__(self, image):
        # see InfoArray.__array_finalize__ for comments
        if image is None:
            return
        self.info = getattr(image, 'info', None)
        self.cmap = getattr(image, 'cmap', None)
        self.origin = getattr(image, 'origin', None)
        self.name = getattr(image, 'name', None)
        self.current_value = getattr(image, 'current_value', None)

    def pass_attr(self, image):
        self.__dict__ = image.__dict__.copy()

    def show(self, num=None, figsize=(20, 20), dpi=40, facecolor=None, edgecolor=None, frameon=True, clear=False,
             cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None,
             extent=None, *, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None,
             data=None, **kwargs):
        plt.figure(num=num, figsize=figsize, dpi=dpi, facecolor=facecolor,
                   edgecolor=edgecolor, frameon=frameon, clear=clear)
        if self.cmap == "IR" or "EDGES":
            cmap = 'gray'
            plt.imshow(self, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent,
                       interpolation_stage=interpolation_stage, filternorm=filternorm, filterrad=filterrad,
                       resample=resample, url=url, data=data, **kwargs)
            plt.xticks([]), plt.yticks([])

        else:
            plt.imshow(self, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent,
                       interpolation_stage=interpolation_stage, filternorm=filternorm, filterrad=filterrad,
                       resample=resample, url=url, data=data, **kwargs)
            plt.xticks([]), plt.yticks([])
        plt.show()

    def GRAYSCALE(self, true_value=True):
        if (self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES') and len(self.shape) == 2:
            return self.copy()
        else:
            if self.cmap == 'GRAYSCALE':
                true_value = False
            if true_value:
                if self.cmap == 'RGB':
                    i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2GRAY), self)
                    i.cmap = 'GRAYSCALE'
                    return i
                elif self.cmap == 'BGR':
                    i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2GRAY), self)
                    i.cmap = 'GRAYSCALE'
                    return i
                elif self.cmap == 'HSV':
                    i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HSV2RGB), self)
                elif self.cmap == "LAB":
                    i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LAB2RGB), self)
                i.cmap = 'RGB'
                return i.GRAYSCALE()
            else:
                i = np.mean(self.copy(), -1)
                i = ImageCustom(i, self)
                i.cmap = 'GRAYSCALE'
                return i

    def HSV(self):
        hsv = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            hsv[:, :, 0] = np.zeros_like(self)
            hsv[:, :, 1] = np.zeros_like(self)
            hsv[:, :, 2] = self.copy()
            i = ImageCustom(hsv, self)
        elif self.cmap == 'HSV':
            return self
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2HSV), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2HSV), self)
        elif self.cmap == "LAB":
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LAB2RGB), self)
            i.cmap = 'RGB'
            return i.HSV()
        i.cmap = 'HSV'
        return i

    def BGR(self):
        if self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2BGR), self)
        elif self.cmap == 'LAB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LAB2BGR), self)
        elif self.cmap == 'HSV':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HSV2BGR), self)
        elif self.cmap == 'BGR':
            return self.copy()
        else:
            i = ImageCustom(cv.cvtColor(self.RGB(), cv.COLOR_RGB2BGR), self)
        i.cmap = 'BGR'
        return i

    def RGB(self, colormap='inferno'):
        if self.cmap == 'RGB':
            return self.copy()
        elif self.cmap == 'HSV':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HSV2RGB), self)
        elif self.cmap == 'LAB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_LAB2RGB), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2RGB), self)
        else:
            rgb = np.empty([self.shape[0], self.shape[1], 3])
            if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
                x = np.linspace(0.0, 1.0, 256)
                cmap_rgb = cm.get_cmap(plt.get_cmap(colormap))(x)[np.newaxis, :, :3]
                rgb[:, :, 0] = cmap_rgb[0, self[:, :], 0]
                rgb[:, :, 1] = cmap_rgb[0, self[:, :], 1]
                rgb[:, :, 2] = cmap_rgb[0, self[:, :], 2]
                rgb *= 255
                i = ImageCustom(rgb, self)
        i.cmap = 'RGB'
        return i

    def LAB(self):
        lab = np.empty([self.shape[0], self.shape[1], 3])
        if self.cmap == 'GRAYSCALE' or self.cmap == 'EDGES':
            lab[:, :, 0] = self.copy()
            lab[:, :, 1] = np.zeros_like(self)
            lab[:, :, 2] = np.zeros_like(self)
            i = ImageCustom(lab, self)
        elif self.cmap == 'LAB':
            return self.copy()
        elif self.cmap == 'RGB':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_RGB2LAB), self)
        elif self.cmap == 'BGR':
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_BGR2LAB), self)
        elif self.cmap == "HSV":
            i = ImageCustom(cv.cvtColor(self.copy(), cv.COLOR_HSV2RGB), self)
            i.cmap = 'RGB'
            return i.LAB()
        i.cmap = 'LAB'
        return i

    # def reset_to_original_image(self):
    #     image = ImageCustom(self.origin, self)
    #     if len(image.shape) > 2:
    #         image.cmap = 'RGB'
    #     else:
    #         image.cmap = 'GRAYSCALE'
    #     return image
    #
    # def reset_to_current_value(self):
    #     image = ImageCustom(self.current_value, self)
    #     if len(image.shape) > 2:
    #         image.cmap = 'RGB'
    #     else:
    #         image.cmap = 'GRAYSCALE'
    #     return image

    def save(self, path=None):
        if not path:
            path = join(Path(dirname(__file__)).parent.absolute(), 'output')
        path = join(path, self.name + ".jpg")
        io.imsave(path, self, plugin=None, check_contrast=True)

    def median_filter(self, size=3):
        im = ImageCustom(median_filter(self, size), self)
        im.current_value = np.asarray(self)
        return im
