"""
different ways to compute norm of the gradient of an image
prewitt, sobel, grad, roberts
"""

import numpy as np
# import textures as textures
from matplotlib import pyplot as plt
from skimage.filters import edges
from ..Image import ImageCustom
import cv2 as cv
from scipy.ndimage import prewitt, median_filter
import time
from .manipulation_tools import *
import pywt
import pywt.data


def edges_extraction(image, method="Canny", kernel_size=3, kernel_blur=5, low_threshold=5, ratio=3, level=2,
                     orientation=False):
    if image.cmap != "GRAYSCALE" and len(image.shape) == 3:
        i = image.GRAYSCALE()
    else:
        i = image.copy()
    orient = np.zeros_like(image)
    if kernel_blur > 1:
        image = cv.bilateralFilter(image, kernel_blur, kernel_blur * 2, kernel_blur / 2)
        # cv.GaussianBlur(image, (kernel_blur, kernel_blur), 0)
    if method == "prewitt" or method == "Prewitt":
        image = np.float_(image)
        # if kernel_blur > 1:
        #     image = cv.GaussianBlur(image, (kernel_blur, kernel_blur), 0)
        Ix = prewitt(image, axis=0)
        Iy = prewitt(image, axis=1)
        I = np.sqrt(Ix ** 2 + Iy ** 2)
        if orientation:
            orient = abs(abs(cv.phase(Ix, Iy, angleInDegrees=True) - 180) - 90)
    elif method == "laplacian" or method == "Laplacian":
        image = np.float_(image)
        I = abs(cv.Laplacian(image, cv.CV_32F, ksize=kernel_size * 2 + 1))

    elif method == "roberts" or method == 'Roberts':
        image = np.float_(image)
        Gx = np.array([[-1, 0], [0, 1]])
        Gy = np.array([[0, -1], [1, 0]])
        Ix = edges.convolve(image, Gx, mode='wrap')
        Iy = edges.convolve(image, Gy, mode='wrap')
        I = Ix ** 2 + Iy ** 2
        if orientation:
            abs(abs(cv.phase(Ix, Iy, angleInDegrees=True) - 180) - 90)
    elif method == "Sobel" or method == 'sobel':
        image = np.float_(image)
        Ix = cv.Sobel(image, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT_101)
        Iy = cv.Sobel(image, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT_101)
        I = np.uint8(np.sqrt(Ix ** 2 + Iy ** 2))
        if orientation:
            abs(abs(cv.phase(Ix, Iy, angleInDegrees=True) - 180) - 90)
    elif method == "Perso" or method == "perso":
        I = Harr_pyr(image, level=level, verbose=False, rebuilt=True)
    elif method == "Perso2" or method == "perso2":
        I = grad(image)
        # text = image.copy()
        # for i in range(level):
        #     text = cv.pyrDown(text)
        # for i in range(level):
        #     text = cv.pyrUp(text)
        # text = cv.resize(text, (image.shape[1], image.shape[0])) / 255
        # I = abs(image / 255 - text) * 255
    else:
        I = cv.Canny(image, low_threshold, low_threshold * ratio)
    I = ImageCustom(I/I.max()*255, i)
    I.cmap = 'EDGES'
    if orientation:
        return I, orient
    else:
        return I


def Wavelet_pyr(image, level=0, verbose=False, rebuilt=False):
    image = image / 255
    titles = ['Approximation', ' Horizontal detail',
              'Vertical detail', 'Diagonal detail', 'Phase of the gradient']
    LL, LH, HL, HH, orient = {}, {}, {}, {}, {}
    LL[0] = image
    for i in range(1, level + 1):
        coeffs2 = pywt.dwt2(LL[i - 1], 'bior1.3')
        LL[i], (LH[i], HL[i], HH[i]) = coeffs2
        orient[i] = cv.phase(LH[i], HL[i], None, angleInDegrees=True)
        # LL[i] = cv.normalize(LL[i], None, 0, 1, cv.NORM_MINMAX)
        # LH[i] = cv.normalize(LH[i], None, 0, 1, cv.NORM_MINMAX)
        # HL[i] = cv.normalize(HL[i], None, 0, 1, cv.NORM_MINMAX)
        # HH[i] = cv.normalize(HH[i], None, 0, 1, cv.NORM_MINMAX)
    fig = plt.figure(figsize=(15, 3))
    for i, a in enumerate([LL[level], LH[level], HL[level], HH[level], orient[level]]):
        ax = fig.add_subplot(1, 5, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def Harr_pyr(image, level=0, verbose=False, rebuilt=False):
    image = image / 255
    temp = {}
    texture = {}
    details = {}
    temp[0] = image.copy()
    for i in range(level + 1):
        temp[i + 1] = cv.pyrDown(temp[i])
        texture[i] = cv.resize(cv.pyrUp(temp[i + 1]), (temp[i].shape[1], temp[i].shape[0]))
        details[i] = abs(texture[i] - temp[i])
        if verbose and not rebuilt:
            im = np.hstack((texture[i], details[i]))
            cv.imshow(f'Level {i}', im)
            cv.waitKey(0)
            cv.destroyAllWindows()
    if rebuilt:
        rebuilt_details = details[level]
        for i in range(level, 0, -1):
            rebuilt_details = cv.resize(cv.pyrUp(rebuilt_details), (details[i - 1].shape[1], details[i - 1].shape[0])) \
                              + details[i - 1]
            if verbose:
                cv.imshow(f'Level {i}', rebuilt_details)
                cv.waitKey(0)
                cv.destroyAllWindows()
        return ImageCustom(abs(rebuilt_details) * 255, image)
    else:
        return texture, details


def edge_correlation(image_ir, image_rgb, idx, x_step, y_step, orient_ir, orient_rgb, level, method, k_orient):
    k = k_orient
    temp = np.zeros([image_ir.shape[0], image_ir.shape[1], level])
    for i, dx in enumerate(idx):
        if y_step[i] < 0:
            if x_step[i] < 0:
                if method == 'Perso2':
                    temp[:y_step[i], :x_step[i], i] = \
                        (image_rgb[:y_step[i], :x_step[i]] * image_ir[-y_step[i]:, -x_step[i]:]).sum(axis=2)
                else:
                    temp[:y_step[i], :x_step[i], i] = \
                        image_rgb[:y_step[i], :x_step[i]] * image_ir[-y_step[i]:, -x_step[i]:] / \
                        (k * abs(orient_rgb[:y_step[i], :x_step[i]] - orient_ir[-y_step[i]:, -x_step[i]:]) + 1)
            elif x_step[i] == 0:
                if method == 'Perso2':
                    temp[:y_step[i], :, i] = (image_rgb[:y_step[i], :] * image_ir[-y_step[i]:, :]).sum(axis=2)
                else:
                    temp[:y_step[i], :, i] = image_rgb[:y_step[i], :] * image_ir[-y_step[i]:, :] / \
                                             (k * abs(orient_rgb[:y_step[i], :] - orient_ir[-y_step[i]:, :]) + 1)
            else:
                if method == 'Perso2':
                    temp[:y_step[i], :-x_step[i], i] = (image_rgb[:y_step[i], x_step[i]:] * image_ir[-y_step[i]:,
                                                                                            :-x_step[i]]).sum(axis=2)
                else:
                    temp[:y_step[i], :-x_step[i], i] = \
                        image_rgb[:y_step[i], x_step[i]:] * image_ir[-y_step[i]:, :-x_step[i]] / \
                        (k * abs(orient_rgb[:y_step[i], x_step[i]:] - orient_ir[-y_step[i]:, :-x_step[i]]) + 1)
        elif y_step[i] == 0:
            if x_step[i] < 0:
                if method == 'Perso2':
                    temp[:, :x_step[i], i] = (image_rgb[:, :x_step[i]] * image_ir[:, -x_step[i]:]).sum(axis=2)
                else:
                    temp[:, :x_step[i], i] = image_rgb[:, :x_step[i]] * image_ir[:, -x_step[i]:] / \
                                             (k * abs(orient_rgb[:, :x_step[i]] - orient_ir[:, -x_step[i]:]) + 1)
            elif x_step[i] == 0:
                if method == 'Perso2':
                    temp[:, :, i] = (image_rgb[:, :] * image_ir[:, :]).sum(axis=2)
                else:
                    temp[:, :, i] = image_rgb[:, :] * image_ir[:, :] / (k * abs(orient_rgb[:, :] - orient_ir[:, :]) + 1)
            else:
                if method == 'Perso2':
                    temp[:, :-x_step[i], i] = (image_rgb[:, x_step[i]:] * image_ir[:, :-x_step[i]]).sum(axis=2)
                else:
                    temp[:, :-x_step[i], i] = image_rgb[:, x_step[i]:] * image_ir[:, :-x_step[i]] / \
                                              (k * abs(orient_rgb[:, x_step[i]:] - orient_ir[:, :-x_step[i]]) + 1)
        else:
            if x_step[i] < 0:
                if method == 'Perso2':
                    temp[:-y_step[i], :x_step[i], i] = \
                        (image_rgb[y_step[i]:, :x_step[i]] * image_ir[:-y_step[i], -x_step[i]:]).sum(axis=2)
                else:
                    temp[:-y_step[i], :x_step[i], i] = \
                        image_rgb[y_step[i]:, :x_step[i]] * image_ir[:-y_step[i], -x_step[i]:] / \
                        (k * abs(orient_rgb[y_step[i]:, :x_step[i]] - orient_ir[:-y_step[i], -x_step[i]:]) + 1)
            elif x_step[i] == 0:
                if method == 'Perso2':
                    temp[:-y_step[i]:, :, i + 1] = (image_rgb[y_step[i]:, :] * image_ir[:-y_step[i], :]).sum(axis=2)
                else:
                    temp[:-y_step[i]:, :, i + 1] = image_rgb[y_step[i]:, :] * image_ir[:-y_step[i], :] / \
                                                   (k * abs(orient_rgb[y_step[i]:, :] - orient_ir[:-y_step[i], :]) + 1)
            else:
                if method == 'Perso2':
                    temp[:-y_step[i]:, :-x_step[i], i] = \
                        (image_rgb[y_step[i]:, x_step[i]:] * image_ir[:-y_step[i], :-x_step[i]]).sum(axis=2)
                else:
                    temp[:-y_step[i]:, :-x_step[i], i] = \
                        image_rgb[y_step[i]:, x_step[i]:] * image_ir[:-y_step[i], :-x_step[i]] / \
                        (k * abs(orient_rgb[y_step[i]:, x_step[i]:] - orient_ir[:-y_step[i], : -x_step[i]]) + 1)
    return temp


def grad(image):
    if len(image.shape) == 3:
        image = ImageCustom(image).GRAYSCALE()
    Ix = cv.Sobel(image, cv.CV_64F, 1, 0, borderType=cv.BORDER_REFLECT_101)
    Iy = cv.Sobel(image, cv.CV_64F, 0, 1, borderType=cv.BORDER_REFLECT_101)
    grad = np.sqrt(Ix ** 2 + Iy ** 2)
    grad[grad < grad.mean()] = 0
    grad[grad > grad.mean()*5] = grad.mean()*5
    orient = cv.phase(Ix, Iy, angleInDegrees=True)

    v = cv.normalize(grad, None, 0, 255, cv.NORM_MINMAX)
    s = np.ones_like(grad) * 255
    h = cv.normalize(abs(abs(orient - 180) - 90), None, 0, 255, cv.NORM_MINMAX)
    output = np.uint8(np.stack([h, s, v], axis=-1))
    output = cv.cvtColor(output, cv.COLOR_HSV2BGR)
    return ImageCustom(output)


def create_gaborfilter(num_filters=4, ksize=5, sigma=2.0):
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
    filters = []
    # The local area to evaluate
    # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters


def apply_filter(img, filters):
    # This general function is designed to apply filters to our image
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1  # remain depth same as original image
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv.filter2D(img, depth, kern)  # Apply filter to image
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage
