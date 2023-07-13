import time
import cv2 as cv
import kornia
import numpy as np
import torch

# from classes.Image import ImageCustom


def perspective2grid(matrix, shape, device):
    height, width = shape[0], shape[1]
    grid_reg = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False, device=device)  # [1 H W 2]
    z = torch.ones_like(grid_reg[:, :, :, 0])
    grid = torch.stack([grid_reg[..., 0], grid_reg[..., 1], z], dim=-1)  # [1 H W 3]

    grid_transformed = (matrix @ grid.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [1 H W 3]
    alpha = grid_transformed[:, :, :, 2]
    grid_transformed[:, :, :, 0] = 2 * (grid_transformed[:, :, :, 0] / alpha) / width - 1  # [1 H W 3]
    grid_transformed[:, :, :, 1] = 2 * (grid_transformed[:, :, :, 1] / alpha) / height - 1  # [1 H W 3]
    return grid_transformed[..., :2]  # [1 H W 2]

def histogram_equalization(image, method=0):
    if method == 0:
        return image
    elif method == 1:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if image.cmap == 'GRAYSCALE':
            # start = time.time()
            image = ImageCustom(clahe.apply(image), image)
            # check1 = time.time() - start
            # print(f"1st step : {check1} sec")
        elif image.cmap == 'RGB' or image.cmap == 'BGR':
            # start = time.time()
            image = image.LAB()
            # check1 = time.time() - start
            # start = time.time()
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            # check2 = time.time() - start
            # start = time.time()
            image = image.BGR()
            # check3 = time.time() - start
            # print(f"1st step : {check1} sec\n"
            #       f"2nd step : {check2} sec\n"
            #       f"3rd step : {check3} sec")
        return image
    elif method == 2:
        if image.cmap == 'GRAYSCALE':
            image = ImageCustom(cv.equalizeHist(image), image)
        elif image.cmap == 'RGB' or image.cmap == 'BGR':
            image = image.LAB()
            image[:, :, 0] = cv.equalizeHist(image[:, :, 0])
            image = image.BGR()
        return image


def normalization_maps(image, image2=None):
    """
    Function used by the mask generator
    :param image: image to normalize
    :return: normalized and center image
    """
    # Normalization
    if image2 is None:
        M = image.max()
        mi = image.min()
        m = image.mean()
    else:
        M = max(image.max(), image2.max())
        mi = min(image.min(), image2.min())
        m = image.mean() / 2 + image2.mean() / 2
        im2 = 1 - abs(image2 / 1.0 - m) / M #Compute the distance to the mean of the image
        im2 = (im2 - im2.min()) / (2 * (im2.max() - im2.min())) + 0.5 #Normalize this distance between 0.5 and 1
        res2 = image2 * im2 #Normalize the input image between 0 and 1 and weight it by the mask computed before
        res2 = (res2 - res2.min()) / (res2.max() - res2.min()) * 255 #Normalize the output between 0 and 255
    im = 1 - abs(image / 1.0 - m) / M
    im = (im - im.min()) / (2 * (im.max() - im.min())) + 0.5
    res = image * im
    res = (res - res.min()) / (res.max() - res.min()) * 255
    if image2 is None:
        return ImageCustom(res)
    else:
        return ImageCustom(res), ImageCustom(res2)


def scaled_fusion(pyramid, method_interval, method_scale, pyramid2=None, method_fusion=None, first_level_scaled=False):
    """
    :param pyramid: Dictionary shaped like a Gaussian pyramid
    :param method_interval: function used to fuse intra-interval images
    :param method_scale: function used to fuse inter-scale images
    :param pyramid2: Optionnal, to fuse two pyramid together at the interval level
    :param method_fusion: Mendatory if a second pyramid is specified, function to fuse images between the two pyramids
    :return:
    """
    new_pyr = {}
    for key in pyramid.keys():
        ref = pyramid[key][list(pyramid[key].keys())[0]]
        for inter, im in pyramid[key].items():
            if pyramid2 is not None:
                im = method_fusion(pyramid[key][inter], pyramid2[key][inter])
            ref = method_interval(ref, im)
        new_pyr[key] = ref
    scale = np.array(list(new_pyr.keys()))
    scale_diff = scale[1:] - scale[:-1]
    if len(scale) == 1 or first_level_scaled:
        return new_pyr[scale[0]]
    for idx, key in enumerate(reversed(scale[1:])):
        temp = new_pyr[key]
        diff = scale_diff[-(1 + idx)]
        for i in range(diff):
            temp = cv.pyrUp(temp)
        im_large = new_pyr[key - diff]
        new_pyr[key - diff] = method_scale(im_large, temp)
    return ImageCustom(new_pyr[scale[0]])


def laplacian_fusion(pyramid, pyramid2, mask_fusion, verbose=False):
    """
    :param pyramid: Dictionary shaped like a Gaussian pyramid
    :param pyramid2: Optionnal, to fuse two pyramid together at the interval level
    :param mask_fusion: Mendatory to fuse images between the two pyramids
    :return:
    """
    new_pyr = {}
    new_pyr2 = {}
    masks = {}
    for key in pyramid.keys():
        if key == 0:
            new_pyr[key] = pyramid[key]
            new_pyr2[key] = pyramid2[key]
        else:
            new_pyr[key] = pyramid[key][1]
            new_pyr2[key] = pyramid2[key][1]
        masks[key] = cv.resize(mask_fusion, (new_pyr[key].shape[1], new_pyr[key].shape[0]))
    scale = np.array(list(new_pyr.keys())[1:])
    scale_diff = scale[1:] - scale[:-1]
    temp = None
    for idx, key in enumerate(reversed(scale)):
        temp1 = new_pyr[key]
        temp2 = new_pyr2[key]
        if idx < len(scale) - 1:
            diff = scale_diff[-(1 + idx)]
            for i in range(diff):
                temp11 = cv.pyrUp(temp1)
                temp22 = cv.pyrUp(temp2)
        else:
            temp11 = temp1
            temp22 = temp2
        detail1 = new_pyr[key - diff] / 255 - temp11 / 255
        detail2 = new_pyr2[key - diff] / 255 - temp22 / 255
        details = detail1 * masks[key - diff] + detail2 * (1 - masks[key - diff])
        if temp is None:
            temp = ImageCustom(temp11 * masks[key - diff] + ImageCustom(temp22 * (1 - masks[key - diff]))
                               + details * 255, pyramid[0])
        elif idx < len(scale) - 1:
            temp = ImageCustom(cv.pyrUp(temp) + details * 255, pyramid[0])
        else:
            temp = temp + details * 255
            temp = ImageCustom((temp - temp.min()) / (temp.max() - temp.min())*255, pyramid[0])
        if verbose:
            cv.imshow('fusion', temp)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return temp.unpad()


def blobdetection(image):
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 50

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.maxCircularity = 1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv.SimpleBlobDetector_create(params)
    if image.dtype == np.float64:
        im = np.uint8(image * 255)
    else:
        im = image.copy()
    keypoints = detector.detect(im)
    print(keypoints)
    im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                         cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints
