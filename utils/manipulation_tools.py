import time

import cv2 as cv
import numpy as np
from utils.classes.Image import ImageCustom


def fusion_scaled(gray, rgb, ratio=0.5):
    if gray.shape[0] == rgb.shape[0]:
        fus = np.uint8(gray * ratio + rgb * (1 - ratio))
    else:
        ## If ratio = 1 : only grayscale, 0 : only vis
        if gray.shape[0] != rgb.shape[0] / 2:
            gray, rgb = size_matcher(gray, rgb)
        temp = cv.pyrDown(rgb)
        detail = cv.subtract(rgb, cv.pyrUp(temp))
        fus = np.uint8(gray * ratio + temp * (1 - ratio))
        fus = cv.add(cv.pyrUp(fus), detail)
    return ImageCustom(fus, rgb)


def crop_image(im, ratio=0, size=(640, 480)):
    m, n = im.shape[:2]

    if not ratio:
        image = cv.resize(im, size)
    elif m / n < ratio:
        dy = round((n - m / ratio) / 2)
        dx = 0
        image = im[:, dy:-dy]
    elif m / n > ratio:
        dy = 0
        dx = round((m - n * ratio) / 2)
        image = im[dx:-dx, :]
    else:
        image = im
    return image


def size_matcher(gray, rgb, size=0):
    if isinstance(size, tuple):
        gray = ImageCustom(cv.resize(gray, size, interpolation=cv.INTER_AREA), gray)
        rgb = ImageCustom(cv.resize(rgb, size, interpolation=cv.INTER_AREA), rgb)
        return gray, rgb
    else:
        if gray.shape[0] > rgb.shape[0]:
            rgb = rgb.copy()
            gray = ImageCustom(
                cv.resize(gray, (rgb.shape[1] * 2, rgb.shape[0] * 2), interpolation=cv.INTER_AREA), gray)
        else:
            gray = gray.copy()
            rgb = ImageCustom(
                cv.resize(rgb, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv.INTER_AREA), rgb)
        # else:
        #     gray = gray.copy()
        #     rgb = rgb.copy()
        return gray, rgb


def normalisation(image, unit='uint8'):
    im = image.copy()
    if unit == 'uint8':
        try:
            im = ImageCustom((image - image.min()) / (image.max() - image.min()) * 255, image)
        except "Divide by zero":
            im = image.copy()
    return im


def normalisation_tensor(image):
    m, M = image.min(), image.max()
    if m != M:
        return (image - m) / (M - m)
    else:
        return image


def manual_calibration(image_ir, rgb):
    global im_temp, pts_temp, rightclick

    def mouseHandler(event, x, y, flags, param):
        global pts_temp, rightclick
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(im_temp, (x, y), 2, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(im_temp, str(len(pts_temp) + 1), (x + 3, y + 3),
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 200, 250), 1, cv.LINE_AA)
            if len(pts_temp) < 50:
                pts_temp = np.append(pts_temp, [(x, y)], axis=0)
            if rightclick == 0:
                cv.imshow('image Vis, first selection window', im_temp)
            elif rightclick == 1:
                cv.imshow('image RGB, second selection window', im_temp)
        if event == cv.EVENT_RBUTTONDOWN or event == cv.EVENT_MOUSEWHEEL:
            rightclick = 2

    rightclick = 0
    # Create a black image, a window and bind the function to window

    # Image RGB
    image_rgb = ImageCustom(cv.cvtColor(rgb, cv.COLOR_RGB2BGR), rgb)

    # Vector temp
    pts_temp = np.empty((0, 2), dtype=np.int32)
    # Create a window
    im_temp = image_ir.BGR()
    cv.namedWindow('image Vis, first selection window')
    cv.imshow('image Vis, first selection window', im_temp)
    cv.namedWindow('image RGB, second selection window')
    cv.imshow('image RGB, second selection window', image_rgb)
    cv.setMouseCallback('image Vis, first selection window', mouseHandler)
    while (1):
        if len(pts_temp) == 12 or cv.waitKey(20) & 0xFF == 27 or rightclick == 2:
            rightclick = 1
            break
    # cv.destroyAllWindows()
    pts_src = pts_temp

    # Destination image
    # gray = image_rgb.GRAYSCALE()
    im_temp = image_rgb.BGR()
    pts_temp = np.empty((0, 2), dtype=np.int32)
    # Create a window
    height, width = im_temp.shape[:2]
    cv.setMouseCallback('image RGB, second selection window', mouseHandler)
    while (1):
        if len(pts_temp) == len(pts_src) or cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()
    pts_dst = pts_temp

    tform, status = cv.findHomography(pts_src, pts_dst)
    im_temp = ImageCustom(cv.warpPerspective(image_ir, tform, (width, height)))
    fusion = np.uint8(im_temp.GRAYSCALE() / 2 + image_rgb.GRAYSCALE() / 2)
    # I = rgb.LAB()
    # I[:, :, 0] = fusion
    # I = cv.cvtColor(I, cv.COLOR_LAB2BGR)
    cv.imshow('fusion', fusion)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return pts_src, pts_dst, tform


def choose(choice1, choice2):
    numpy_horizontal = np.hstack((choice1, choice2))
    global choice

    def mouseHandler(event, x, y, flags, param):
        global choice
        if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:

            if x < choice1.shape[0]:
                choice = 1
            else:
                choice = 2

    choice = 0
    cv.imshow('Choose a result !', numpy_horizontal)
    cv.setMouseCallback('Choose a result !', mouseHandler)
    while 1:
        if choice != 0 or cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()
    return choice


def function_timer(activate):
    def overload_func(func):
        if activate:
            print(f'The function {func} is timed'
                  f'\nExecution....')

            def decorateur(*args, **kwargs):
                start = time.time()
                output = func(*args, **kwargs)
                print(f'Total time : {time.time() - start} seconds')
                return output

            return decorateur
        else:
            def decorateur(*args, **kwargs):
                output = func(*args, **kwargs)
                return output

            return decorateur

    return overload_func
