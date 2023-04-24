from classes.Image import ImageCustom
import numpy as np
import cv2 as cv

im = ImageCustom("/home/godeta/PycharmProjects/Disparity_Pipeline/results/Test/disp_other/0001_disp_other.png")
im0 = np.uint8(im < 5) * 255

cv.imshow('orignal', im)
cv.imshow('filtered', im0)
cv.waitKey(0)