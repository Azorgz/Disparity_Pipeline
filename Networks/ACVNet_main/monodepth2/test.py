import cv2 as cv

from Image import ImageCustom

im = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Video_frame/Day/visible/monocular_disparity/left00000_disp.jpeg").GRAYSCALE()

im.show()
im_ori = ImageCustom("/home/godeta/PycharmProjects/LYNRED/Video_frame/Day/visible/left/left00000.png")
fus = im_ori.HSV()
fus[:, :, 2] = im
cv.putText(fus, 'Left visible source, ' + 'Day', (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                               (255, 255, 255), 1)
fus = fus.GRAYSCALE().RGB().BGR()

cv.imshow('fus image', fus)
cv.waitKey(0)
cv.destroyAllWindows()
# cv.imwrite("/home/godeta/Pycharm
# Projects/monodepth2/assets/test_image2_cropped.png", im)