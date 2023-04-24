# from FUSION.tools.gradient_tools import edges_extraction, Harr_pyr
# from FUSION.tools.manipulation_tools import *
from ..Image import ImageCustom
from classes.tools.image_processing_tools import laplacian_fusion



def laplacian_pyr_fusion(image1, image2, mask, octave=4, verbose=False):
    if image1.shape[0] > image2.shape[0]:
        temp = image1.LAB()[:, :, 0]
        image_detail = ImageCustom(cv.pyrUp(cv.pyrDown(temp))).diff(temp)
        image1 = ImageCustom(cv.pyrDown(temp))
        image2 = ImageCustom(image2)
    elif image1.shape[0] < image2.shape[0]:
        temp = image2.LAB()[:, :, 0]
        image_detail = ImageCustom(cv.pyrUp(cv.pyrDown(temp)))/255 - temp/255
        image2 = ImageCustom(cv.pyrDown(temp))
        image1 = ImageCustom(image1)
    else:
        image_detail = None
    pyr1 = image1.GRAYSCALE().pyr_gauss(octave=octave, interval=2, sigma0=2)
    pyr2 = image2.GRAYSCALE().pyr_gauss(octave=octave, interval=2, sigma0=2)
    fus = laplacian_fusion(pyr1, pyr2, mask, verbose=verbose)
    if image_detail is not None:
        return ImageCustom(cv.pyrUp(fus)) + image_detail
    else:
        return fus

def mean(im1, im2):
    return ImageCustom(im1/2 + im2/2)

