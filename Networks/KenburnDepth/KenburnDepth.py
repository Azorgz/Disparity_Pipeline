import os
import time
from typing import Union

import numba
import numpy as np
import torch
from kornia.utils import get_cuda_device_if_available
from torch import nn, tensor, Tensor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

from .estimation import Semantics, Disparity
from .refinement import Refine

from utils.classes import ImageTensor
from utils.classes.Image import DepthTensor
from utils.misc import time_fct
from ultralytics import YOLO


class KenburnDepth(nn.Module):

    def __init__(self, config, device=None):
        super(KenburnDepth, self).__init__()
        path = config["path_checkpoint"]
        self.semantic_adjustment = config['network_args'].semantic_adjustment
        self.semantic_network = config['network_args'].semantic_network
        self.device = device if device is not None else get_cuda_device_if_available()
        self.netSemantics = Semantics().to(device=self.device).eval()
        self.netDisparity = Disparity().to(device=self.device).eval()
        self.netDisparity.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                           torch.load(path + "/network-disparity.pytorch").items()})
        if self.semantic_adjustment:
            if self.semantic_network == 'YOLO':
                self.netMaskrcnn = YOLO(os.getcwd() + '/' + path + '/yolov8s-seg.pt').to(device=self.device)
                self.netMaskrcnn.training = False
            else:
                self.netMaskrcnn = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(
                    device=self.device).eval()
        self.netRefine = Refine().cuda().eval()
        self.netRefine.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                        torch.load(path + "/network-refinement.pytorch").items()})

    @torch.no_grad()
    def _disparity_estimation(self, im):
        # im = im.pyrDown().pyrDown()
        size = 512
        intWidth = im.shape[3]
        intHeight = im.shape[2]

        fltRatio = float(intWidth) / float(intHeight)

        intWidth = min(int(size * fltRatio), size)
        intHeight = min(int(size / fltRatio), size)

        tenImage = torch.nn.functional.interpolate(input=tensor(im), size=(intHeight, intWidth), mode='bilinear',
                                                   align_corners=False)

        return self.netDisparity(tenImage, self.netSemantics(tenImage))

    @torch.no_grad()
    def _disparity_adjustment(self, im, disparity):
        assert (im.shape[0] == 1)
        assert (disparity.shape[0] == 1)

        boolUsed = {}
        tenMasks = []

        if self.semantic_network == 'YOLO':
            objPredictions = self.netMaskrcnn(source=im)[0]
            # Iter over the masks found
            idx_yolo = [0, 1, 3, 13, *np.arange(24, len(objPredictions.names)).tolist()]
            for intMask in range(objPredictions.masks.shape[0]):
                if intMask in boolUsed:
                    continue

                elif objPredictions.boxes.conf[intMask] < 0.7:
                    continue

                elif objPredictions.boxes.cls[intMask] not in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                               17, 18, 19, 25]:
                    continue

                boolUsed[intMask] = True
                tenMask = (objPredictions.masks.data[(intMask + 0):(intMask + 1), :, :]).unsqueeze(0)

                if tenMask.sum() < 64:
                    continue
                # end
                if objPredictions.boxes.cls[intMask] in idx_yolo:
                    for intMerge in range(intMask, objPredictions.masks.shape[0], 1):
                        if intMerge in boolUsed:
                            continue

                        elif objPredictions.boxes.conf[intMerge] < 0.7:
                            continue

                        elif objPredictions.boxes.cls[intMerge] not in idx_yolo:
                            continue

                        # end

                        tenMerge = (objPredictions.masks.data[(intMerge + 0):(intMerge + 1), :, :]).unsqueeze(0)

                        if ((tenMask + tenMerge) > 1.0).sum().item() < 0.03 * tenMerge.sum().item():
                            continue
                        # end

                        boolUsed[intMerge] = True
                        tenMask = (tenMask + tenMerge).clip(0.0, 1.0)
                tenMasks.append(tenMask)
            # end
        else:
            objPredictions = self.netMaskrcnn([im[0, [2, 0, 1], :, :]])[0]
            for intMask in range(objPredictions['masks'].shape[0]):
                if intMask in boolUsed:
                    continue

                elif objPredictions['scores'][intMask].item() < 0.7:
                    continue

                elif objPredictions['labels'][intMask].item() not in [1, 3, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23,
                                                                      24,
                                                                      25]:
                    continue

                boolUsed[intMask] = True
                tenMask = (objPredictions['masks'][(intMask + 0):(intMask + 1), :, :, :] > 0.5).float()

                if tenMask.sum().item() < 64:
                    continue

                for intMerge in range(objPredictions['masks'].shape[0]):
                    if intMerge in boolUsed:
                        continue

                    elif objPredictions['scores'][intMerge].item() < 0.7:
                        continue

                    elif objPredictions['labels'][intMerge].item() not in [2, 4, 27, 28, 31, 32, 33]:
                        continue

                    # end

                    tenMerge = (objPredictions['masks'][(intMerge + 0):(intMerge + 1), :, :, :] > 0.5).float()

                    if ((tenMask + tenMerge) > 1.0).sum().item() < 0.03 * tenMerge.sum().item():
                        continue
                    # end

                    boolUsed[intMerge] = True
                    tenMask = (tenMask + tenMerge).clip(0.0, 1.0)
                # end
                tenMasks.append(tenMask)

        tenAdjusted = torch.nn.functional.interpolate(input=disparity, size=(im.shape[2], im.shape[3]),
                                                      mode='bilinear', align_corners=False)

        for tenAdjust in tenMasks:
            tenPlane = tenAdjusted * tenAdjust

            tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
            tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()

            if tenPlane.sum().item() == 0:
                continue

            # intLeft = (tenPlane.sum([2], True) > 0.0).flatten().nonzero()[0].item()
            intTop = (tenPlane.sum([3], True) > 0.0).flatten().nonzero()[0].item()
            # intRight = (tenPlane.sum([2], True) > 0.0).flatten().nonzero()[-1].item()
            intBottom = (tenPlane.sum([3], True) > 0.0).flatten().nonzero()[-1].item()

            tenAdjusted = ((1.0 - tenAdjust) * tenAdjusted) + (
                    tenAdjust * tenPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max())
        # end

        return torch.nn.functional.interpolate(input=tenAdjusted, size=(disparity.shape[2], disparity.shape[3]),
                                               mode='bilinear', align_corners=False)

    @torch.no_grad()
    def _disparity_refinement(self, im, disparity):
        return self.netRefine(im, disparity)

    @torch.no_grad()
    def forward(self, images: Union[np.array, torch.tensor, ImageTensor, list, dict], *args, focal: Union[list, float, int] = 1, **kwargs):
        """
        Takes as many images as input as you want. It will predict their depth successively
        The focal argument need to be only one float or int (in mm) or a list (1 per images)
        """
        names = []
        if isinstance(images, dict):
            im = []
            for k, v in images.items():
                names.append(k)
                im.append(v)
            images = im
        if not isinstance(images, list):
            images = [images]
        if args:
            images.append(*kwargs)
        if not isinstance(focal, list):
            focal = [focal]
        if len(focal) > 1:
            assert len(focal) == len(images)
        else:
            focal = [focal[0] for i in range(len(images))]
        depth = []
        for image, f in zip(images, focal):
            f *= 1000
            image = Tensor(image)
            disparity = self._disparity_estimation(image)
            if self.semantic_adjustment:
                disparity = self._disparity_adjustment(image, disparity)
            disparity = self._disparity_refinement(image, disparity)
            disparity = disparity / disparity.max()
            tenDepth = f / (disparity + 0.0000001)
            depth.append(tenDepth)
        if names:
            return {name: d for name, d in zip(names, depth)}
        else:
            return depth

# if __name__ == '__main__':
#     cpt = 'perso' if os.getcwd().split('/')[2] == 'aurelien' else 'pro'
#     if cpt == 'pro':
#         im_path = "/home/godeta/PycharmProjects/LYNRED/Images/"
#     else:
#         im_path = "/home/aurelien/Images/Images/"
#     NN = KenburnDepth(os.getcwd() + "/pretrained")
#     im_path = im_path + "Day/master/visible/"
#     for file in os.listdir(im_path):
#         if file.split('.')[-1] == 'md':
#             continue
#         image = ImageTensor(im_path + file).RGB().pyrDown()
#         start = time.time()
#         depth_im = NN(image)
#         print(f'time : {time.time() - start}')
#         depth_im = DepthTensor(depth_im).pyrUp()
#         print(depth_im.max_value)
#         depth_im.show()
