import os
import time
import numba
import numpy as np
import torch
from torch import nn, tensor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

from estimation import Semantics, Disparity
from refinement import Refine

from utils.classes import ImageTensor
from utils.classes.Image import DepthTensor
from utils.misc import time_fct
from ultralytics import YOLO


class KenburnDepth(nn.Module):

    def __init__(self, config):
        super(KenburnDepth, self).__init__()
        path = config["path_checkpoint"]
        self.semantic_adjustment = config['network_args']['semantic_adjustment']
        self.netSemantics = Semantics().cuda().eval()
        self.netDisparity = Disparity().cuda().eval()
        self.netDisparity.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                           torch.load(path + "/network-disparity.pytorch").items()})
        # self.netMaskrcnn = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).cuda().eval()
        # Charger un modèle YOLOv8n pré-entraîné
        self.model = YOLO('pretrained/yolov8s-seg.pt').cuda()
        self.netRefine = Refine().cuda().eval()
        self.netRefine.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                                        torch.load(path + "/network-refinement.pytorch").items()})

    @torch.no_grad()
    @time_fct
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
    @time_fct
    def _disparity_adjustment(self, im, disparity):
        assert (im.shape[0] == 1)
        assert (disparity.shape[0] == 1)

        boolUsed = {}
        tenMasks = []
        # start = time.time()
        # objPredictions = self.netMaskrcnn([im[0, [2, 0, 1], :, :]])[0]
        start = time.time()
        objPredictions = self.model(source=im)[0]
        print(f"maskcrnn : {time.time()-start} seconds")
        # Iter over the masks found
        idx_yolo = [0, 1, 3, 13, *np.arange(24, len(objPredictions.names)).tolist()]
        for intMask in range(objPredictions.masks.shape[0]):
            if intMask in boolUsed:
                continue

            elif objPredictions.boxes.conf[intMask] < 0.7:
                continue

            elif objPredictions.boxes.cls[intMask] not in [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 25]:
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
            # end

            tenMasks.append(tenMask)

        # for intMask in range(objPredictions['masks'].shape[0]):
        #     if intMask in boolUsed:
        #         continue
        #
        #     elif objPredictions['scores'][intMask].item() < 0.7:
        #         continue
        #
        #     elif objPredictions['labels'][intMask].item() not in [1, 3, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        #                                                           25]:
        #         continue
        #
        #     # end
        #
        #     boolUsed[intMask] = True
        #     tenMask = (objPredictions['masks'][(intMask + 0):(intMask + 1), :, :, :] > 0.5).float()
        #
        #     if tenMask.sum().item() < 64:
        #         continue
        #     # end
        #
        #     for intMerge in range(objPredictions['masks'].shape[0]):
        #         if intMerge in boolUsed:
        #             continue
        #
        #         elif objPredictions['scores'][intMerge].item() < 0.7:
        #             continue
        #
        #         elif objPredictions['labels'][intMerge].item() not in [2, 4, 27, 28, 31, 32, 33]:
        #             continue
        #
        #         # end
        #
        #         tenMerge = (objPredictions['masks'][(intMerge + 0):(intMerge + 1), :, :, :] > 0.5).float()
        #
        #         if ((tenMask + tenMerge) > 1.0).sum().item() < 0.03 * tenMerge.sum().item():
        #             continue
        #         # end
        #
        #         boolUsed[intMerge] = True
        #         tenMask = (tenMask + tenMerge).clip(0.0, 1.0)
        #     # end
        #
        #     tenMasks.append(tenMask)
        # end
        tenAdjusted = torch.nn.functional.interpolate(input=disparity, size=(im.shape[2], im.shape[3]),
                                                      mode='bilinear', align_corners=False)

        for tenAdjust in tenMasks:
            tenPlane = tenAdjusted * tenAdjust

            tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
            tenPlane = torch.nn.functional.max_pool2d(input=tenPlane.neg(), kernel_size=3, stride=1, padding=1).neg()

            if tenPlane.sum().item() == 0:
                continue

            intLeft = (tenPlane.sum([2], True) > 0.0).flatten().nonzero()[0].item()
            intTop = (tenPlane.sum([3], True) > 0.0).flatten().nonzero()[0].item()
            intRight = (tenPlane.sum([2], True) > 0.0).flatten().nonzero()[-1].item()
            intBottom = (tenPlane.sum([3], True) > 0.0).flatten().nonzero()[-1].item()

            tenAdjusted = ((1.0 - tenAdjust) * tenAdjusted) + (
                    tenAdjust * tenPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max())
        # end

        return torch.nn.functional.interpolate(input=tenAdjusted, size=(disparity.shape[2], disparity.shape[3]),
                                               mode='bilinear', align_corners=False)

    @torch.no_grad()
    @time_fct
    def _disparity_refinement(self, im, disparity):
        return self.netRefine(im, disparity)

    @torch.no_grad()
    def forward(self, im):
        disparity = self._disparity_estimation(im)
        if self.semantic_adjustement:
            disparity = self._disparity_adjustment(im, disparity)
        return self._disparity_refinement(im, disparity)


if __name__ == '__main__':
    cpt = 'perso' if os.getcwd().split('/')[2] == 'aurelien' else 'pro'
    if cpt == 'pro':
        im_path = "/home/godeta/PycharmProjects/LYNRED/Images/"
    else:
        im_path = "/home/aurelien/Images/Images/"
    NN = KenburnDepth(os.getcwd() + "/pretrained")
    im_path = im_path + "Day/master/visible/"
    for file in os.listdir(im_path):
        if file.split('.')[-1] == 'md':
            continue
        image = ImageTensor(im_path + file).RGB().pyrDown()
        start = time.time()
        depth_im = NN(image)
        print(f'time : {time.time() - start}')
        depth_im = DepthTensor(depth_im).pyrUp()
        print(depth_im.max_value)
        depth_im.show()



