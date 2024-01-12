import torch
import matplotlib.pyplot as plt

from module.BaseModule import BaseModule
# from utils.classes import ImageCustom
from utils.misc import timeit
from utils.transforms import Unpad, ResizeDisp, Compose, ResizeDepth
from torchvision.transforms.functional import hflip


class PostProcessing:
    """
    A class which is instanced automatically if some post-process function are demanded or if the
    Disparity image needs a Resize or an un-padding
    """

    def __init__(self, config, device, task=None, pad=None, resize=None, post_process=None):
        pad, resize, post_process = pad, resize, post_process
        self.config = config
        self.device = device
        self.task = task
        self.transform = []
        if pad:
            self.ori_size = pad.ori_size
            self.transform.append(Unpad(pad.pad, pad.ori_size))
        if resize:
            self.ori_size = resize.ori_size
            if task == 'disparity':
                self.transform.append(ResizeDisp(self.ori_size))
            else:
                self.transform.append(ResizeDepth(self.ori_size))
        if post_process:
            pass
        self.transform = Compose(self.transform, self.device)
        # self.histo = None

    def __call__(self, sample, *args):
        for size, (key, disp) in zip(self.ori_size, sample.items()):
            disp = self.transform(disp, size=size)
            if self.config['pred_bidir'] and self.task != 'depth' and self.task != 'monocular':
                disp[1] = hflip(disp[1])
            elif self.config['pred_right'] and self.task != 'monocular':
                disp = hflip(disp)
            sample[key] = disp
        return sample
