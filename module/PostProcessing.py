import torch
import matplotlib.pyplot as plt

from module.BaseModule import BaseModule
from utils.classes import ImageCustom
from utils.misc import timeit
from utils.transforms import Unpad, Resize_disp, Compose
from torchvision.transforms.functional import hflip


class PostProcessing(BaseModule):
    """
    A class which is instanced automatically if some post-process function are demanded or if the
    Disparity image needs a Resize or an un-padding
    """

    def __init__(self, config, pad=None, resize=None, post_process=None):
        super(PostProcessing, self).__init__(config, pad=pad, resize=resize, post_process=post_process)

    def _update_conf(self, config, *args, **kwargs):
        # path_ref = "/home/godeta/PycharmProjects/Disparity_Pipeline/average_disp.png"
        # self.ref = torch.tensor(ImageCustom(path_ref)).to(torch.device('cuda'))/256
        pad, resize, post_process = kwargs['pad'], kwargs['resize'], kwargs['post_process']
        self.pred_bidir_disp = self.config['dataset']["pred_bidir_disp"]
        self.pred_right_disp = self.config['dataset']["pred_right_disp"]
        self.transform = []
        if pad:
            self.transform.append(Unpad(pad.pad, pad.ori_size))
        if resize:
            self.ori_size = resize.ori_size
            self.transform.append(Resize_disp(self.ori_size))
        if post_process:
            pass
        self.transform = Compose(self.transform, self.device)
        # self.histo = None

    def __str__(self):
        return ''

    @timeit
    def __call__(self, disp, *args):
        disp = self.transform(disp)
        if self.pred_bidir_disp:
            disp[1] = hflip(disp[1])
        elif self.pred_right_disp:
            disp = hflip(disp)
        # self.bins = int(disp[0].max())
        # self.histo = torch.histc(disp[0], bins=self.bins)
        # ref_hist = torch.histc(self.ref*disp.max(), bins=self.bins)
        # self.histo -= ref_hist
        return disp
