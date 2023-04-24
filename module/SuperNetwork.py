import torch.nn.functional as F
from torchvision.transforms.functional import hflip

from utils.misc import timeit


class SuperNetwork:
    """
    This class add a layer for the data post-processing & the inputs args according each Network.
    To Run it, a normal Forward call with 2 images as inputs would do it.
    """

    def __init__(self, model, args, name: str, timeit=False):
        self.model = model
        self.name = name
        self.args = args
        # If the DisparityPipe need to measure the execution time of each block, this parameter will be set to True.
        if timeit:
            self.timeit = []

    @timeit
    def __call__(self, im_left, im_right, *args, **kwargs):
        pred_disp = None
        if self.name == "unimatch":
            pred_disp = self.model(im_left, im_right,
                                   attn_type=self.args.attn_type,
                                   attn_splits_list=self.args.attn_splits_list,
                                   corr_radius_list=self.args.corr_radius_list,
                                   prop_radius_list=self.args.prop_radius_list,
                                   num_reg_refine=self.args.num_reg_refine,
                                   task='stereo')['flow_preds'][-1]
            if self.args.inference_size[0] != self.args.ori_size[0] or \
                    self.args.inference_size[1] != self.args.ori_size[1]:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=self.args.ori_size,
                                          mode='bilinear',
                                          align_corners=True).squeeze(1)  # [1, H, W]
                pred_disp = pred_disp * self.args.ori_size[-1] / float(self.args.inference_size[-1])
        if self.args.pred_bidir_disp:
            pred_disp[1] = hflip(pred_disp[1])
        elif self.args.pred_right_disp:
            pred_disp = hflip(pred_disp)
        return pred_disp
