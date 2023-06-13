from module.BaseModule import BaseModule
from module.PostProcessing import PostProcessing
from utils import transforms
from utils.misc import timeit
from utils.transforms import Resize, Pad


class Preprocessing(BaseModule):
    """
    A class which define apply the preprocessing necessary for each Network
    """

    def __init__(self, config):
        super(Preprocessing, self).__init__(config)

    def _update_conf(self, config, *args, **kwargs):
        transform = config["transform"]
        transform.append(transforms.DispSide(config['dataset']["pred_right_disp"],
                                             config['dataset']["pred_bidir_disp"]))
        self.transforms = transforms.Compose(transform, config["device"]["device"])
        self.pad = None
        self.resize = None

    def __str__(self):
        return ""

    @timeit
    def __call__(self, sample, dp, *args, **kwargs):
        self.inference_size = None
        self.ori_size = None
        if self.transforms is not None:
            sample = self.transforms(sample)
        for t in self.transforms.transforms:
            if isinstance(t, Resize):
                self.inference_size = t.inference_size
                self.ori_size = t.ori_size
                self.resize = t
            if isinstance(t, Pad):
                self.pad = t
                self.ori_size = t.ori_size
        if not isinstance(dp.postprocessing, PostProcessing):
            dp.disparity_network.args.inference_size = self.inference_size
            dp.disparity_network.args.ori_size = self.ori_size
            dp.disparity_network.args.pred_right_disp = dp.config['dataset']["pred_right_disp"]
            dp.disparity_network.args.pred_bidir_disp = dp.config['dataset']["pred_bidir_disp"]
            dp.postprocessing = PostProcessing(self.config,
                                               pad=self.pad,
                                               resize=self.resize,
                                               post_process=None)
            dp.modules.append(dp.postprocessing)
        return sample
