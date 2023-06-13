from Config.Config import ConfigPipe
from Networks.ACVNet.models import ACVNet
from Networks.UniMatch.unimatch.unimatch import UniMatch
from module.BaseModule import BaseModule
from utils.misc import timeit, count_parameter
import torch


class SuperNetwork(BaseModule):
    """
    This class add a layer for the data post-processing & the inputs args according each Network.
    To Run it, a normal Forward call with 2 images as inputs would do it.
    """

    def __init__(self, config: ConfigPipe):  # model, args, name: str, timeit=False):
        super(SuperNetwork, self).__init__(config)

    def _update_conf(self, config):
        self.args = config['network']["network_args"]
        self.device_index = config["device"]["index"]
        self.__class__.__name__ = 'Neural Network'
        self.name = config['network']["name"]
        if self.name == ('unimatch' or 'uni' or 'Uni' or 'Unimatch'):
            model = UniMatch(feature_channels=self.args.feature_channels,
                             num_scales=self.args.num_scales,
                             upsample_factor=self.args.upsample_factor,
                             num_head=self.args.num_head,
                             ffn_dim_expansion=self.args.ffn_dim_expansion,
                             num_transformer_layers=self.args.num_transformer_layers,
                             reg_refine=self.args.reg_refine,
                             task=self.args.task).to(self.device)
        if self.name == ("acvNet" or 'acv' or 'avc' or 'avcNet'):
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            model = ACVNet(self.args.maxdisp, self.args.attn_weights_only, self.args.freeze_attn_weights).to(
                self.device)

        if self.name == "custom":
            # self.feature_extraction = self._initialize_features_extraction_(self.config['network'])
            # self.transformer = self._initialize_transformer_(self.config['network'])
            # self.detection_head = self._initialize_detection_head_(self.config['network'])
            # self.model = torch.nn.Sequential([self.feature_extraction, self.transformer, self.detection_head])
            pass

        self.model = model.eval()
        if torch.cuda.device_count() > 1 or self.name == ("acvNet" or 'acv' or 'avc' or 'avcNet'):
            print('Use %d GPUs' % torch.cuda.device_count())
            model.model = torch.nn.DataParallel(model.model)

        if self.config['network']["path_checkpoint"]:
            checkpoint = torch.load(self.config['network']["path_checkpoint"], map_location=self.device_index)
            model_dict = self.model.state_dict()
            pre_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            model_dict.update(pre_dict)
            self.model.load_state_dict(model_dict)

    def __str__(self):
        string = super().__str__()
        string += f'The model "{self.name}" has been initialized'
        string += f"\nLoad checkpoint: {self.config['network']['path_checkpoint']}\n"
        string += count_parameter(self.model)
        return string

    @torch.no_grad()
    @timeit
    def __call__(self, im_left, im_right, *args, **kwargs):
        if self.name == "unimatch":
            return self.model(im_left, im_right,
                              attn_type=self.args.attn_type,
                              attn_splits_list=self.args.attn_splits_list,
                              corr_radius_list=self.args.corr_radius_list,
                              prop_radius_list=self.args.prop_radius_list,
                              num_reg_refine=self.args.num_reg_refine,
                              task='stereo')['flow_preds'][-1]
        elif self.name == "acvNet":
            return self.model(im_left, im_right)[0]

    def refine(self, im, flow, *args, **kwargs):
        return self.model.disp_refine(im, flow,
                                      attn_type=self.args.attn_type,
                                      attn_splits_list=self.args.attn_splits_list,
                                      corr_radius_list=self.args.corr_radius_list,
                                      prop_radius_list=self.args.prop_radius_list,
                                      num_reg_refine=self.args.num_reg_refine)['flow_preds'][-1]
