import time
import warnings

import torch
from torch import Tensor
from Networks.ACVNet.models import ACVNet
from Networks.KenburnDepth.KenburnDepth import KenburnDepth
from Networks.UniMatch.unimatch.unimatch import UniMatch
from config.Config import ConfigPipe
from module.BaseModule import BaseModule
from module.Preprocessing import Preprocessing
from utils.classes.Image import DepthTensor
from utils.misc import timeit, count_parameter


class SuperNetwork(BaseModule):
    Network = {'disparity': ['ACVNet', 'UniMatch'], 'depth': ['UniMatch', 'Kenburn']}
    """
    This class add a layer for the data post-processing & the inputs args according each Network implemented.
    To Run it, a normal Forward call with 2 images as inputs would do it.
    """

    def __init__(self, config: ConfigPipe, *args, **kwargs):  # model, args, name: str, timeit=False):
        self.name_disparity = ''
        self.model_disparity = None
        self.model_depth = None
        self.name_depth = ''
        self.preprocessing_disparity = None
        self.preprocessing_depth = None
        self.pred_right = False
        self.pred_bidir = False
        super(SuperNetwork, self).__init__(config, *args, **kwargs)

    def _update_conf(self, config, *args, **kwargs):
        self.args_disparity = config['disparity_network']["network_args"]
        self.args_depth = config['depth_network']["network_args"]
        self.device_index = config["device"]["index"]
        self.__class__.__name__ = 'Neural Network'
        self.disparity_network_init(config)
        self.depth_network_init(config)
        # self.preprocessing_disparity_init(config)
        # self.preprocessing_depth_init(config)

    def update_pred_right(self, activate: bool = True):
        self.pred_right = activate
        self.update_preprocessing()

    def update_pred_bidir(self, activate: bool = True):
        self.pred_bidir = activate
        self.update_preprocessing()

    def update_size(self, size, network='disparity'):
        if size is not None:
            if network == 'disparity':
                self.preprocessing_disparity.inference_size = size
            else:
                self.preprocessing_depth.inference_size = size

    def update_preprocessing(self):
        self.preprocessing_disparity = Preprocessing(self.config["disparity_network"]["preprocessing"], self.device,
                                                     task='disparity', pred_right=self.pred_right,
                                                     pred_bidir=self.pred_bidir)
        self.preprocessing_depth = Preprocessing(self.config["depth_network"]["preprocessing"], self.device,
                                                 task='depth', pred_right=self.pred_right, pred_bidir=self.pred_bidir)

    def disparity_network_init(self, config):
        # Disparity Network initialization
        self.name_disparity = config['disparity_network']["name"].upper()
        if self.name_disparity == 'UNI' or self.name_disparity == 'UNIMATCH':
            self.name_disparity = 'unimatch'
            model = UniMatch(feature_channels=self.args_disparity.feature_channels,
                             num_scales=self.args_disparity.num_scales,
                             upsample_factor=self.args_disparity.upsample_factor,
                             num_head=self.args_disparity.num_head,
                             ffn_dim_expansion=self.args_disparity.ffn_dim_expansion,
                             num_transformer_layers=self.args_disparity.num_transformer_layers,
                             reg_refine=self.args_disparity.reg_refine,
                             task=self.args_disparity.task).to(self.device)
        elif self.name_disparity == "ACVNET" or self.name_disparity == 'ACV' or self.name_disparity == 'AVC':
            self.name_disparity = 'acvNet'
            torch.manual_seed(1)
            torch.cuda.manual_seed(1)
            model = ACVNet(self.args_disparity.maxdisp, self.args_disparity.attn_weights_only,
                           self.args_disparity.freeze_attn_weights).to(self.device)
        # elif config['network']["name"] == "custom":
        # self.name_disparity = 'custom'
        # self.feature_extraction = self._initialize_features_extraction_(self.config['network'])
        # self.transformer = self._initialize_transformer_(self.config['network'])
        # self.detection_head = self._initialize_detection_head_(self.config['network'])
        # self.model = torch.nn.Sequential([self.feature_extraction, self.transformer, self.detection_head])
        # pass
        else:
            model = None
        self.model_disparity = model.eval()
        if torch.cuda.device_count() > 1 or self.name_disparity.upper() == ("ACVNET" or 'ACV' or 'AVC'):
            print('Use %d GPUs' % torch.cuda.device_count())
            model.model = torch.nn.DataParallel(model.model)

        if self.config['disparity_network']["path_checkpoint"]:
            checkpoint = torch.load(self.config['disparity_network']["path_checkpoint"], map_location=self.device_index)
            model_dict = self.model_disparity.state_dict()
            pre_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            model_dict.update(pre_dict)
            self.model_disparity.load_state_dict(model_dict)
        self.preprocessing_disparity = Preprocessing(config["disparity_network"]["preprocessing"], self.device,
                                                     task='disparity',
                                                     pred_right=self.pred_right, pred_bidir=self.pred_bidir)

    def depth_network_init(self, config):
        # Disparity Network initialization
        self.name_depth = config['depth_network']["name"].upper()
        if self.name_depth == 'UNI' or self.name_depth == 'UNIMATCH':
            self.name_depth = "unimatch"
            model = UniMatch(feature_channels=self.args_depth.feature_channels,
                             num_scales=self.args_depth.num_scales,
                             upsample_factor=self.args_depth.upsample_factor,
                             num_head=self.args_depth.num_head,
                             ffn_dim_expansion=self.args_depth.ffn_dim_expansion,
                             num_transformer_layers=self.args_depth.num_transformer_layers,
                             reg_refine=self.args_depth.reg_refine,
                             task='depth')
            self.model_depth = model.to(device=self.device)
            if torch.cuda.device_count() > 1:
                print('Use %d GPUs' % torch.cuda.device_count())
                self.model_depth.model = torch.nn.DataParallel(self.model_depth.model)
            if self.config['depth_network']["path_checkpoint"]:
                checkpoint = torch.load(self.config['depth_network']["path_checkpoint"], map_location=self.device_index)
                model_dict = self.model_depth.state_dict()
                pre_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
                model_dict.update(pre_dict)
                self.model_depth.load_state_dict(model_dict)
        elif self.name_depth == 'KENBURNDEPTH' or self.name_depth == 'KENBURN' or self.name_depth == 'KEN':
            self.name_depth = "KenBurnDepth"
            model = KenburnDepth(self.config['depth_network'], device=self.device)
            self.model_depth = model.to(device=self.device)
        elif config['depth_network']["name"] == "custom":
            self.name_depth = "custom"
            # self.feature_extraction = self._initialize_features_extraction_(self.config['network'])
            # self.transformer = self._initialize_transformer_(self.config['network'])
            # self.detection_head = self._initialize_detection_head_(self.config['network'])
            # self.model = torch.nn.Sequential([self.feature_extraction, self.transformer, self.detection_head])
            pass
        else:
            model = None
        self.preprocessing_depth = Preprocessing(config["depth_network"]["preprocessing"], self.device, task='depth',
                                                 pred_right=self.pred_right, pred_bidir=False)

    def __str__(self):
        string = super().__str__()
        string += f'The model "{self.name_disparity}" has been initialized for disparity'
        string += f"\nLoad checkpoint: {self.config['disparity_network']['path_checkpoint']}\n"
        string += count_parameter(self.model_disparity)
        string += f'\nThe model "{self.name_depth}" has been initialized for depth'
        string += f"\nLoad checkpoint: {self.config['depth_network']['path_checkpoint']}\n"
        string += count_parameter(self.model_depth)
        return string

    @torch.no_grad()
    @timeit
    def __call__(self, sample, *args, depth=False, intrinsics=None, pose=None, focal=0, **kwargs):
        if not depth:
            sample = self.preprocessing_disparity(sample.copy())
            im_left, im_right = sample['left'], sample['right']
            if self.name_disparity == "unimatch":
                if im_left.im_type == 'IR':
                    im_left = im_left.RGB('gray')
                if im_right.im_type == 'IR':
                    im_right = im_right.RGB('gray')
                res = self.model_disparity(Tensor(im_left), Tensor(im_right),
                                           attn_type=self.args_disparity.attn_type,
                                           attn_splits_list=self.args_disparity.attn_splits_list,
                                           corr_radius_list=self.args_disparity.corr_radius_list,
                                           prop_radius_list=self.args_disparity.prop_radius_list,
                                           num_reg_refine=self.args_disparity.num_reg_refine,
                                           task='stereo')['flow_preds'][-1]
            elif self.name_disparity == "acvNet":
                res = self.model_disparity(im_left, im_right)[0]
            else:
                warnings.warn('This Network is not implemented')
            if self.pred_bidir:
                left = DepthTensor(res[0], device=self.device).scale()
                left.im_name = im_left.im_name
                right = DepthTensor(res[1], device=self.device).scale()
                right.im_name = im_right.im_name
                res = {'left': left, 'right': right}
            elif self.pred_right:
                right = DepthTensor(res[1], device=self.device).scale()
                right.im_name = im_right.im_name
                res = {'right': right}
            else:
                left = DepthTensor(res[0], device=self.device).scale()
                left.im_name = im_left.im_name
                res = {'left': left}
            res = self.preprocessing_disparity(res, reverse=True)
            return res

        else:
            sample = self.preprocessing_depth(sample)
            img_ref, img_tgt = sample['ref'], sample['target']
            if self.name_depth == "unimatch":
                intrinsics = intrinsics.clone()
                intrinsics[0, 0, 2] /= self.preprocessing_depth.ori_size[1] / img_ref.shape[-1]
                intrinsics[0, 1, 2] /= self.preprocessing_depth.ori_size[0] / img_ref.shape[-2]
                intrinsics[0, 0, 0] /= self.preprocessing_depth.ori_size[1] / img_ref.shape[-1]
                intrinsics[0, 1, 0] /= self.preprocessing_depth.ori_size[0] / img_ref.shape[-2]

                res = self.model_depth(Tensor(img_ref), Tensor(img_tgt),
                                                 attn_type=self.args_depth.attn_type,
                                                 attn_splits_list=self.args_depth.attn_splits_list,
                                                 prop_radius_list=self.args_depth.prop_radius_list,
                                                 num_reg_refine=self.args_depth.num_reg_refine,
                                                 intrinsics=intrinsics,
                                                 pose=pose,
                                                 min_depth=1. / self.args_depth.max_depth,
                                                 max_depth=1. / self.args_depth.min_depth,
                                                 num_depth_candidates=self.args_depth.num_depth_candidates,
                                                 pred_bidir_depth=False,
                                                 depth_from_argmax=self.args_depth.depth_from_argmax,
                                                 task='depth')['flow_preds'][-1]  # [1, H, W]
            elif self.name_depth == "KenBurnDepth":
                res = self.model_depth(Tensor(img_ref), Tensor(img_tgt),
                                                 focal=focal, baseline=torch.abs(pose[0, 0, -1]),
                                                 pred_bidir=self.pred_bidir)
            else:
                warnings.warn('This Network is not implemented')
                return 0
            if self.pred_bidir:
                ref = DepthTensor(res[0], device=self.device).scale()
                ref.im_name = sample['ref'].im_name
                target = DepthTensor(res[1], device=self.device).scale()
                target.im_name = sample['target'].im_name
                res = {'ref': ref, 'target': target}
            elif self.pred_right:
                target = DepthTensor(res[1], device=self.device).scale()
                target.im_name = sample['target'].im_name
                res = {'target': target}
            else:
                ref = DepthTensor(res[0], device=self.device).scale()
                ref.im_name = sample['ref'].im_name
                res = {'ref': ref}
            self.preprocessing_depth(res, reverse=True)
            return res
