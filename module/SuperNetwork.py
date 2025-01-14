import os
import sys
import warnings

import torch
from torch import Tensor
from Networks.ACVNet.models import ACVNet
from Networks.Depth_anythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from Networks.KenburnDepth.KenburnDepth import KenburnDepth
from Networks.UniMatch.unimatch.unimatch import UniMatch
from config.Config import ConfigPipe
from module.BaseModule import BaseModule
from module.Preprocessing import Preprocessing
from utils.ImagesCameras.Image.Image import DepthTensor
from utils.misc import timeit, count_parameter


class SuperNetwork(BaseModule):
    Network = {'disparity': ['ACVNet', 'UniMatch'], 'depth': ['UniMatch'],
               'monocular': ['Kenburn', 'DepthAnything', 'DepthAnythingV2']}
    """
    This class add a layer for the data post-processing & the inputs args according each Network implemented.
    To Run it, a normal Forward call with 2 images as inputs would do it.
    """

    def __init__(self, config: ConfigPipe, *args, **kwargs):  # model, args, name: str, timeit=False):
        self.name_disparity = ''
        self.model_disparity = None
        self.model_depth = None
        self.model_monocular = None
        self.name_depth = ''
        self.name_disparity = ''
        self.name_monocular = ''
        self.preprocessing_disparity = None
        self.preprocessing_depth = None
        self.preprocessing_monocular = None
        self.pred_right = False
        self.pred_bidir = False
        super(SuperNetwork, self).__init__(config, *args, **kwargs)

    def _update_conf(self, config, *args, **kwargs):
        self.args_disparity = config['disparity_network']["network_args"]
        self.args_depth = config['depth_network']["network_args"]
        self.args_monocular = config['monocular_network']["network_args"]
        self.device_index = config["device"]["index"]
        self.__class__.__name__ = 'Neural Network'
        self.disparity_network_init(config)
        self.depth_network_init(config)
        self.monocular_network_init(config)
        # self.preprocessing_disparity_init(config)
        # self.preprocessing_depth_init(config)

    def update_pred_right(self, activate: bool = True):
        self.pred_right = activate
        self.update_preprocessing()

    def update_pred_bidir(self, activate: bool = True):
        self.pred_bidir = activate
        self.update_preprocessing()

    def update_size(self, size, task='disparity'):
        if size is not None:
            if task == 'disparity':
                self.preprocessing_disparity.inference_size = size
            elif task == 'depth':
                self.preprocessing_depth.inference_size = size
            elif task == 'monocular':
                self.preprocessing_monocular.inference_size = size

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
            model = ACVNet(self.args_disparity.maxdisp,
                           self.args_disparity.attn_weights_only,
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
            self.model_disparity = torch.nn.DataParallel(self.model_disparity)

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

    def monocular_network_init(self, config):
        # Disparity Network initialization
        self.name_monocular = config['monocular_network']["name"].upper()
        if self.name_monocular == 'KENBURNDEPTH' or self.name_monocular == 'KENBURN' or self.name_monocular == 'KEN':
            self.name_monocular = "KenBurnDepth"
            model = KenburnDepth(path_ckpt=self.config['monocular_network']["path_checkpoint"],
                                 semantic_adjustment=self.config['monocular_network']['network_args'].semantic_adjustment,
                                 semantic_network=self.config['monocular_network']['network_args'].semantic_network,
                                 device=self.device)
        elif self.name_monocular == "DEPTHANYTHING":
            self.name_monocular = "DepthAnything"
            self.args_monocular.pretrained_resource = self.args_monocular.path_checkpoint
            from Networks.Depth_anything.metric_depth.zoedepth.models.builder import build_model
            sys.path.append(os.getcwd() + '/Networks/Depth_anything/metric_depth')
            model = build_model(self.args_monocular).eval()
        elif self.name_monocular == "DEPTHANYTHINGV2":
            self.name_monocular = "DepthAnythingV2"
            checkpoint = self.args_monocular.path_checkpoint
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            path = os.getcwd() + str(
                self.args_monocular.path_checkpoint) + f'/depth_anything_v2_metric_{"hypersim" if self.args_monocular.stage == "indoor" else "vkitti"}_{self.args_monocular.encoder}.pth'
            model = DepthAnythingV2(
                **{**model_configs[self.args_monocular.encoder], 'max_depth': self.args_monocular.max_depth})
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model = model.eval()
        elif config['depth_network']["name"] == "to_be_implemented":
            self.name_monocular = "custom"
            model = None
            pass
        else:
            model = None
        self.model_monocular = model.to(device=self.device)
        self.preprocessing_monocular = Preprocessing(config["monocular_network"]["preprocessing"], self.device,
                                                     task='monocular')

    def __str__(self):
        string = super().__str__()
        string += f'The model "{self.name_disparity}" has been initialized for disparity'
        string += f"\nLoad checkpoint: {self.config['disparity_network']['path_checkpoint']}\n"
        string += count_parameter(self.model_disparity)
        string += f'\nThe model "{self.name_depth}" has been initialized for depth'
        string += f"\nLoad checkpoint: {self.config['depth_network']['path_checkpoint']}\n"
        string += count_parameter(self.model_depth)
        string += f'\nThe model "{self.name_monocular}" has been initialized for monocular depth'
        string += f"\nLoad checkpoint: {self.config['monocular_network']['path_checkpoint']}\n"
        string += count_parameter(self.model_monocular)
        return string

    @torch.no_grad()
    @timeit
    def __call__(self, sample, *args, task='depth', intrinsics=None, pose=None, focal=0, **kwargs):
        if task == 'disparity':
            sample = self.preprocessing_disparity(sample.copy())
            im_left, im_right = sample['left'], sample['right']
            if self.name_disparity == "unimatch":
                if im_left.modality == 'Any':
                    im_left = im_left.RGB('gray')
                if im_right.modality == 'Any':
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
                left = DepthTensor(res[0], device=self.device)
                left.name = im_left.name
                right = DepthTensor(res[1], device=self.device)
                right.name = im_right.name
                res = {'left': left, 'right': right}
            elif self.pred_right:
                right = DepthTensor(res[0], device=self.device)
                right.name = im_right.name
                res = {'right': right}
            else:
                left = DepthTensor(res[0], device=self.device)
                left.name = im_left.name
                res = {'left': left}
            res = self.preprocessing_disparity(res, reverse=True)
            return res
        elif task == 'depth':
            sample = self.preprocessing_depth(sample)
            img_ref, img_tgt = sample['ref'], sample['target']
            if self.name_depth == "unimatch":
                intrinsics = intrinsics.clone()
                intrinsics[0, 0, 2] /= self.preprocessing_depth.ori_size[0][1] / img_ref.shape[-1]
                intrinsics[0, 1, 2] /= self.preprocessing_depth.ori_size[0][0] / img_ref.shape[-2]
                intrinsics[0, 0, 0] /= self.preprocessing_depth.ori_size[0][1] / img_ref.shape[-1]
                intrinsics[0, 1, 0] /= self.preprocessing_depth.ori_size[0][0] / img_ref.shape[-2]

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
            # elif self.name_depth == "KenBurnDepth":
            #     res = self.model_depth(Tensor(img_ref), Tensor(img_tgt),
            #                            focal=focal, baseline=torch.abs(pose[0, 0, -1]),
            #                            pred_bidir=self.pred_bidir)
            else:
                warnings.warn('This Network is not implemented')
                return 0
            if self.pred_bidir:
                ref = DepthTensor(res[0], device=self.device)
                ref.name = sample['ref'].name
                target = DepthTensor(res[1], device=self.device)
                target.name = sample['target'].name
                res = {'ref': ref, 'target': target}
            elif self.pred_right:
                target = DepthTensor(res[1], device=self.device)
                target.name = sample['target'].name
                res = {'target': target}
            else:
                ref = DepthTensor(res[0], device=self.device)
                ref.name = sample['ref'].name
                res = {'ref': ref}
            self.preprocessing_depth(res, reverse=True)
            return res
        else:
            sample = self.preprocessing_monocular(sample)
            if self.name_monocular == "KenBurnDepth":
                res = self.model_monocular(sample, focal=focal, intrinsics=intrinsics)
            elif self.name_monocular == "DepthAnything":
                for key, im in sample.items():
                    res = {}
                    res[key] = self.model_monocular(im, focal=focal)['metric_depth']
            elif self.name_monocular == "DepthAnythingV2":
                for key, im in sample.items():
                    res = {}
                    res[key] = self.model_monocular(im)
            else:
                warnings.warn('This Network is not implemented')
                return 0
            for cam, im in res.items():
                res[cam] = DepthTensor(im, device=self.device)
                res[cam].name = sample[cam].name
            self.preprocessing_monocular(res, reverse=True)
            return res
