import argparse
import os
from glob import glob
from pathlib import Path

from kornia.utils import get_cuda_device_if_available

from Networks.UniMatch.parser import get_args_parser_depth
import yaml
import torch
from utils import transforms as transforms
from utils.misc import update


class ConfigPipe(dict):
    def __init__(self, config_file: dict = None):
        super(ConfigPipe, self).__init__()
        with open('config/config.yml', 'r') as file:
            config = yaml.safe_load(file)
        if config_file is not None:
            update(config, config_file)
        self["save_inputs"] = config["save_inputs"]
        self["print_info"] = config["print_info"]
        self["save_disp"] = config["save_disp"]
        self["save_reg_images"] = config["save_reg_images"]
        self["reset_images_name"] = config["reset_images_name"]
        self["timeit"] = config["timeit"]

        self.config_device(config["device"])
        self.config_dataset(config["dataset"])
        self.config_setup(config["setup"])
        self.config_disparity_network(config["disparity_network"])
        self.config_depth_network(config["depth_network"])
        # self.config_refinement(config["refinement"])
        self.config_reconstruction(config["reconstruction"])
        self.config_validation(config["validation"])
        # self.config_pointCLoud(config['pointsCloud'])
        self["output_path"] = f'{os.getcwd()}/{config["output_path"]}/{config["name_experiment"]}'
        self["name_experiment"] = config["name_experiment"]
        # self["output_path"], self["name_experiment"] = check_path(self)

    def config_device(self, config):
        self["device"] = {}
        if isinstance(config, str):
            if config == "auto":
                self["device"]["device"] = get_cuda_device_if_available()
                self["device"]["index"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            elif config == "multiple":
                pass
            elif config == "cpu":
                self["device"]["device"] = torch.device('cpu')
                self["device"]["index"] = 'cpu'

    def config_dataset(self, config):
        # Creation of the dataset dictionary
        self["dataset"] = {}
        self["dataset"]["shuffle"] = config["shuffle"]
        if config["indexes"]:
            self["dataset"]["number_of_sample"] = len(config["indexes"])
            self["dataset"]["indexes"] = config["indexes"]
        else:
            self["dataset"]["number_of_sample"] = config["number_of_sample"]
            self["dataset"]["indexes"] = None
        # self["dataset"]["save_file_list_in_conf"] = config["save_file_list_in_conf"]

    def config_setup(self, config):
        self["setup"] = {}

        if os.path.isdir(config['path']):
            self["setup"]['path'] = sorted(glob(config['path'] + '/*.yaml'))
            self["setup"]['multi'] = True
        else:
            self["setup"]['path'] = config['path']
            self["setup"]['multi'] = False

    def config_disparity_network(self, config):
        self["disparity_network"] = {}
        self['disparity_network']["use_bidir_disp"] = config["use_bidir_disp"]
        self["disparity_network"]["name"] = config["name"]
        if self["disparity_network"]["name"].upper() == "UNIMATCH":
            from Networks.UniMatch.parser import get_args_parser_disparity
            parser = get_args_parser_disparity()
            args = configure_parser(parser,
                                    config["preprocessing"],
                                    path_config='Networks/UniMatch/config_unimatch_disparity.yml',
                                    dict_vars=self["dataset"])
            self["disparity_network"]["network_args"] = args
        elif self["disparity_network"]["name"].upper() == "ACVNET":
            from Networks.ACVNet.parser import get_args_parser
            parser = get_args_parser()
            args = configure_parser(parser,
                                    config["preprocessing"],
                                    path_config='Networks/ACVNet/config_acvNet.yml',
                                    dict_vars=self["dataset"])
            self["disparity_network"]["network_args"] = args
        else:
            pass
        if args.path_checkpoint:
            self["disparity_network"]["path_checkpoint"] = args.path_checkpoint
        else:
            self["disparity_network"]["path_checkpoint"] = config["path_checkpoint"]
        self["disparity_network"]["preprocessing"] = self.config_preprocessing(config)

    def config_depth_network(self, config):
        self["depth_network"] = {}
        self['depth_network']["use_bidir_depth"] = config["use_bidir_depth"]
        self["depth_network"]["name"] = config["name"]
        if self["depth_network"]["name"].upper() == "UNIMATCH":
            from Networks.UniMatch.parser import get_args_parser_disparity
            parser = get_args_parser_depth()
            args = configure_parser(parser,
                                    config["preprocessing"],
                                    path_config='Networks/UniMatch/config_unimatch_depth.yml',
                                    dict_vars=self["dataset"])
            self["depth_network"]["network_args"] = args
        elif self["depth_network"]["name"].upper() == "KENBURN":
            args = configure_parser(None,
                                    config["preprocessing"],
                                    path_config='Networks/KenburnDepth/config_Kenburn.yml',
                                    dict_vars=self["dataset"])
            self["depth_network"]["network_args"] = args
        else:
            pass
        if args.path_checkpoint:
            self["depth_network"]["path_checkpoint"] = args.path_checkpoint
        else:
            self["depth_network"]["path_checkpoint"] = config["path_checkpoint"]
        self["depth_network"]["preprocessing"] = self.config_preprocessing(config, target="depth_network")

    def config_preprocessing(self, config, target='disparity_network'):
        transform = []
        if config['name'].upper() == 'UNIMATCH':
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform = [transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                         transforms.Resize(self[target]["network_args"].inference_size,
                                           self[target]["network_args"].padding_factor)]
        elif config['name'].upper() == 'ACVNET':
            transform = [transforms.ToFloatTensor(),
                         transforms.Pad(self[target]["network_args"].inference_size, keep_ratio=True)]
        elif config['name'].upper() == 'KENBURN':
            transform = [transforms.Resize(self[target]["network_args"].inference_size, 0)]
        # else:
        #     if isinstance(config["preprocessing"]["normalize"], tuple or list):
        #         m, std = config["preprocessing"]["normalize"]
        #         transform.append(transforms.Normalize(mean=m, std=std))
        #     elif config["preprocessing"]["normalize"]:
        #         IMAGENET_MEAN = [0.485, 0.456, 0.406]
        #         IMAGENET_STD = [0.229, 0.224, 0.225]
        #         transform.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        return transform

    def config_refinement(self, config):
        pass

    def config_reconstruction(self, config):
        self["reconstruction"] = {}
        self["reconstruction"]["remove_occlusion"] = config["remove_occlusion"]
        self["reconstruction"]["post_process_image"] = config["post_process_image"]
        self["reconstruction"]["post_process_depth"] = config["post_process_depth"]

    def config_validation(self, config):
        self['validation'] = {}
        self['validation']['indices'] = {'rmse': config['indices']['rmse'],
                                         'psnr': config['indices']['psnr'],
                                         'ssim': config['indices']['ssim'],
                                         'ms_ssim': config['indices']["ms_ssim"],
                                         'nec': config['indices']['nec']}
        self['validation']['stats'] = {'mean': config['stats']['mean'],
                                       'std': config['stats']['std']}
        self['validation']['activated'] = config['activated']


def configure_parser(parser, config, path_config=None, dict_vars=None):
    dict_pars = vars(parser.parse_args()) if parser is not None else {}
    config_vars = {}
    if path_config:
        if isinstance(path_config, str or os.path):
            with open(path_config, 'r') as file:
                config_vars = yaml.safe_load(file)
        else:
            raise TypeError("A path or a String is expected for the config file")
    if not dict_vars:
        dict_vars = {}
    config_vars = config_vars | dict_vars
    config_vars = config_vars | config
    for key, value in config_vars.items():
        try:
            dict_pars[key] = value
        except KeyError:
            print(f"The Key {key} in the config file doesn't exist in this parser")
    return argparse.Namespace(**dict_pars)


def load_conf(path_config):
    with open(path_config, 'r') as file:
        dataset_config = yaml.safe_load(file)
    return dataset_config

#
# def check_path(configpipe):
#     path = os.getcwd() + configpipe["output_path"]
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)
#         os.chmod(path, 0o777)
#     name_exp = configpipe["name_experiment"]
#     if name_exp:
#         path_exp = os.path.join(path, name_exp)
#         if not os.path.exists(path_exp):
#             os.makedirs(path_exp, exist_ok=True)
#             os.chmod(path_exp, 0o777)
#             print(f'Directory created for results at : {os.path.abspath(path_exp)}')
#         elif os.listdir(path_exp):
#             resp = input(f'The specified output path ({path_exp}) is not empty, do we clear the data? (y/n)')
#             if resp == "y" or resp == "Y":
#                 from utils.misc import clear_folder
#                 clear_folder(path_exp)
#             else:
#                 from utils.misc import update_name
#                 path_exp = update_name(path_exp)
#                 os.makedirs(path_exp, exist_ok=True)
#                 os.chmod(path_exp, 0o777)
#                 print(f'Directory created for results at : {os.path.abspath(path_exp)}')
#     else:
#         from datetime import datetime
#         now = datetime.now()
#         name_exp = now.strftime("%m_%d_%Y_%H:%M:%S")
#         path_exp = os.path.join(path, name_exp)
#         os.makedirs(path_exp, exist_ok=True)
#         print(f'Directory created for results at : {os.path.abspath(path_exp)}')
#
#     configpipe["output_path"] = path_exp
#     configpipe["name_experiment"] = name_exp
#     if configpipe["save_inputs"]:
#         create_dir_input(path_exp)
#
#     if configpipe["save_disp"]:
#         create_dir_disp(configpipe)
#
#     if configpipe["save_reg_images"]:
#         path_disp = os.path.join(path_exp, "reg_images")
#         os.makedirs(path_disp, exist_ok=True)
#
#     if configpipe['pointsCloud']['activated']:
#         path_cloud = os.path.join(path_exp, "pointsCloud")
#         os.makedirs(path_cloud, exist_ok=True)
#
#     return os.path.realpath(path_exp), name_exp


# def create_dir_input(path_exp):
#     path_input = os.path.join(path_exp, "input/left")
#     os.makedirs(path_input, exist_ok=True)
#     path_input = os.path.join(path_exp, "input/right")
#     os.makedirs(path_input, exist_ok=True)
#     path_input = os.path.join(path_exp, "input/other")
#     os.makedirs(path_input, exist_ok=True)
#
#
# def create_dir_disp(config):
#     if config['dataset']["pred_right_disp"]:
#         path_disp = os.path.join(config["output_path"], "disp_right")
#         os.makedirs(path_disp, exist_ok=True)
#     elif config['dataset']["pred_bidir_disp"]:
#         path_disp = os.path.join(config["output_path"], "disp_right")
#         os.makedirs(path_disp, exist_ok=True)
#         path_disp = os.path.join(config["output_path"], "disp_left")
#         os.makedirs(path_disp, exist_ok=True)
#     else:
#         path_disp = os.path.join(config["output_path"], "disp_left")
#         os.makedirs(path_disp, exist_ok=True)
#     path_disp = os.path.join(config["output_path"], "disp_other")
#     os.makedirs(path_disp, exist_ok=True)

    # if __name__ == '__main__':
#     config = Config_pipe()



#     def configure_projection_option(self):
#         if self['dataset']['use_pos']:
#             # Configuration of the option of projection depending on the given position of the camera
#             if (0 <= self['dataset']["position_setup"][1] <= self['dataset']["position_setup"][0]) and self['dataset']["use_bidir_disp"]:
#                 self['dataset']["pred_bidir_disp"] = True
#             else:
#                 self['dataset']["pred_bidir_disp"] = False
#             if self['dataset']["type"] == '2ir':
#                 self['dataset']["pred_bidir_disp"] = False
#             if abs(self['dataset']["position_setup"][1]) > \
#                     abs(abs(self['dataset']["position_setup"][0]) - self['dataset']["position_setup"][1]):
#                 self['dataset']["proj_right"] = True
#             else:
#                 self['dataset']["proj_right"] = False
#             if not self['dataset']["pred_bidir_disp"] and self['dataset']["proj_right"]:
#                 self['dataset']["pred_right_disp"] = True
#             else:
#                 self['dataset']["pred_right_disp"] = False
#             assert self['dataset']["position_setup"][0] > 0, "The Stereo Camera must have an offset"
#
#         else:
#             self['dataset']["position_setup"] = None
#             self['dataset']["pred_bidir_disp"] = False
#             self['dataset']["proj_right"] = False
#             self['dataset']["pred_right_disp"] = False
#
#
# def define_target_ref(config):
#     case_dict = {}
#     if not config["use_pos"]:
#         case_dict['step1'] = 98
#         case_dict['step2'] = 99
#     else:
#         case_dict['step1'] = 1 if config['proj_right'] else 0
#         if config["type"] == '2vis':
#             if config["position_setup"][1] < 0:
#                 case_dict['step2'] = 10
#             elif config["position_setup"][1] < config["position_setup"][0]:
#                 case_dict['step2'] = 14 if config["proj_right"] else 13
#             elif config["position_setup"][1] > config["position_setup"][0]:
#                 case_dict['step2'] = 17
#             else:
#                 case_dict['step2'] = 99
#         if config["type"] == '2ir':
#             if config["position_setup"][1] < 0:
#                 case_dict['step2'] = 11
#             elif config["position_setup"][1] < config["position_setup"][0]:
#                 case_dict['step2'] = 15 if config["proj_right"] else 12
#             elif config["position_setup"][1] > config["position_setup"][0]:
#                 case_dict['step2'] = 16
#             else:
#                 case_dict['step2'] = 99
#     return case_dict

