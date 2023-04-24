import argparse
import os
from pathlib import Path

from classes.Image import ImageCustom
import yaml
import torch
import cv2 as cv
from Networks.UniMatch.unimatch.unimatch import UniMatch
from utils import transforms


class ConfigPipe(dict):
    def __init__(self, interface=False):
        super(ConfigPipe, self).__init__()
        if interface is False:
            with open('Config/config.yml', 'r') as file:
                config = yaml.safe_load(file)
        self["save_inputs"] = config["save_inputs"]
        self["print_info"] = config["print_info"]
        self["save_disp"] = config["save_disp"]
        self["save_reg_images"] = config["save_reg_images"]
        self["reset_images_name"] = config["reset_images_name"]
        self["timeit"] = config["timeit"]
        self["always_proj_infrared"] = config["always_proj_infrared"]
        self.config_device(config["device"])
        self.config_dataset(config["dataset"])
        self.config_network(config['network'])
        self.config_preprocessing(config)
        self.config_refinement(config["refinement"])
        self.config_reconstruction(config["reconstruction"])
        self["output_path"], self['name_experiment'] = check_path(config, self)

    def config_device(self, config):
        self["device"] = {}
        if isinstance(config, str):
            if config == "auto":
                self["device"]["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self["device"]["index"] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            elif config == "multiple":
                pass
            elif config == "cpu":
                self["device"]["device"] = torch.device('cpu')
                self["device"]["index"] = 'cpu'

    def config_dataset(self, config):
        self["dataset"] = {}
        self["dataset"]["save_dataset_conf"] = config["save_dataset_conf"]
        if config["path"]["unique"] is not None:
            p = config["path"]["unique"]
        elif os.path.exists(config["path"]["left"]) \
                and os.path.exists(config["path"]["right"]) \
                and os.path.exists(config["path"]["other"]):
            p = Path(config["path"]["left"]).parent.absolute()
        else:
            raise FileNotFoundError("The given Paths do not exist")
        if config["load_dataset_conf_if_available"] and os.path.exists(os.path.join(p, "dataset.yaml")):
            self["dataset"]["dataset_config"] = load_conf(os.path.join(p, "dataset.yaml"))
        elif config["save_dataset_conf"]:
            print("There is no configuration file for this dataset, a new one will be generated "
                  "following the global Config File")
            self["dataset"]["dataset_config"] = None
        else:
            print("The Dataset has got no configuration file, it will be generated following the global Config File")
            self["dataset"]["dataset_config"] = None
        if self["dataset"]["dataset_config"] is None:
            if config["path"]["unique"] is not None:
                if os.path.exists(os.path.join(p, 'left')) and \
                        os.path.exists(os.path.join(p, 'right')) and \
                        os.path.exists(os.path.join(p, 'other')):

                    self["dataset"]["inference_dir_left"] = os.path.join(p, 'left')
                    self["dataset"]["inference_dir_right"] = os.path.join(p, 'right')
                    self["dataset"]["inference_dir_other"] = os.path.join(p, 'other')
                else:
                    print(
                        """If a "unique" path is set, it has to respect the structure :\nPATH --> /left /other /right""")
                    if os.path.exists(config["path"]["left"]) \
                            and os.path.exists(config["path"]["right"]) \
                            and os.path.exists(config["path"]["other"]):
                        print("\nThe 3 specified paths will be used")
                        self["dataset"] = {}
                        self["dataset"]["inference_dir_left"] = config["path"]["left"]
                        self["dataset"]["inference_dir_right"] = config["path"]["right"]
                        self["dataset"]["inference_dir_other"] = config["path"]["other"]
            else:
                self["dataset"] = {}
                self["dataset"]["inference_dir_left"] = config["path"]["left"]
                self["dataset"]["inference_dir_right"] = config["path"]["right"]
                self["dataset"]["inference_dir_other"] = config["path"]["other"]

            self['dataset']["type"] = check_data_type(self["dataset"]["inference_dir_left"],
                                                      self["dataset"]["inference_dir_right"],
                                                      self["dataset"]["inference_dir_other"],
                                                      autorize_unitype=False)
            self['dataset']["position_setup"] = (config["pos"]["right"] - config["pos"]["left"],
                                                 config["pos"]["other"] - config["pos"]["left"])
            if (config["pos"]["left"] <= config["pos"]["other"] <= config["pos"]["right"]) and config["use_bidir_disp"]:
                self['dataset']["pred_bidir_disp"] = True
            else:
                self['dataset']["pred_bidir_disp"] = False
            if self["always_proj_infrared"] and self['dataset']["type"] == '2ir':
                self['dataset']["pred_bidir_disp"] = False
            if abs(config["pos"]["other"] - config["pos"]["left"]) > abs(
                    config["pos"]["other"] - config["pos"]["right"]):
                self['dataset']["proj_right"] = True
            else:
                self['dataset']["proj_right"] = False
            if not self['dataset']["pred_bidir_disp"] and self['dataset']["proj_right"]:
                self['dataset']["pred_right_disp"] = True
            else:
                self['dataset']["pred_right_disp"] = False
            self['dataset']["position_setup"] = [config["pos"]["right"] - config["pos"]["left"],
                                                 config["pos"]["other"] - config["pos"]["left"]]
            assert self['dataset']["position_setup"][0] > 0, "The Stereo Camera must have an offset"

        else:
            self["dataset"]["inference_dir_left"] = None
            self["dataset"]["inference_dir_right"] = None
            self["dataset"]["inference_dir_other"] = None
            self['dataset']["type"] = self["dataset"]["dataset_config"]["type"]
            self['dataset']["position_setup"] = self["dataset"]["dataset_config"]["position_setup"]
            self['dataset']["pred_bidir_disp"] = self["dataset"]["dataset_config"]["pred_bidir_disp"]
            self['dataset']["proj_right"] = self["dataset"]["dataset_config"]["proj_right"]
            self['dataset']["pred_right_disp"] = self["dataset"]["dataset_config"]["pred_right_disp"]

    def config_network(self, config):
        self["network"] = {}
        self["network"]["name"] = config["name"]
        self["network"]["path_checkpoint"] = config["path_checkpoint"]
        if self["network"]["name"] == "unimatch":
            from Networks.UniMatch.main_stereo import get_args_parser
            parser = get_args_parser()
            args = configure_parser(parser,
                                    path_config='Networks/UniMatch/config_unimatch.yml',
                                    dict_vars=self["dataset"])
            self["network"]["network_args"] = args
        elif self["network"]["name"] == "acvNet":
            pass

    def config_preprocessing(self, config):
        transform = []
        if config['network']['name'] == 'unimatch':
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform = [transforms.ToTensor(),
                         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                         transforms.Resize(self["network"]["network_args"].inference_size,
                                           self["network"]["network_args"].padding_factor)]
        elif config['network']['name'] == 'avcNet':
            pass
        else:
            if isinstance(config["preprocessing"]["normalize"], tuple or list):
                m, std = config["preprocessing"]["normalize"]
                transform.append(transforms.Normalize(mean=m, std=std))
            elif config["preprocessing"]["normalize"]:
                IMAGENET_MEAN = [0.485, 0.456, 0.406]
                IMAGENET_STD = [0.229, 0.224, 0.225]
                transform.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transform.append(transforms.DispSide(self['dataset']["pred_right_disp"],
                                             self['dataset']["pred_bidir_disp"]))
        self["transforms"] = transforms.Compose(transform, self["device"]["device"])
        self["shuffle"] = config["preprocessing"]["shuffle"]

    def config_refinement(self, config):
        pass

    def config_reconstruction(self, config):
        self["reconstruction"] = {}
        self["reconstruction"]["method"] = config["method"]
        if self["device"]["device"] == 'cpu' and config["method"] == "pytorch":
            print(f'The disparity is computed on CPU, The pytorch method is only available on GPU. '
                  f'The opencv method is set')
            self["reconstruction"]["method"] = "fullOpenCv"
        if self["reconstruction"]["method"] == "fullOpenCv":
            self["reconstruction"]["interpolation"] = config["opencv_options"]["interpolation"]
            self["reconstruction"]["border"] = config["opencv_options"]["border"]
        if self["reconstruction"]["method"] == "algo":
            self["reconstruction"]["inpainting"] = config["algo_options"]["inpainting"]


def configure_parser(parser, path_config=None, dict_vars=None):
    dict_pars = vars(parser.parse_args())
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


def check_path(config, Config_Pipe):
    path = os.getcwd() + config["output_path"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True, mode=755)
    name_exp = config["name_experiment"]
    if name_exp:
        path_exp = os.path.join(path, name_exp)
        if not os.path.exists(path_exp):
            os.makedirs(path_exp, exist_ok=True)
            print(f'Directory created for results at : {os.path.abspath(path_exp)}')
        elif os.listdir(path_exp):
            resp = input(f'The specified output path ({path_exp}) is not empty, do we clear the data? (y/n)')
            if resp == "y" or resp == "Y":
                from utils.misc import clear_folder
                clear_folder(path_exp)
            else:
                from utils.misc import update_name
                path_exp = update_name(path_exp)
                os.makedirs(path_exp, exist_ok=True)
                print(f'Directory created for results at : {os.path.abspath(path_exp)}')
    else:
        from datetime import datetime
        now = datetime.now()
        name_exp = now.strftime("%m_%d_%Y_%H:%M:%S")
        path_exp = os.path.join(path, name_exp)
        os.makedirs(path_exp, exist_ok=True)
        print(f'Directory created for results at : {os.path.abspath(path_exp)}')

    if config["save_inputs"]:
        path_input = os.path.join(path_exp, "input/left")
        os.makedirs(path_input, exist_ok=True)
        path_input = os.path.join(path_exp, "input/right")
        os.makedirs(path_input, exist_ok=True)
        path_input = os.path.join(path_exp, "input/other")
        os.makedirs(path_input, exist_ok=True)
    if config["save_disp"]:
        if Config_Pipe['dataset']["pred_right_disp"]:
            path_disp = os.path.join(path_exp, "disp_right")
            os.makedirs(path_disp, exist_ok=True)
        elif Config_Pipe['dataset']["pred_bidir_disp"]:
            path_disp = os.path.join(path_exp, "disp_right")
            os.makedirs(path_disp, exist_ok=True)
            path_disp = os.path.join(path_exp, "disp_left")
            os.makedirs(path_disp, exist_ok=True)
        else:
            path_disp = os.path.join(path_exp, "disp_left")
            os.makedirs(path_disp, exist_ok=True)
        path_disp = os.path.join(path_exp, "disp_other")
        os.makedirs(path_disp, exist_ok=True)
    if config["save_reg_images"]:
        path_disp = os.path.join(path_exp, "reg_images")
        os.makedirs(path_disp, exist_ok=True)

    return path_exp, name_exp


def check_data_type(path_left, path_right, path_other, autorize_unitype=False):
    left_sample = os.listdir(path_left)[0]
    right_sample = os.listdir(path_right)[0]
    other_sample = os.listdir(path_other)[0]
    left = ImageCustom(os.path.join(path_left, left_sample))
    right = ImageCustom(os.path.join(path_right, right_sample))
    other = ImageCustom(os.path.join(path_other, other_sample))
    if len(left.shape) == 3 and len(right.shape) == len(left.shape):
        if len(other.shape) == 2:
            return "2vis"
        elif len(other.shape) == 3 and not autorize_unitype:
            raise TypeError("All the images are colored images")
    elif len(right.shape) == len(left.shape) and len(left.shape) == 2:
        if len(other.shape) == 3:
            return "2ir"
        elif len(other.shape) == 2 and not autorize_unitype:
            raise TypeError("All the images are gray images")

    # if __name__ == '__main__':
#     config = Config_pipe()
