import argparse
import os
from pathlib import Path

from utils.classes.Image import ImageCustom
import yaml
import torch
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
        self["save_calibration"] = config["save_calibration"]

        self.config_device(config["device"])
        self.config_dataset(config["dataset"])
        self.config_network(config["network"])
        self.config_preprocessing(config)
        self.config_refinement(config["refinement"])
        self.config_reconstruction(config["reconstruction"])
        self.config_validation(config["validation"])
        self.config_pointCLoud(config['pointsCloud'])
        self.config_cameras(config['cameras'])
        self["output_path"] = config["output_path"]
        self["name_experiment"] = config["name_experiment"]
        self["output_path"], self["name_experiment"] = check_path(self)

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
        # Creation of the dataset dictionary
        self["dataset"] = {}
        self["dataset"]["shuffle"] = config["shuffle"]
        self["dataset"]["number_of_sample"] = config["number_of_sample"]
        self["dataset"]["save_dataset_conf"] = config["save_dataset_conf"]
        self["dataset"]["save_file_list_in_conf"] = config["save_file_list_in_conf"]
        # Check if the given paths exist
        if config["path"]["unique"] is not None:
            p = config["path"]["unique"]
        elif os.path.exists(config["path"]["left"]) \
                and os.path.exists(config["path"]["right"]) \
                and os.path.exists(config["path"]["other"]):
            p = Path(config["path"]["left"]).parent.absolute()
        else:
            raise FileNotFoundError("The given Paths do not exist")
        # Load a config file for a Dataset is such file exist and the option allows it
        if config["load_dataset_conf_if_available"] and os.path.exists(os.path.join(p, "dataset.yaml")):
            self["dataset"]["dataset_config"] = load_conf(os.path.join(p, "dataset.yaml"))
            self["dataset"]['load_dataset_conf_if_available'] = True
            self["dataset"]["save_dataset_conf"] = False
        elif config["save_dataset_conf"]:
            print("There is no configuration file for this dataset, a new one will be generated "
                  "following the global Config File")
            self["dataset"]["dataset_config"] = None
            self["dataset"]['load_dataset_conf_if_available'] = False
        else:
            print("The Dataset has got no configuration file, it will be generated following the global Config File")
            self["dataset"]["dataset_config"] = None
            self["dataset"]['load_dataset_conf_if_available'] = False
            # If no config file has been load, the following part will continue normally
        if self["dataset"]["dataset_config"] is None:
            self['dataset']['alignment_isDone'] = config["alignment_isDone"]
            self['dataset']['alignment_auto'] = config["alignment_auto"]
            # Definition of the 3 paths toward the Stereo Pair images, and the other modality
            if config["path"]["unique"] is not None:
                if os.path.exists(os.path.join(p, 'left')) and \
                        os.path.exists(os.path.join(p, 'right')) and \
                        os.path.exists(os.path.join(p, 'other')):

                    self["dataset"]["inference_dir_left"] = os.path.join(p, 'left')
                    self["dataset"]["inference_dir_right"] = os.path.join(p, 'right')
                    self["dataset"]["inference_dir_other"] = os.path.join(p, 'other')
                else:
                    print("""If a "unique" path is set, it has to respect the structure :
                    \nPATH --> /left /other /right""")
                    if os.path.exists(config["path"]["left"]) \
                            and os.path.exists(config["path"]["right"]) \
                            and os.path.exists(config["path"]["other"]):
                        print("\nThe 3 specified paths will be used")
                        self["dataset"]["inference_dir_left"] = config["path"]["left"]
                        self["dataset"]["inference_dir_right"] = config["path"]["right"]
                        self["dataset"]["inference_dir_other"] = config["path"]["other"]
            else:
                self["dataset"]["inference_dir_left"] = config["path"]["left"]
                self["dataset"]["inference_dir_right"] = config["path"]["right"]
                self["dataset"]["inference_dir_other"] = config["path"]["other"]

            # Automatic detection of the Data type : stereo infrared + visible or stereo visible + infrared
            self['dataset']["type"] = check_data_type(self["dataset"]["inference_dir_left"],
                                                      self["dataset"]["inference_dir_right"],
                                                      self["dataset"]["inference_dir_other"],
                                                      autorize_unitype=False)
            # Position setup : difference "other-left" and "right-left" is computed if use_pos is True
            self['dataset']["position_setup"] = [config["pos"]["right"] - config["pos"]["left"],
                                                 config["pos"]["other"] - config["pos"]["left"]]
            self['dataset']['use_pos'] = config["use_pos"]
            self['dataset']['alignment_isDone'] = config['alignment_isDone']
            self['dataset']['alignment_auto'] = config["alignment_auto"]
            self['dataset']["use_bidir_disp"] = config["use_bidir_disp"]
            self.configure_projection_option()
        else:
            self["dataset"]["inference_dir_left"] = self["dataset"]["dataset_config"]["1. Paths"]["inference_dir_left"]
            self["dataset"]["inference_dir_right"] = self["dataset"]["dataset_config"]["1. Paths"]["inference_dir_right"]
            self["dataset"]["inference_dir_other"] = self["dataset"]["dataset_config"]["1. Paths"]["inference_dir_other"]
            self['dataset']["type"] = self["dataset"]["dataset_config"]["0. Generalities"]["type"]
            self['dataset']["position_setup"] = self["dataset"]["dataset_config"]["3. Position and alignment"]["position_setup"]
            self['dataset']["pred_bidir_disp"] = self["dataset"]["dataset_config"]["2. disparity configuration"]["pred_bidir_disp"]
            self['dataset']["proj_right"] = self["dataset"]["dataset_config"]["2. disparity configuration"]["proj_right"]
            self['dataset']["pred_right_disp"] = self["dataset"]["dataset_config"]["2. disparity configuration"]["pred_right_disp"]
            self['dataset']['alignment_isDone'] = self["dataset"]["dataset_config"]["3. Position and alignment"]["alignment_isDone"]
            self['dataset']['alignment_auto'] = self["dataset"]["dataset_config"]["3. Position and alignment"]["alignment_auto"]
            self['dataset']['use_pos'] = True
        self['dataset']['projection_process'] = define_target_ref(self['dataset'])

    def config_network(self, config):
        self["network"] = {}
        self["network"]["name"] = config["name"]
        if self["network"]["name"] == "unimatch":
            from Networks.UniMatch.parser import get_args_parser
            parser = get_args_parser()
            args = configure_parser(parser,
                                    path_config='Networks/UniMatch/config_unimatch.yml',
                                    dict_vars=self["dataset"])
            self["network"]["network_args"] = args
        elif self["network"]["name"] == "acvNet":
            from Networks.ACVNet.parser import get_args_parser
            parser = get_args_parser()
            args = configure_parser(parser,
                                    path_config='Networks/ACVNet/config_acvNet.yml',
                                    dict_vars=self["dataset"])
            self["network"]["network_args"] = args
        else:
            pass
        if args.path_checkpoint:
            self["network"]["path_checkpoint"] = args.path_checkpoint
        else:
            self["network"]["path_checkpoint"] = config["path_checkpoint"]

    def config_preprocessing(self, config):
        transform = []
        if config['network']['name'] == 'unimatch':
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            transform = [transforms.ToTensor(),
                         transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                         transforms.Resize(self["network"]["network_args"].inference_size,
                                           self["network"]["network_args"].padding_factor)]
        elif config['network']['name'] == 'acvNet':
            transform = [transforms.ToFloatTensor(),
                         transforms.Pad(self["network"]["network_args"].inference_size, keep_ratio=True)]
        else:
            if isinstance(config["preprocessing"]["normalize"], tuple or list):
                m, std = config["preprocessing"]["normalize"]
                transform.append(transforms.Normalize(mean=m, std=std))
            elif config["preprocessing"]["normalize"]:
                IMAGENET_MEAN = [0.485, 0.456, 0.406]
                IMAGENET_STD = [0.229, 0.224, 0.225]
                transform.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self["transform"] = transform #s.Compose(transform, self["device"]["device"])

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

    def config_validation(self, config):
        self['validation'] = {}
        self['validation']['indices'] = {'rmse': config['indices']['rmse'],
                                         'nmi': config['indices']['nmi']
                                         if self["reconstruction"]["method"] != "pytorch" else False,
                                         'psnr': config['indices']['psnr'],
                                         'ssim': config['indices']['ssim'],
                                         'ms_ssim': config['indices']["ms_ssim"]
                                         if self["reconstruction"]["method"] == "pytorch" else False,
                                         'nec': config['indices']['nec']}
        self['validation']['stats'] = {'mean': config['stats']['mean'],
                                       'std': config['stats']['std']}
        self['validation']['method'] = self["reconstruction"]["method"]
        self['validation']['activated'] = config['activated']
        self['validation']['compare_smaller'] = config['compare_smaller']

    def config_pointCLoud(self, config):
        self['pointsCloud'] = {}
        self['pointsCloud']['activated'] = config['activated']
        self['pointsCloud']['disparity'] = config['disparity']
        self['pointsCloud']['visualisation'] = config['visualisation']
        self['pointsCloud']['save'] = config['save']
        self['pointsCloud']['use_bidir'] = config['use_bidir']
        if config['min_disparity'] > 1:
            self['pointsCloud']['min_disparity'] = config['min_disparity']/100
        else:
            self['pointsCloud']['min_disparity'] = config['min_disparity']
        if config['both']:
            self['pointsCloud']['mode'] = 'both'
        elif config['stereo']:
            self['pointsCloud']['mode'] = 'stereo'
        else:
            self['pointsCloud']['mode'] = 'other'

    def config_cameras(self, config):
        self['cameras'] = {}
        if config['left']:
            self['cameras']['left'] = load_conf(config['left'])
        else:
            self['cameras']['left'] = None
        if config['right']:
            self['cameras']['right'] = load_conf(config['right'])
        else:
            self['cameras']['right'] = None
        if config['other']:
            self['cameras']['other'] = load_conf(config['other'])
        else:
            self['cameras']['other'] = None

    def configure_projection_option(self):
        if self['dataset']['use_pos']:
            # Configuration of the option of projection depending on the given position of the camera
            if (0 <= self['dataset']["position_setup"][1] <= self['dataset']["position_setup"][0]) and self['dataset']["use_bidir_disp"]:
                self['dataset']["pred_bidir_disp"] = True
            else:
                self['dataset']["pred_bidir_disp"] = False
            if self['dataset']["type"] == '2ir':
                self['dataset']["pred_bidir_disp"] = False
            if abs(self['dataset']["position_setup"][1]) > \
                    abs(abs(self['dataset']["position_setup"][0]) - self['dataset']["position_setup"][1]):
                self['dataset']["proj_right"] = True
            else:
                self['dataset']["proj_right"] = False
            if not self['dataset']["pred_bidir_disp"] and self['dataset']["proj_right"]:
                self['dataset']["pred_right_disp"] = True
            else:
                self['dataset']["pred_right_disp"] = False
            assert self['dataset']["position_setup"][0] > 0, "The Stereo Camera must have an offset"

        else:
            self['dataset']["position_setup"] = None
            self['dataset']["pred_bidir_disp"] = False
            self['dataset']["proj_right"] = False
            self['dataset']["pred_right_disp"] = False


def define_target_ref(config):
    case_dict = {}
    if not config["use_pos"]:
        case_dict['step1'] = 98
        case_dict['step2'] = 99
    else:
        case_dict['step1'] = 1 if config['proj_right'] else 0
        if config["type"] == '2vis':
            if config["position_setup"][1] < 0:
                case_dict['step2'] = 10
            elif config["position_setup"][1] < config["position_setup"][0]:
                case_dict['step2'] = 14 if config["proj_right"] else 13
            elif config["position_setup"][1] > config["position_setup"][0]:
                case_dict['step2'] = 17
            else:
                case_dict['step2'] = 99
        if config["type"] == '2ir':
            if config["position_setup"][1] < 0:
                case_dict['step2'] = 11
            elif config["position_setup"][1] < config["position_setup"][0]:
                case_dict['step2'] = 15 if config["proj_right"] else 12
            elif config["position_setup"][1] > config["position_setup"][0]:
                case_dict['step2'] = 16
            else:
                case_dict['step2'] = 99
    return case_dict


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


def check_path(configpipe):
    path = os.getcwd() + configpipe["output_path"]
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o777)
    name_exp = configpipe["name_experiment"]
    if name_exp:
        path_exp = os.path.join(path, name_exp)
        if not os.path.exists(path_exp):
            os.makedirs(path_exp, exist_ok=True)
            os.chmod(path_exp, 0o777)
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
                os.chmod(path_exp, 0o777)
                print(f'Directory created for results at : {os.path.abspath(path_exp)}')
    else:
        from datetime import datetime
        now = datetime.now()
        name_exp = now.strftime("%m_%d_%Y_%H:%M:%S")
        path_exp = os.path.join(path, name_exp)
        os.makedirs(path_exp, exist_ok=True)
        print(f'Directory created for results at : {os.path.abspath(path_exp)}')

    configpipe["output_path"] = path_exp
    configpipe["name_experiment"] = name_exp
    if configpipe["save_inputs"]:
        create_dir_input(path_exp)

    if configpipe["save_disp"]:
        create_dir_disp(configpipe)

    if configpipe["save_reg_images"]:
        path_disp = os.path.join(path_exp, "reg_images")
        os.makedirs(path_disp, exist_ok=True)

    if configpipe['pointsCloud']['activated']:
        path_cloud = os.path.join(path_exp, "pointsCloud")
        os.makedirs(path_cloud, exist_ok=True)

    return os.path.realpath(path_exp), name_exp


def create_dir_input(path_exp):
    path_input = os.path.join(path_exp, "input/left")
    os.makedirs(path_input, exist_ok=True)
    path_input = os.path.join(path_exp, "input/right")
    os.makedirs(path_input, exist_ok=True)
    path_input = os.path.join(path_exp, "input/other")
    os.makedirs(path_input, exist_ok=True)


def create_dir_disp(config):
    if config['dataset']["pred_right_disp"]:
        path_disp = os.path.join(config["output_path"], "disp_right")
        os.makedirs(path_disp, exist_ok=True)
    elif config['dataset']["pred_bidir_disp"]:
        path_disp = os.path.join(config["output_path"], "disp_right")
        os.makedirs(path_disp, exist_ok=True)
        path_disp = os.path.join(config["output_path"], "disp_left")
        os.makedirs(path_disp, exist_ok=True)
    else:
        path_disp = os.path.join(config["output_path"], "disp_left")
        os.makedirs(path_disp, exist_ok=True)
    path_disp = os.path.join(config["output_path"], "disp_other")
    os.makedirs(path_disp, exist_ok=True)


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
