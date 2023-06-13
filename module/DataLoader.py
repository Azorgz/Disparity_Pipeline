import os
import random
from glob import glob
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml
from torch.utils.data import Dataset
from module import Projection_process
from utils.classes.Image import ImageCustom
from utils.misc import timeit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataLoader(Dataset):
    """
    This dataloader loads the files according to the options set.
    With an input stereo + mono, specify the geometry on the camera disposition
    """

    def __init__(self, config):
        super(StereoDataLoader, self).__init__()
        self.__update_conf__(config)
        if config["timeit"]:
            self.timeit = []
        self.samples = []
        # Initialize the setup of the stereo pair and the other channel even if it wasn't defined in the Config file
        # Create the lists of files for the left, right and other camera
        dataset_conf = config["dataset"]["dataset_config"]
        paths = [config["dataset"]["inference_dir_left"],
                 config["dataset"]["inference_dir_right"],
                 config["dataset"]["inference_dir_other"]]
        left_files = None
        right_files = None
        other_files = None
        # If another extension of image is needed here it can be added
        if dataset_conf:
            try:
                left_files = dataset_conf['4. file_list']['left']
                right_files = dataset_conf['4. file_list']['right']
                other_files = dataset_conf['4. file_list']['other']
                self.path_left = str(Path(left_files[0]).parent.absolute())
                self.path_right = str(Path(right_files[0]).parent.absolute())
                self.path_other = str(Path(other_files[0]).parent.absolute())
            except KeyError:
                self.path_left = dataset_conf['1. Paths']['inference_dir_left']
                self.path_right = dataset_conf['1. Paths']['inference_dir_right']
                self.path_other = dataset_conf['1. Paths']['inference_dir_other']

        elif paths:
            self.path_left = paths[0]
            self.path_right = paths[1]
            self.path_other = paths[2]

        else:
            raise FileNotFoundError("We need paths for the data")
        if left_files is None:
            left_files = glob(self.path_left + '/*.png') + \
                         glob(self.path_left + '/*.jpg') + \
                         glob(self.path_left + '/*.jpeg')
        if right_files is None:
            right_files = glob(self.path_right + '/*.png') + \
                          glob(self.path_right + '/*.jpg') + \
                          glob(self.path_right + '/*.jpeg')
        if other_files is None:
            other_files = glob(self.path_other + '/*.png') + \
                          glob(self.path_other + '/*.jpg') + \
                          glob(self.path_other + '/*.jpeg')

        if config["dataset"]["number_of_sample"] <= 0:
            nb = len(left_files)
        else:
            nb = int(config["dataset"]["number_of_sample"])
        # If the data needs to be shuffled
        left_files = sorted(left_files)
        right_files = sorted(right_files)
        other_files = sorted(other_files)
        if config["dataset"]["shuffle"]:
            idx = np.arange(0, len(left_files))
            random.shuffle(idx)
            idx = idx[:nb]
            left_files = [left_files[i] for i in idx]
            right_files = [right_files[i] for i in idx]
            other_files = [other_files[i] for i in idx]
        else:
            left_files = left_files[:nb]
            right_files = right_files[:nb]
            other_files = other_files[:nb]

        assert len(left_files) == len(right_files) == len(other_files)
        num_samples = len(left_files)
        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['other'] = other_files[i]

            self.samples.append(sample)
        if config['dataset']["save_dataset_conf"]:
            if config["dataset"]["save_file_list_in_conf"]:
                self.save_conf(left=left_files, right=right_files, other=other_files)
            else:
                self.save_conf(left=None, right=None, other=None)
        if self.config['print_info']:
            print(self)

    @timeit
    def __getitem__(self, index, dtype=None):
        sample = {}
        sample_path = self.samples[index]
        if self.data_type == "2vis":
            sample['left'] = ImageCustom(sample_path['left'], dtype=dtype)  # [H, W, 3]
            sample['right'] = ImageCustom(sample_path['right'], dtype=dtype)  # [H, W, 3]
            sample['other'] = ImageCustom(sample_path['other'], dtype=dtype).match_shape(sample['right'],
                                                                                         keep_ratio=True,
                                                                                         channel=True)  # [H, W, 3]
        else:
            sample['left'] = ImageCustom(sample_path['left'], dtype=dtype).expand_dims()  # [H, W, 3]
            sample['right'] = ImageCustom(sample_path['right'], dtype=dtype).expand_dims()  # [H, W, 3]
            sample['other'] = ImageCustom(sample_path['other'], dtype=dtype).match_shape(sample['left'],
                                                                                         keep_ratio=True,
                                                                                         channel=False)  # [H, W, 3]
        return sample

    def __get_ref_images__(self):
        return self.__getitem__(0, dtype=None)

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self

    def __str__(self):
        string = f'\n############# DATASET ######'
        if self.config["dataset"]["type"] == '2vis':
            string += f'\nThe dataset is composed of {2 * len(self)} RGB images and {len(self)} infrared images'
        else:
            string += f'\nThe dataset is composed of {2 * len(self)} infrared images and {len(self)} RGB images'
        return string

    def __update_conf__(self, config):
        # Validation option
        self.config = config
        p = config['dataset']['projection_process']
        self.validation = config["validation"]["activated"]
        self.target = Projection_process[p['step1']][1 - Projection_process[p['step1']][2]]
        self.ref = Projection_process[p['step2']][Projection_process[p['step2']][2]]

        # Path to the main folder of the data. If several are given, the left folder parent will be the reference path
        self.path = Path(config["dataset"]["inference_dir_left"]).parent.absolute()
        config["dataset"]["path"] = str(self.path)
        self.device = config["device"]["device"]
        # Save the special parameter for the chosen network
        self.network_args = config["network"]["network_args"]
        self.data_type = config['dataset']["type"]
        self.save_inputs = config['save_inputs']

        ### The geometrical configuation is stored here #########
        self.setup = {"position_setup": config['dataset']["position_setup"],
                      "pred_bidir_disp": config['dataset']["pred_bidir_disp"],
                      "proj_right": config['dataset']["proj_right"],
                      "pred_right_disp": config['dataset']["pred_right_disp"]}

    def save_conf(self, load_and_save=False, left=None, right=None, other=None):
        name = os.path.join(self.path, "dataset.yaml")
        if load_and_save and os.path.exists(name):
            with open(name, "r") as file:
                dataset_conf = yaml.safe_load(file)
        else:
            dataset_conf = {}
        dataset_conf['0. Generalities'] = {"type": self.data_type,
                                           "Number of sample": len(self)}
        dataset_conf['1. Paths'] = {"inference_dir_left": self.path_left,
                                    "inference_dir_right": self.path_right,
                                    "inference_dir_other": self.path_other}
        dataset_conf['2. disparity configuration'] = {"pred_bidir_disp": self.setup["pred_bidir_disp"],
                                                      "proj_right": self.setup["proj_right"],
                                                      "pred_right_disp": self.setup["pred_right_disp"]}
        dataset_conf['3. Position and alignment'] = {"position_setup": self.setup["position_setup"]}
        if left is not None and right is not None and other is not None:
            dataset_conf['4. file_list'] = {'left': left,
                                            'right': right,
                                            'other': other}
        with open(name, "w") as file:
            yaml.dump(dataset_conf, file)
