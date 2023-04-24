import os
import random
from glob import glob
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml
from torch.utils.data import Dataset

from classes.Image import ImageCustom
from utils.misc import timeit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataLoader(Dataset):
    """
    This dataloader loads the files according to the options set.
    With an input stereo + mono, specify the geometry on the camera disposition
    """

    def __init__(self,config, setup=None):
                 # dataset_conf: dict or None,
                 # data_type: str,
                 # *args,
                 # config):
                 # save_inputs=False,
                 # half_resolution=False,
                 # device=None,
                 # setup=None,
                 # shuffle=False,
                 # timeit=False,
                 # save_dataset_conf=False):
        super(StereoDataLoader, self).__init__()
        # If the DisparityPipe need to measure the execution time of each block, this parameter will be set to True.
        paths = [config["dataset"]["inference_dir_left"],
                 config["dataset"]["inference_dir_right"],
                 config["dataset"]["inference_dir_other"]]
        dataset_conf = config["dataset"]["dataset_config"]

        if config["timeit"]:
            self.timeit = []
        # Save the special parameter for the chosen network
        self.network_args = config["network"]["network_args"]
        # Initialize the setup of the stereo pair and the other channel even if it wasn't defined in the Config file
        if setup is None:
            self.setup = {"position_setup": [70, 341],
                          "pred_bidir_disp": False,
                          "proj_right": False,
                          "pred_right_disp": False}
        else:
            self.setup = setup
        self.device = config["device"]["device"]
        self.always_proj_infrared = config["always_proj_infrared"]
        # Create the lists of files for the left, right and other camera
        # If another extension of image is needed here it can be added
        if dataset_conf:
            left_files = dataset_conf['file_list']['left']
            right_files = dataset_conf['file_list']['right']
            other_files = dataset_conf['file_list']['other']
        elif paths:
            self.path_left = paths[0]
            self.path_right = paths[1]
            self.path_other = paths[2]
            left_files = glob(self.path_left + '/*.png') + \
                         glob(self.path_left + '/*.jpg') + \
                         glob(self.path_left + '/*.jpeg')
            right_files = glob(self.path_right + '/*.png') + \
                          glob(self.path_right + '/*.jpg') + \
                          glob(self.path_right + '/*.jpeg')
            other_files = glob(self.path_other + '/*.png') + \
                          glob(self.path_other + '/*.jpg') + \
                          glob(self.path_other + '/*.jpeg')
        else:
            raise FileNotFoundError("We need paths for the data")
        # If the data needs to be shuffled
        if config["shuffle"]:
            idx = np.arange(0, len(left_files))
            random.shuffle(idx)
            left_files = left_files[idx]
            right_files = right_files[idx]
            other_files = other_files[idx]
        else:
            left_files = sorted(left_files)
            right_files = sorted(right_files)
            other_files = sorted(other_files)

        self.data_type = config['dataset']["type"]
        self.half_resolution = self.network_args.half_resolution
        self.save_inputs = config['save_inputs']
        self.samples = []
        assert len(left_files) == len(right_files) == len(other_files)
        num_samples = len(left_files)
        for i in range(num_samples):
            sample = dict()

            sample['left'] = left_files[i]
            sample['right'] = right_files[i]
            sample['other'] = other_files[i]

            self.samples.append(sample)
        if config['dataset']["save_dataset_conf"]:
            self.save_conf(left=left_files, right=right_files, other=other_files, type=self.data_type)

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
        elif self.data_type == "2ir":
            sample['left'] = ImageCustom(sample_path['left'], dtype=dtype).expand_dims()  # [H, W, 3]
            sample['right'] = ImageCustom(sample_path['right'], dtype=dtype).expand_dims()  # [H, W, 3]
            if self.always_proj_infrared:
                if self.setup["proj_right"]:
                    sample['other'] = ImageCustom(sample_path['right'], dtype=dtype).new_axis()  # [H, W, 1]
                else:
                    sample['other'] = ImageCustom(sample_path['left'], dtype=dtype).new_axis()  # [H, W, 1]
            else:
                sample['other'] = ImageCustom(sample_path['other'], dtype=dtype).match_shape(sample['left'],
                                                                                             keep_ratio=True,
                                                                                             channel=False)  # [H, W, 3]
        # print(sample['left'].shape, sample['right'].shape, sample['other'].shape)
        if self.half_resolution:
            sample['left'] = ImageCustom(
                cv.resize(sample['left'], None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR),
                sample['left'], dtype=dtype)
            sample['right'] = ImageCustom(
                cv.resize(sample['right'], None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR),
                sample['right'], dtype=dtype)
            sample['other'] = ImageCustom(
                cv.resize(sample['other'], None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR),
                sample['other'], dtype=dtype)
        if self.save_inputs:
            sample_inputs = {"left": ImageCustom(sample_path['left']),
                             "right": ImageCustom(sample_path['right']),
                             "other": ImageCustom(sample_path['other'])}
            return sample, sample_inputs
        else:
            return sample, None

    def __len__(self):
        return len(self.samples)

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self

    def save_conf(self, left=None, right=None, other=None, type=None):
        dataset_conf = {"position_setup": self.setup["position_setup"],
                        "pred_bidir_disp": self.setup["pred_bidir_disp"],
                        "proj_right": self.setup["proj_right"],
                        "pred_right_disp": self.setup["pred_right_disp"],
                        "type": type,
                        'file_list': {'left': left,
                                      'right': right,
                                      'other': other}
                        }
        path = Path(left[0])
        name = os.path.join(path.parent.parent.absolute(), "dataset.yaml")
        with open(name, "w") as file:
            yaml.dump(dataset_conf, file)
#
# class UnimatchDataLoader(StereoDataLoader):
#     def __init__(self,
#                  paths: list,
#                  unique_path: os.path or str,
#                  data_type: str,
#                  save_inputs=False,
#                  half_resolution=False,
#                  args=None,
#                  device=None,
#                  transforms=None,
#                  setup=None,
#                  shuffle=False):
#         super(UnimatchDataLoader, self).__init__(paths,
#                                                  unique_path,
#                                                  data_type,
#                                                  save_inputs=save_inputs,
#                                                  half_resolution=half_resolution,
#                                                  device=device,
#                                                  setup=setup,
#                                                  shuffle=shuffle)
#
#         self.transform = transforms
#         self.args = args
#
#     def __getitem__(self, index, **kwargs):
#
#         sample, sample_inputs = super(UnimatchDataLoader, self).__getitem__(index, dtype=np.float32)
#
#         if self.transform is not None:
#             sample = self.transform(sample)
#
#         self.args.inference_size = self.transform.transforms[-1].inference_size
#         self.args.ori_size = self.transform.transforms[-1].ori_size
#
#         if not self.save_inputs:
#             return sample, None
#         else:
#             return sample, sample_inputs
