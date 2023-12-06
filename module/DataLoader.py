import os
import random
from collections import OrderedDict
from glob import glob
import numpy as np
import yaml
from torch.utils.data import Dataset, DataLoader, TensorDataset
from module.SetupCameras import CameraSetup
from utils.classes.Image import ImageTensor
from utils.manipulation_tools import merge_dict, list_to_dict
from utils.misc import timeit, name_generator
from torch import device


# class StereoDataLoader(DataLoader):
#     """
#     This class implement a DataLoader instance from the StereoDataset class implemented from the camera setup
#     """
#     def __init__(self, setup: CameraSetup, config, batch_size=1):
#         self.config = config
#         stereoDataSet = StereoDataSet(setup, config)
#         super(StereoDataLoader, self).__init__(stereoDataSet,
#                                                batch_size=batch_size,
#                                                shuffle=stereoDataSet.shuffle)
#
#     @property
#     def timeit(self):
#         if self.config["timeit"]:
#             return self.dataset.timeit
#
#     @timeit.setter
#     def timeit(self, value):
#         if self.config["timeit"]:
#             self.dataset.timeit = value


class StereoDataLoader(Dataset):
    """
    This dataloader loads the files according to the options set.
    With an input stereo + mono, specify the geometry on the camera disposition
    """

    def __init__(self, setup: CameraSetup, config):
        super(StereoDataLoader, self).__init__()
        self.__update_conf__(config)
        if config["timeit"]:
            self.timeit = []
        self.path = setup.cameras[setup.camera_ref].path
        self.camera_setup = setup

        # Initialize the setup of the stereo pair and the other channel even if it wasn't defined in the Config file
        # Create the lists of files for the left, right and other camera
        # dataset_conf = config["dataset"]["dataset_config"]
        self.cameras_paths = {key: cam.path for key, cam in setup.cameras.items()}
        # If another extension of image is needed here it can be added
        if self.cameras_paths == {}:
            raise FileNotFoundError("We need paths for the data")
        self.files = {}
        for key, p in self.cameras_paths.items():
            self.files[key] = sorted(glob(p + '/*.png') +
                                     glob(p + '/*.jpg') +
                                     glob(p + '/*.jpeg'))

        if config["dataset"]["number_of_sample"] <= 0:
            self.nb = len(self.files[setup.camera_ref])
        else:
            self.nb = int(config["dataset"]["number_of_sample"])
        # If the data needs to be shuffled
        idx = config["dataset"]["indexes"]
        if idx is not None:
            idx = np.uint64(idx) % len(self.files[setup.camera_ref])
        self.shuffle = config["dataset"]["shuffle"]
        if config["dataset"]["shuffle"]:
            idx = np.arange(0, len(self.files[setup.camera_ref])) if idx is None else idx
            random.shuffle(idx)
            idx = idx[:self.nb]
            for key, f in self.files.items():
                self.files[key] = [self.files[key][i] for i in idx]
        else:
            for key, f in self.files.items():
                if self.nb < len(self.files[setup.camera_ref]):
                    self.files[key] = self.files[key][:self.nb] if idx is None else np.array(self.files[key])[idx].tolist()
                else:
                    self.files[key] = self.files[key] if idx is None else np.array(self.files[key])[idx].tolist()


        self.samples = []
        self.camera_used = []
        self.reset_images_name = config['reset_images_name']

    @timeit
    def __getitem__(self, index):
        sample = {key: ImageTensor(p, device=self.device) for key, p in self.samples[index].items()}
        if self.reset_images_name:
            for key in sample.keys():
                sample[key].im_name = f'{key}_{name_generator(index, max_number=len(self))}'
        return sample

    def __get_ref_images__(self):
        sample = {key: cam.im_calib for key, cam in self.camera_setup.cameras.items()}
        return sample

    def __len__(self):
        return len(self.samples)

    @property
    def camera_used(self):
        return self._camera_used

    @camera_used.setter
    def camera_used(self, value):
        self._camera_used = value
        self.samples = [{key: self.files[key][i] for key in self.camera_used} for i in range(self.nb)]

    def __rmul__(self, v):
        self.samples = v * self.samples
        return self

    def __str__(self):
        string = f'\n############# DATASET ######'
        string += f'\nThe dataset is composed of {len(self.camera_setup.cameras_RGB) * len(self)} ' \
                  f'RGB images and {len(self.camera_setup.cameras_IR) * len(self)} infrared images'
        return string

    def __update_conf__(self, config):
        # Validation option
        self.config = config
        # Path to the main folder of the data. If several are given, the left folder parent will be the reference path
        self.device = config["device"]["device"]
        self.save_inputs = config['save_inputs']

    def save_conf(self, output_path):
        name = os.path.join(output_path, "dataset.yaml")
        dataset_conf = OrderedDict({'Number of sample': len(self),
                                    'Files': list_to_dict(self.samples)})
        with open(name, "w") as file:
            yaml.dump(dataset_conf, file)
