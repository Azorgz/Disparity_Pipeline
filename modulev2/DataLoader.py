import os
import random
from glob import glob
import numpy as np
import yaml
from torch.utils.data import Dataset
from utils.classes.SetupCameras import CameraSetup
from utils.classes.Image import ImageTensor
from utils.misc import timeit, name_generator


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
            nb = len(self.files[setup.camera_ref])
        else:
            nb = int(config["dataset"]["number_of_sample"])
        # If the data needs to be shuffled

        if config["dataset"]["shuffle"]:
            idx = np.arange(0, len(self.files[setup.camera_ref]))
            random.shuffle(idx)
            idx = idx[:nb]
            for key, f in self.files.items():
                self.files[key] = [self.files[key][i] for i in idx]
        else:
            for key, f in self.files.items():
                self.files[key] = self.files[key][:nb]

        self.samples = [{key: p[i] for key, p in self.files.items()} for i in range(nb)]
        self.reset_images_name = config['reset_images_name']

    @timeit
    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = {key: ImageTensor(p) for key, p in sample_path.items()}
        if self.reset_images_name:
            for key in sample.keys():
                sample[key].im_name = f'{key}_{name_generator(index, max_number=10**(len(self)%10+1))}'
        return sample

    def __get_ref_images__(self):
        sample = {key: cam.im_calib for key, cam in self.camera_setup.cameras.items()}
        return sample

    def __len__(self):
        return len(self.samples)

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

    def save_conf(self, load_and_save: bool = False, files: bool = False):
        name = os.path.join(self.path, "dataset.yaml")
        if load_and_save and os.path.exists(name):
            with open(name, "r") as file:
                dataset_conf = yaml.safe_load(file)
        else:
            dataset_conf = {}
        dataset_conf['0.Number of sample'] = len(self)
        dataset_conf['1.Paths'] = {key: p for key, p in self.cameras_paths.items()}
        if files:
            dataset_conf['2.Files'] = {key: p for key, p in self.files.items()}
        with open(name, "w") as file:
            yaml.dump(dataset_conf, file)
