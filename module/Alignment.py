import os
from pathlib import Path
from Config.Config import ConfigPipe, define_target_ref, check_path, create_dir_disp
from module.BaseModule import BaseModule
from utils.classes.Image import ImageCustom
from module.DataLoader import StereoDataLoader
from module.SuperNetwork import SuperNetwork
from module import Projection_process
from utils.misc import timeit
from utils.registration_tools import manual_registration, crop_from_cut, automatic_registration, \
    manual_position_calibration
import cv2 as cv
import numpy as np
import yaml


class Alignment(BaseModule):
    """
    A block which is called when the dataset need an alignment
    It can be automatic if the option is set.
    """

    def __init__(self, dataloader: StereoDataLoader, config: ConfigPipe, model: SuperNetwork):
        super(Alignment, self).__init__(config, dataloader, model)

    def _update_conf(self, config, *args, **kwargs):
        dataloader, model = args[0], args[1]
        self.alignment_isDone = self.config['dataset']['alignment_isDone']
        self.alignment_auto = self.config['dataset']['alignment_auto']
        self.inverted_left_right = False
        self.transformation_matrix_step1 = None
        self.transformation_matrix_step2 = None
        self.Cut = [0, 0, 0, 0]  # Top, Left, Bot, Right
        self.path = Path(config["dataset"]["inference_dir_left"]).parent.absolute()
        self.save_calibration = config['save_calibration']
        self.dataloader = dataloader
        self.ref_images = dataloader.__get_ref_images__()
        self.config['dataset']['ori_size'] = self.ref_images['left'].shape[:2]
        self.calib = False
        if not self.alignment_isDone:
            self.pos = dataloader.setup["position_setup"]
            self.projection_process = config['dataset']['projection_process']
            self.alignment_auto = self.alignment_auto
            self.model = model
            self.path = Path(config["dataset"]["inference_dir_left"]).parent.absolute()
            self.load_if_available = config["dataset"]["load_dataset_conf_if_available"]
            self.data_type = config['dataset']["type"]
            if self.load_if_available:
                with open(os.path.join(self.path, "dataset.yaml"), 'r') as file:
                    dataset_config = yaml.safe_load(file)
                try:
                    self.transformation_matrix_step1 = np.array(
                        dataset_config["3. Position and alignment"]['transformation_matrix_step1'])
                    self.Cut = dataset_config["3. Position and alignment"]["Cut"]
                    self.transformation_matrix_step2 = np.array(
                        dataset_config["3. Position and alignment"]['transformation_matrix_step2'])
                    self.inverted_left_right = dataset_config["3. Position and alignment"]['inverted_left_right']
                except KeyError:
                    print("No Transformation Matrix found, Calibration required")
                    self._calibrate()
                    self.calib = True
            else:
                self._calibrate()
                self.calib = True
            if self.save_calibration:
                self.save_calib()
        elif self.save_calibration:
            self.save_calib(alignment_isDone=True)

    def __str__(self):
        string = super().__str__()
        if self.config['dataset']['alignment_isDone']:
            string += "The dataset is already aligned"
        else:
            if self.config['dataset']['alignment_auto']:
                string += "An automatic registration using SIFT will be done"
            else:
                string += "A manual registration will be done"
        return string

    @timeit
    def __call__(self, sample, alignment_step=0, **kwargs):
        if isinstance(sample, dict):
            new_sample = {}
            if alignment_step == 1:
                if self.inverted_left_right:
                    sample['left'], sample['right'] = sample['right'], sample['left']
                if self.transformation_matrix_step1 is not None:
                    p = Projection_process[self.projection_process['step1']]
                    new_sample = self._warpPerspective(sample, p, self.transformation_matrix_step1, no_cut=True)
                    new_sample[p[0]], new_sample[p[1]] = \
                        crop_from_cut(new_sample[p[0]], None, self.Cut), crop_from_cut(new_sample[p[1]], None, self.Cut)
            if alignment_step == 2:
                if self.transformation_matrix_step2 is not None:
                    p = Projection_process[self.projection_process['step2']]
                    new_sample = self._warpPerspective(sample, p, self.transformation_matrix_step2, no_cut=True)
                    new_sample[p[0]], new_sample[p[1]] = \
                        crop_from_cut(new_sample[p[0]], None, self.Cut), crop_from_cut(new_sample[p[1]], None, self.Cut)
            else:
                if self.inverted_left_right:
                    sample['left'], sample['right'] = sample['right'], sample['left']
                if self.transformation_matrix_step1 is not None:
                    p = Projection_process[self.projection_process['step1']]
                    new_sample = self._warpPerspective(sample, p, self.transformation_matrix_step1, no_cut=True)
                if self.transformation_matrix_step2 is not None:
                    p = Projection_process[self.projection_process['step2']]
                    new_sample = self._warpPerspective(sample, p, self.transformation_matrix_step2, no_cut=True)
                for key, im in new_sample.items():
                    new_sample[key] = crop_from_cut(im, None, self.Cut)
            return new_sample
            # if alignment_step == 1:
            #     if self.transformation_matrix_step1 is not None:
            #         sample = self._warpPerspective(sample, Projection_process[self.projection_process['step1']],
            #                                        self.transformation_matrix_step1, no_cut=True)
            #     if alignment_step == 2:
            #         if self.transformation_matrix_step2 is not None:
            #             sample = self._warpPerspective(sample, Projection_process[self.projection_process['step2']],
            #                                            self.transformation_matrix_step2, no_cut=True)
            # return sample
        elif isinstance(sample, list) and self.data_type == '2ir':
            new_sample = []
            for im in sample:
                if alignment_step == 1:
                    if self.transformation_matrix_step1 is not None:
                        new_sample.append(self._warpPerspective(im, None, self.transformation_matrix_step1, no_cut=True))
                elif alignment_step == 2:
                    if self.transformation_matrix_step2 is not None:
                        new_sample.append(self._warpPerspective(im, None, self.transformation_matrix_step2, no_cut=True))
                else:
                    if self.transformation_matrix_step1 is not None:
                        im = (self._warpPerspective(im, None, self.transformation_matrix_step1, no_cut=True))
                    if self.transformation_matrix_step2 is not None:
                        new_sample.append(self._warpPerspective(im, None, self.transformation_matrix_step2, no_cut=False))
            return new_sample

    def _warpPerspective(self, sample, process, matrix, no_cut=False):
        if isinstance(sample, dict):
            im = sample[process[process[2]]].copy()
            height, width = im.shape[:2]
            im = cv.warpPerspective(im, matrix, (width, height), flags=cv.INTER_LINEAR)
            if not no_cut:
                im = crop_from_cut(im, None, self.Cut)
            sample[process[process[2]]] = ImageCustom(im, sample[process[process[2]]])
        else:
            height, width = sample.shape[:2]
            sample = cv.warpPerspective(sample, matrix, (width, height), flags=cv.INTER_LINEAR)
            sample = crop_from_cut(sample, None, self.Cut)
        return sample

    def _calibrate(self):
        """
        """
        print(self.projection_process)
        pos, self.inverted_left_right = manual_position_calibration(self.ref_images['left'], self.ref_images['right'],
                                                                    [0, 0])
        if self.inverted_left_right:
            self.ref_images['left'], self.ref_images['right'] = self.ref_images['right'], self.ref_images['left']
            pos[0] = -pos[0]
        pos, _ = manual_position_calibration(self.ref_images['left'], self.ref_images['other'], pos)
        if self.pos is None or not self.config['dataset']["use_pos"]:
            self.pos = pos
            self.config['dataset']["position_setup"] = self.pos
        self.config['dataset']["use_pos"] = True
        self.config.configure_projection_option()
        self.projection_process = define_target_ref(self.config['dataset'])
        self.config['dataset']['projection_process'] = self.projection_process

        if self.config['save_disp']:
            create_dir_disp(self.config)
        self.dataloader.__update_conf__(self.config)
        self.dataloader.save_conf(load_and_save=True)
        if self.alignment_auto:
            self.transformation_matrix_step1, self.Cut = automatic_registration(self.ref_images,
                                                                                self.path,
                                                                                self.data_type,
                                                                                Projection_process[
                                                                                    self.projection_process['step1']])
            self.ref_images = self(self.ref_images, alignment_step=1)
            self.transformation_matrix_step2, self.Cut = automatic_registration(self.ref_images,
                                                                                self.path,
                                                                                self.data_type,
                                                                                Projection_process[
                                                                                    self.projection_process['step2']])
        else:
            self.transformation_matrix_step1, self.Cut = manual_registration(
                self.ref_images,
                self.model,
                Projection_process[self.projection_process['step1']],
                self.Cut)
            # self.ref_images = self(self.ref_images, alignment_step=1)
            self.transformation_matrix_step2, self.Cut = manual_registration(
                self.ref_images,
                self.model,
                Projection_process[self.projection_process['step2']],
                self.Cut)
            self.model.timeit = []

    def save_calib(self, alignment_isDone=False):
        # path = os.path.join(self.path, "calib")
        name = os.path.join(self.path, "dataset.yaml")
        with open(name, 'r') as file:
            dataset_config = yaml.safe_load(file)
        if not alignment_isDone:
            dataset_config['3. Position and alignment'][
                "transformation_matrix_step1"] = self.transformation_matrix_step1.tolist()
            dataset_config['3. Position and alignment']["Cut"] = self.Cut
            dataset_config['3. Position and alignment'][
                "transformation_matrix_step2"] = self.transformation_matrix_step2.tolist()
            dataset_config['3. Position and alignment']["alignment_isDone"] = False
            dataset_config['3. Position and alignment']["alignment_auto"] = self.alignment_auto
            dataset_config['3. Position and alignment']["position_setup"] = self.pos
            dataset_config['3. Position and alignment']["inverted_left_right"] = self.inverted_left_right
        else:
            dataset_config['3. Position and alignment']["transformation_matrix_step1"] = None
            dataset_config['3. Position and alignment']["Cut"] = [0, 0, 0, 0]
            dataset_config['3. Position and alignment']["transformation_matrix_step2"] = None
            dataset_config['3. Position and alignment']["alignment_isDone"] = True
            dataset_config['3. Position and alignment']["alignment_auto"] = False
            dataset_config['3. Position and alignment']["inverted_left_right"] = False
        with open(name, "w") as file:
            yaml.dump(dataset_config, file)
