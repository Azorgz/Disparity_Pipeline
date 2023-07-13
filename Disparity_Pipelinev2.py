import os
import torch

import cv2 as cv
import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings
# from modulev2.PointCloudTransformer import PointCloudTransformer
from modulev2.disparityRefinement import disparity_refinement
# module
from modulev2.SuperNetwork import SuperNetwork
from modulev2.DataLoader import StereoDataLoader
from modulev2.Reconstruction import Reconstruction
from modulev2.Validation import Validation

# Config
from configv2.Config import ConfigPipe
from utils.classes.Image import ImageTensor
from utils.classes.SetupCameras import CameraSetup
# from utils.disparity_tools import find_occluded_pixel

# Utils
from utils.misc import count_parameter, name_generator, time2str, form_cloud_data  # , form_cloud_data

# Networks
from Networks.UniMatch.unimatch.unimatch import UniMatch
from Networks.ACVNet.models.acv import ACVNet

# from classes.Image import ImageCustom
from Networks.UniMatch.utils.visualization import vis_disparity
from utils.transforms import ToNumpy, ToTensor, Resize_disp


class Pipe:
    """
    A class defining the pipe of processing of the chosen process
    Option of Input :  2 Channels visible + 1 Infrared OR 1 Channel visible + 2 Infrared
    An instance is created using the parameter given either by the interface, either by the config file.
    """

    def __init__(self, config: ConfigPipe):

        self.config = config
        # The basic functional options are stored here ###############
        self.print_info = config["print_info"]
        self.timeit = config["timeit"]
        self.device = config["device"]["device"]
        if self.print_info and str(self.device) != 'cpu':
            print(f'\n############# DEVICE ######'
                  f'\nThe process will run on {torch.cuda.get_device_name(device=self.device)}')

        # The Data information is stored here ###############
        self.save_inputs = config["save_inputs"]
        self.path_output = config["output_path"]
        self.save_disp = config["save_disp"]
        self.save_reg_images = config["save_reg_images"]
        self.name_experiment = config['name_experiment']

        # The different modules of the Pipe are initialized ###############
        self.modules = []
        self._init_setup_()
        self._init_dataloader_()
        self._init_network_()
        # self._init_reconstruction_()
        # self._init_validation_()
        # self._init_pointsCloud_()

        # self.time_consistency_block = Time_consistency_block(config.time_consistency_block)
        # self.post_processing = Post_processing(config.Post_processing)
        # self.reconstruction_module = self._initialize_reconstruction_(config)

    @torch.no_grad()
    def run(self, process):
        for idx, sample in tqdm(enumerate(self.dataloader),
                                total=len(self.dataloader),
                                desc="Nombre d'itérations : "):
            # Preparation of the result dictionary ################
            # result = self.init_result(sample, idx)

            # # Disparity Inference with the selected Network ##############
            # intrinsics = self.setup.cameras['RGB2'].intrinsics[:, :3, :3].to(torch.float32)
            # pose = torch.linalg.inv(self.setup.cameras['RGB2'].extrinsics.to(torch.float32)) @ self.setup.cameras['RGB'].extrinsics.to(torch.float32)
            self.network.update_pred_bidir(activate=False)
            # pred_disp = self.disparity_network(sample['RGB'], sample['RGB2'], intrinsics, pose, depth=True)
            new_sample = self.setup.stereo_pair('RGB', 'RGB2')(sample, cut_roi_min=True)
            pred_disp = self.network(new_sample)
            # disp_new = disparity_refinement(new_sample['left'], pred_disp['left'] )
            pred_disp = self.setup.stereo_pair('RGB', 'RGB2')(pred_disp, reverse=True)
            pred_disp['RGB'].show()
            (pred_disp['RGB'] * sample['RGB']).show()
            (pred_disp['RGB2'] * sample['RGB2']).show()

            #
            # # Postprocessing following the Preprocessing and other chosen options ##############
            # pred_disp = self.postprocessing(pred_disp)
            # # Completion of the result dictionary #######
            # if self.save_disp:
            #     if self.config['dataset']["pred_bidir_disp"]:
            #         disp = pred_disp.cpu().numpy()
            #         result["disp_left"] = vis_disparity(disp[0])
            #         result["disp_right"] = vis_disparity(disp[1])
            #     else:
            #         result["disp"] = vis_disparity(pred_disp[0].cpu().numpy())
            # cv.imshow('disp', result["disp_left"])
            # cv.waitKey(0)
            # x = range(self.postprocessing.bins)
            # plt.bar(x, self.postprocessing.histo.cpu(), align='center')
            # plt.xlabel('Bins')
            # plt.show()
            # Reconstruction using the estimated disparity ###########
            # side = 'right' if self.config['dataset']["pred_bidir_disp"] else 'left'
            # new_sample[side], pred_disp = self.alignment([sample[side], pred_disp.cpu().numpy()], alignment_step=2)
            # sample_preprocess = self.preprocessing(new_sample, self)
            # image_reg, new_disp = self.reconstruction(pred_disp, sample_preprocess)
            # if self.reconstruction.method != 'pytorch':
            #     resize_disp = Resize_disp(self.config['network']["network_args"].inference_size)
            #     new_disp_tensor = resize_disp(torch.tensor(new_disp, device=self.device).unsqueeze(0), self.device).unsqueeze(0)
            # else:
            #     ref_tensor, new_disp_tensor = sample[self.dataloader.ref], new_disp
            # new_disp_reg = self.disparity_network.refine(sample_preprocess[self.dataloader.ref], new_disp_tensor.to(sample_preprocess[self.dataloader.ref].dtype)).squeeze().cpu()
            # cv.imshow('before', vis_disparity(to_numpy(new_disp)))

        #     # Visualisation 3D of the result ###########
        #     sample_cloud, disp_cloud = form_cloud_data(sample, pred_disp, image_reg, new_disp, self.config)
        #     self.pointCloudTransformer(disp_cloud, sample_cloud, result["inputs_name"]['left'])
        #
        #     # Assessment of the result quality ###########
        #     self.validation(image_reg, sample[self.dataloader.target], sample[self.dataloader.ref])
        #     if self.reconstruction.method == "pytorch":
        #         result["image_reg"] = to_numpy_normalize(image_reg)
        #         result["new_disp"] = vis_disparity(to_numpy(new_disp))
        #     else:
        #         if image_reg.max() <= 1:
        #             result["image_reg"] = (image_reg * 255.).astype("uint8")
        #         else:
        #             result["image_reg"] = image_reg.astype("uint8")
        #         result["new_disp"] = vis_disparity(new_disp)
        #     # to_np = ToNumpy()
        #     # sample_im = to_np(sample)
        #     # cv.imshow('fus', result["image_reg"]/510+sample_im["left"]/510)
        #     # cv.waitKey(0)
        #     self.save_result(result)
        # self.validation.statistic()
        # self.validation.save(self.path_output)
        # if self.timeit:
        #     self.save_timers()

    def save_result(self, result):
        if self.save_inputs:
            for key in result["inputs"].keys():
                path_input = os.path.join(self.path_output, "input", key)
                name = result["inputs_name"][key] + f'_{key}.png'
                res = cv.cvtColor(np.uint8(result["inputs"][key]), cv.COLOR_RGB2BGR)
                if not cv.imwrite(os.path.join(path_input, name), res):
                    raise Exception("Could not write image")
        if self.save_disp:
            if self.config['dataset']["pred_bidir_disp"]:
                path_disp = os.path.join(self.path_output, "disp_left")
                name = result["inputs_name"]["left"] + '_disp_left.png'
                if not cv.imwrite(os.path.join(path_disp, name), result["disp_left"]):
                    raise Exception("Could not write image")
                path_disp = os.path.join(self.path_output, "disp_right")
                name = result["inputs_name"]["right"] + '_disp_right.png'
                if not cv.imwrite(os.path.join(path_disp, name), result["disp_right"]):
                    raise Exception("Could not write image")
            elif self.config['dataset']['pred_right_disp']:
                path_disp = os.path.join(self.path_output, "disp_right")
                name = result["inputs_name"]["left"] + '_right.png'
                if not cv.imwrite(os.path.join(path_disp, name), result["disp"]):
                    raise Exception("Could not write image")
            else:
                path_disp = os.path.join(self.path_output, "disp_left")
                name = result["inputs_name"]["left"] + '_disp_left.png'
                if not cv.imwrite(os.path.join(path_disp, name), result["disp"]):
                    raise Exception("Could not write image")
            path_disp = os.path.join(self.path_output, "disp_other")
            name = result["inputs_name"]["other"] + '_disp_other.png'
            if not cv.imwrite(os.path.join(path_disp, name), result["new_disp"]):
                raise Exception("Could not write image")
        if self.save_reg_images:
            path = os.path.join(self.path_output, "reg_images")
            name = result["inputs_name"]["other"] + '_reg.png'
            if not cv.imwrite(os.path.join(path, name), result["image_reg"]):
                raise Exception("Could not write image")

    def save_timers(self):
        time_dict = {"1. Sample Number": len(self.dataloader),
                     "2. Total Execution time": "0",
                     "3. Time per module": {}}
        tot = 0
        for m in self.modules:
            name = m.__class__.__name__
            time = m.timeit
            if time:
                time_dict["3. Time per module"][name] = {
                    "Average time": time2str(sum(time) / (len(time) + 0.00001)),
                    "Min time": time2str(min(time)),
                    "Max time": time2str(max(time)),
                    "Total time": time2str(sum(time))}
                tot += sum(time)
        time_dict["2. Total Execution time"] = time2str(tot)
        name = os.path.join(self.path_output, "Execution_time.yaml")
        with open(name, "w") as file:
            yaml.dump(time_dict, file)

    @torch.no_grad()
    def _init_setup_(self):
        assert self.config["setup"]['path'] is not None
        self.setup = CameraSetup(from_file=self.config["setup"]['path'])
        self.modules.append(self.setup)

    @torch.no_grad()
    def _init_dataloader_(self):
        self.dataloader = StereoDataLoader(self.setup, self.config)
        self.modules.append(self.dataloader)

    @torch.no_grad()
    def _init_network_(self):
        self.network = SuperNetwork(config)
        # self.network_name = self.disparity_network.name
        self.modules.append(self.network)

    # @torch.no_grad()
    # def _init_reconstruction_(self):
    #     self.reconstruction = Reconstruction(self.config)
    #     self.modules.append(self.reconstruction)
    #
    # @torch.no_grad()
    # def _init_validation_(self):
    #     self.validation = Validation(self.config)
    #     if self.validation.activated:
    #         self.modules.append(self.validation)
    #
    # @torch.no_grad()
    # def _init_pointsCloud_(self):
    #     self.pointCloudTransformer = PointCloudTransformer(self.config)
    #     if self.pointCloudTransformer.activated:
    #         self.modules.append(self.pointCloudTransformer)
    #
    # def _initialize_features_extraction_(self):
    #     return 0
    #
    # def _initialize_transformer_(self):
    #     return 0
    #
    # def _initialize_detection_head_(self):
    #     return 0
    #
    # def init_result(self, sample, idx):
    #     result = {"inputs_name": {}, "inputs": {}}
    #     if config["reset_images_name"]:
    #         result["inputs_name"]["left"] = f'{name_generator(idx, 1000)}'
    #         result["inputs_name"]["right"] = f'{name_generator(idx, 1000)}'
    #         result["inputs_name"]["other"] = f'{name_generator(idx, 1000)}'
    #     else:
    #         result["inputs_name"]["left"] = sample["left"].name
    #         result["inputs_name"]["right"] = sample["right"].name
    #         result["inputs_name"]["other"] = sample["other"].name
    #     if self.save_inputs:
    #         result["inputs"]["left"] = np.uint8(sample["left"])
    #         result["inputs"]["right"] = np.uint8(sample["right"])
    #         result["inputs"]["other"] = np.uint8(sample["other"])
    #     return result


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = ConfigPipe()
    pipe = Pipe(config)
    pipe.run(None)
    print('Done !')
