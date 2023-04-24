import os

from module.SuperNetwork import SuperNetwork
from Networks.UniMatch.unimatch.unimatch import UniMatch
from module.DataLoader import StereoDataLoader
from Networks.UniMatch.utils.visualization import vis_disparity
from Config.Config import ConfigPipe
from module.Preprocessing import Preprocessing
import torch
import cv2 as cv
import numpy as np
import yaml
from module.Reconstruction_block import ReconstructionBlock
from utils.misc import count_parameter, name_generator
from tqdm import tqdm
import warnings


class Pipe:
    """
    A class defining the pipe of processing of the chosen process
    Option of Input :  2 Channels visible + 1 Infrared OR 1 Channel visible + 2 Infrared
    An instance is created using the parameter given either by the interface, either by the config file.
    """

    def __init__(self, config: ConfigPipe):

        ### The basic functional options are stored here ###############
        self.print_info = config["print_info"]
        self.timeit = config["timeit"]
        self.device = config["device"]["device"]
        if self.print_info:
            print(f'\n'
                  f'############# DEVICE ######'
                  f'\nThe process will run on {torch.cuda.get_device_name(device=self.device)}')
        self.device_index = config["device"]["index"]

        ### The Data informations are stored here ###############
        self.save_inputs = config["save_inputs"]
        self.path_output = config["output_path"]
        self.save_disp = config["save_disp"]
        self.save_reg_images = config["save_reg_images"]
        self.name_experiment = config['name_experiment']

        ### The geometrical configuation is stored here #########
        self.setup = {"position_setup": config['dataset']["position_setup"],
                      "pred_bidir_disp": config['dataset']["pred_bidir_disp"],
                      "proj_right": config['dataset']["proj_right"],
                      "pred_right_disp": config['dataset']["pred_right_disp"]}

        ### The different modules of the Pipe are initialized ###############
        self.dataloader = self._init_dataloader_(config)
        self.preprocessing = self._init_preprocessing_(config)
        self.disparity_network = self._init_network_(config["network"])
        self.reconstruction = self._init_reconstruction_(config)

        # self.time_consistency_block = Time_consistency_block(config.time_consistency_block)
        # self.post_processing = Post_processing(config.Post_processing)
        # self.reconstruction_module = self._initialize_reconstruction_(config)

    @torch.no_grad()
    def run(self):
        for idx, (sample, sample_input) in tqdm(enumerate(self.dataloader),
                                                total=len(self.dataloader),
                                                desc="Nombre d'itérations : "):
            ######### Preparation of the result dictionnary ################
            result = {"inputs_name": {}, "inputs": {}}
            if config["reset_images_name"]:
                result["inputs_name"]["left"] = f'{name_generator(idx, 1000)}'
                result["inputs_name"]["right"] = f'{name_generator(idx, 1000)}'
                result["inputs_name"]["other"] = f'{name_generator(idx, 1000)}'
            else:
                result["inputs_name"]["left"] = sample["left"].name
                result["inputs_name"]["right"] = sample["right"].name
                result["inputs_name"]["other"] = sample["other"].name
            if self.save_inputs:
                result["inputs"]["left"] = np.uint8(sample_input["left"])
                result["inputs"]["right"] = np.uint8(sample_input["right"])
                result["inputs"]["other"] = np.uint8(sample_input["other"])
            ######### Preprocessing according the chosen Network ##############
            sample = self.preprocessing(sample)
            self.disparity_network.args.inference_size = self.preprocessing.inference_size
            self.disparity_network.args.ori_size = self.preprocessing.ori_size
            self.disparity_network.args.pred_right_disp = self.setup["pred_right_disp"]
            self.disparity_network.args.pred_bidir_disp = self.setup["pred_bidir_disp"]
            ######### Disparity Inference with the selected Network ##############
            pred_disp = self.disparity_network(sample["left"], sample["right"])
            if self.save_disp:
                if self.setup["pred_bidir_disp"]:
                    disp = pred_disp.cpu().numpy()
                    result["disp_left"] = vis_disparity(disp[0])
                    result["disp_right"] = vis_disparity(disp[1])

                else:
                    disp = pred_disp[0].cpu().numpy()
                    result["disp"] = vis_disparity(disp)
            ######## Reconstruction using the estimated disparity ###########
            image_reg, new_disp = self.reconstruction(pred_disp, sample)
            if self.reconstruction.method == "pytorch":
                result["image_reg"] = (image_reg.cpu().numpy() * 255.).astype("uint8")
                result["new_disp"] = vis_disparity(new_disp.cpu().numpy())
            else:
                if image_reg.max() <= 1:
                    result["image_reg"] = (image_reg * 255.).astype("uint8")
                else:
                    result["image_reg"] = image_reg.astype("uint8")
                result["new_disp"] = vis_disparity(new_disp)
            self.save_result(result)
        if self.timeit:
            self.save_timers()

    def save_result(self, result):
        if self.save_inputs:
            for key in result["inputs"].keys():
                path_input = os.path.join(self.path_output, "input", key)
                name = result["inputs_name"][key] + f'_{key}.png'
                res = cv.cvtColor(np.uint8(result["inputs"][key]), cv.COLOR_RGB2BGR)
                cv.imwrite(os.path.join(path_input, name), res)
        if self.save_disp:
            if self.setup["pred_bidir_disp"]:
                path_disp = os.path.join(self.path_output, "disp_left")
                name = result["inputs_name"]["left"] + '_disp_left.png'
                cv.imwrite(os.path.join(path_disp, name), result["disp_left"])
                path_disp = os.path.join(self.path_output, "disp_right")
                name = result["inputs_name"]["right"] + '_disp_right.png'
                cv.imwrite(os.path.join(path_disp, name), result["disp_right"])
            elif self.setup['pred_right_disp']:
                path_disp = os.path.join(self.path_output, "disp_right")
                name = result["inputs_name"]["left"] + '_right.png'
                cv.imwrite(os.path.join(path_disp, name), result["disp"])
            else:
                path_disp = os.path.join(self.path_output, "disp_left")
                name = result["inputs_name"]["left"] + '_disp_left.png'
                cv.imwrite(os.path.join(path_disp, name), result["disp"])
            path_disp = os.path.join(self.path_output, "disp_other")
            name = result["inputs_name"]["other"] + '_disp_other.png'
            cv.imwrite(os.path.join(path_disp, name), result["new_disp"])
        if self.save_reg_images:
            path = os.path.join(self.path_output, "reg_images")
            name = result["inputs_name"]["other"] + '_reg.png'
            cv.imwrite(os.path.join(path, name), result["image_reg"])

    def save_timers(self):
        dataloader_time = self.dataloader.timeit
        preprocessing_time = self.preprocessing.timeit
        disparity_network_time = self.disparity_network.timeit
        reconstruction_time = self.reconstruction.timeit
        time_dict = {"Sample Number": len(self.dataloader),
                     "Total execution time": round(
                         sum(dataloader_time + preprocessing_time + disparity_network_time + reconstruction_time), 3),
                     "Dataloader": {"Average time": round(sum(dataloader_time) / len(dataloader_time), 2),
                                    "Min time": round(min(dataloader_time), 3),
                                    "Max time": round(max(dataloader_time), 3),
                                    "Total time": round(sum(dataloader_time), 3)},
                     "Preprocessing": {"Average time": round(sum(preprocessing_time) / len(preprocessing_time), 3),
                                       "Min time": round(min(preprocessing_time), 3),
                                       "Max time": round(max(preprocessing_time), 3),
                                       "Total time": round(sum(preprocessing_time), 3)},
                     "Disparity network": {
                         "Average time": round(sum(disparity_network_time) / len(disparity_network_time), 3),
                         "Min time": round(min(disparity_network_time), 3),
                         "Max time": round(max(disparity_network_time), 3),
                         "Total time": round(sum(disparity_network_time), 3)},
                     "Reconstruction": {"Average time": round(sum(reconstruction_time) / len(reconstruction_time), 3),
                                        "Min time": round(min(reconstruction_time), 3),
                                        "Max time": round(max(reconstruction_time), 3),
                                        "Total time": round(sum(reconstruction_time), 3)}}
        name = os.path.join(self.path_output, "Execution_time.yaml")
        with open(name, "w") as file:
            yaml.dump(time_dict, file)

    @torch.no_grad()
    def _init_dataloader_(self, config):
        dataloader = StereoDataLoader(config, setup=self.setup)
        if self.print_info:
            print(f'\n'
                  f'############# DATASET ######'
                  f'\n', f'The dataset is composed of {2 * len(dataloader)} RGB images and {len(dataloader)} '
                         'infrared images' if config["dataset"]["type"] == '2vis' else
                  f'\nThe dataset is composed of {2 * len(dataloader)} infrared images and {len(dataloader)} RGB images')
        return dataloader

    @torch.no_grad()
    def _init_preprocessing_(self, config):
        return Preprocessing(config)

    @torch.no_grad()
    def _init_network_(self, config):
        if self.print_info:
            print(f'\n'
                  f'############# NETWORK ######')
        args = config["network_args"]
        self.network_name = config["name"]
        if self.network_name == "unimatch":
            model = SuperNetwork(UniMatch(feature_channels=args.feature_channels,
                                          num_scales=args.num_scales,
                                          upsample_factor=args.upsample_factor,
                                          num_head=args.num_head,
                                          ffn_dim_expansion=args.ffn_dim_expansion,
                                          num_transformer_layers=args.num_transformer_layers,
                                          reg_refine=args.reg_refine,
                                          task=args.task).to(self.device),
                                 args,
                                 name="unimatch",
                                 timeit=self.timeit)
        elif self.network_name == "avcNet":
            pass
        elif self.network_name == "custom":
            self.feature_extraction = self._initialize_features_extraction_(config)
            self.transformer = self._initialize_transformer_(config)
            self.detection_head = self._initialize_detection_head_(config)
            self.model = torch.nn.Sequential([self.feature_extraction, self.transformer, self.detection_head])
        if self.print_info:
            print(f'The model "{self.network_name}" has been initialized')
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model.model)
            model = model.model.module
        if self.print_info:
            count_parameter(model.model)
        if config["path_checkpoint"]:
            checkpoint = torch.load(config["path_checkpoint"], map_location=self.device_index)
            model.model.load_state_dict(checkpoint['model'], strict=True)
            if self.print_info:
                print(f"Load checkpoint: {config['path_checkpoint']}\n")
        return model

    @torch.no_grad()
    def _init_reconstruction_(self, config):
        if self.print_info:
            print(f'############# RECONSTRUCTION ######')
            print(f'Reconstruction method : {config["reconstruction"]["method"]}')
            if config['dataset']["pred_bidir_disp"]:
                print(f'The projection of disparity will use the both left and right disparity images')
            if config["reconstruction"]["method"] == 'fullOpencv':
                print(f'')
            if config['dataset']["type"] == '2vis':
                print(f"The infrared Image will be projected to the "
                      f"{'right' if self.setup['proj_right'] else 'left'} RGB image\n")
            elif config["always_proj_infrared"]:
                print(f"The {'right' if self.setup['proj_right'] else 'left'} "
                      f"infrared image will be projected to the RGB image\n")
            else:
                print(f"The RGB image will be projected to the {'right' if self.setup['proj_right'] else 'left'} "
                      f"infrared image\n")
        return ReconstructionBlock(config, self.setup)

    def _initialize_features_extraction_(self):
        return 0

    def _initialize_transformer_(self):
        return 0

    def _initialize_detection_head_(self):
        return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = ConfigPipe()
    pipe = Pipe(config)
    pipe.run()
    print('Done !')
