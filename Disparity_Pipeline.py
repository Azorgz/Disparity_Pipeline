import os
import warnings
import oyaml as yaml
import torch
from tqdm import tqdm
# Config
from config.Config import ConfigPipe
# module
from module.DataLoader import StereoDataLoader
from module.ImageSaver import ImageSaver
from module.ImageWrapper import ImageWrapper
from module.Process import Process
from module.SuperNetwork import SuperNetwork
from module.Validation import Validation
from utils.misc import merge_dict
# Utils
from utils.misc import time2str, update_name_tree
from utils.ImagesCameras import CameraSetup


# Networks


class Pipe:
    """
    A class defining the pipe of processing of the chosen process
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
        self.modules = {}
        self._init_setup_()
        self._init_dataloader_()
        self._init_network_()
        self._init_wrapper_(verbose=False)
        self._init_saver_(verbose=False)
        self._init_validation_()
        # self.time_consistency_block = Time_consistency_block(config.time_consistency_block)

    @torch.no_grad()
    def run(self, process=None):
        if process is not None:
            process.init_process(self)
            if self.print_info:
                print(process)
            for name_experiment, experiment in process.items():
                self.save_experiment(experiment)
                with tqdm(total=len(self.dataloader),
                          desc=f"Nombre d'itérations for {name_experiment}: ", leave=True, position=0) as bar:
                    name = None
                    if experiment.setup is not self.setup:
                        self._init_dataloader_(experiment.setup)
                        self._init_wrapper_(experiment.setup, verbose=False)
                    self.dataloader.camera_used = process.camera_used
                    for idx, sample in enumerate(self.dataloader):
                        update_name_tree(sample, experiment.setup.name)
                        experiment(sample, setup=experiment.setup, wrapper=self.wrapper)
                        bar.update(1)
                if self.timeit:
                    self.save_timers(experiment, name)
                    self.reset_timers()
                if self.save_inputs:
                    self.dataloader.dataset.save_conf(experiment.path)
                self.validation.statistic()
                self.validation.save(experiment.path)
                self.validation.reset()
                if experiment.setup is not self.setup:
                    self._init_dataloader_(self.setup)
                    self._init_wrapper_(self.setup, verbose=False)

        else:
            for idx, sample in tqdm(enumerate(self.dataloader),
                                    total=len(self.dataloader),
                                    desc="Nombre d'itérations : "):
                # Preparation of the result dictionary ################
                # result = self.init_result(sample, idx)
                pred_depth = {}
                # # Disparity Inference with the selected Network ##############
                self.network.update_pred_bidir(activate=True)
                # pred_disp = self.network(sample, intrinsics, pose, depth=True)
                new_sample = self.setup.stereo_pair('RGB', 'RGB2')(sample, cut_roi_max=True)
                pred_disp = self.network(new_sample)
                pred_depth.update(self.setup.stereo_pair('RGB', 'RGB2')(pred_disp, reverse=True))
                # pred_depth['RGB'].show()
                # (pred_disp['RGB'] * sample['RGB']).show()
                # (pred_disp['RGB2'] * sample['RGB2']).show()

                # Reconstruction using the estimated disparity ###########
                image_reg_disp = self.wrapper(pred_depth, sample, 'IR', 'RGB', 'RGB2', depth=False)
                image_reg_depth = self.wrapper(pred_depth, sample, 'IR', 'RGB', depth=True)
                (image_reg_disp * 0.5 + sample['RGB'] * 0.5).hstack(image_reg_depth * 0.5 + sample['RGB'] * 0.5).show()
            if self.timeit:
                self.save_timers()

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

    def save_experiment(self, experiment):
        path = experiment.path
        if path is None:
            path = self.path_output
        name = os.path.join(path, f"Summary_experiment.yaml")
        summary = experiment.summary()
        with open(name, "w") as file:
            yaml.dump(summary, file)

    def save_timers(self, experiment=None, filename=None, replace=False):
        path = experiment.path
        time_dict = {"1. Sample Number": len(self.dataloader),
                     "2. Total Execution time": {experiment.name: "0"},
                     "3. Time per module": {experiment.name: {}}}
        tot = 0
        for m in self.modules.values():
            name = m.__class__.__name__
            time = m.timeit
            if time:
                time_dict["3. Time per module"][experiment.name][name] = {
                    "Number of calls": len(time),
                    "Average time": time2str(sum(time) / (len(time) + 0.00001)),
                    "Min time": time2str(min(time)),
                    "Max time": time2str(max(time)),
                    "Total time": time2str(sum(time))}
                tot += sum(time)
        time_dict["2. Total Execution time"][experiment.name] = time2str(tot)
        if path is None:
            path = self.path_output
        if filename is not None:
            name = os.path.join(path, f"Execution_time_{filename}.yaml")
        else:
            name = os.path.join(path, "Execution_time.yaml")
        if replace:
            try:
                with open(name, 'r') as file:
                    timer = yaml.safe_load(file)
                time_dict = merge_dict(timer, time_dict)
            except FileNotFoundError:
                pass
        with open(name, "w") as file:
            yaml.dump(time_dict, file)

    def reset_timers(self):
        for m in self.modules.values():
            m.timeit = []

    @torch.no_grad()
    def _init_setup_(self):
        self.setup = CameraSetup(from_file=self.config["setup"]['path'], device=self.device,
                                 max_depth=self.config["setup"]['max_depth'],
                                 min_depth=self.config["setup"]['min_depth'])

    @torch.no_grad()
    def _init_dataloader_(self, setup: CameraSetup = None):
        setup = self.setup if setup is None else setup
        self.dataloader = StereoDataLoader(setup, self.config)
        self.modules['dataloader'] = self.dataloader

    @torch.no_grad()
    def _init_network_(self, *args, **kwargs):
        self.network = SuperNetwork(self.config, *args, **kwargs)
        self.modules['network'] = self.network

    @torch.no_grad()
    def _init_wrapper_(self, *args, setup: CameraSetup = None, **kwargs):
        setup = self.setup if setup is None else setup
        self.wrapper = ImageWrapper(self.config, setup, *args, **kwargs)
        self.modules['wrapper'] = self.wrapper

    def _init_saver_(self, *args, **kwargs):
        self.saver = ImageSaver(self.config, *args, **kwargs)
        self.modules['saver'] = self.saver

    @torch.no_grad()
    def _init_validation_(self, *args, **kwargs):
        self.validation = Validation(self.config, *args, **kwargs)
        self.modules['validation'] = self.validation

    # def _initialize_features_extraction_(self):
    #     return 0
    #
    # def _initialize_transformer_(self):
    #     return 0
    #
    # def _initialize_detection_head_(self):
    #     return 0


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    Process = Process(os.getcwd() + '/Process_resolution.yaml')
    config = ConfigPipe(Process.option)
    pipe = Pipe(config)
    pipe.run(Process)
    print('Done !')
