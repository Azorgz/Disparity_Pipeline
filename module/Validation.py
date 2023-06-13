# from torch.fft import fft2, ifft2
import os

import torch
import yaml
from utils.classes import norms_dict, stats_dict, norms_dict_gpu
import numpy as np

from utils.misc import timeit, deactivated
from utils.transforms import ToNumpy, ToTensor


class Validation:
    """
    A class which compute for each input sample the reconstruction error
    """

    def __init__(self, config):
        self.activated = config["validation"]['activated']
        if self.activated:
            if config["timeit"]:
                self.timeit = []
            self.compare_smaller = config["validation"]['compare_smaller']
            self.device = config["device"]["device"]
            self.norms = {}
            self.ref = {}
            self.res = {}
            self.stats = {}
            self.res_stats = {}
            norms = config["validation"]['indices']
            self.method = config["validation"]['method']

            for key, value in norms.items():
                if value:
                    self.norms[key] = norms_dict[key] if self.method != 'pytorch' else norms_dict_gpu[key]
                    self.res[key] = []
            stats = config["validation"]['stats']
            for key, value in stats.items():
                if value:
                    self.stats[key] = stats_dict[key]
            print(f'############# VALIDATION ######',
                  f'\nThe validation of the results will compute the following indexes : {list(self.norms.keys())}',
                  f'\nThe Statistic selected for those indexes are the following : {list(self.stats.keys())}')

    @deactivated
    @timeit
    def __call__(self, sample, target, ref_image):
        if self.method != 'pytorch':
            if sample.max() <= 1:
                to_numpy = ToNumpy()
            else:
                to_numpy = ToNumpy(normalize=False)
            sample = to_numpy(sample)
        else:
            to_tensor = ToTensor()
            sample = sample.unsqueeze(0)
            target = to_tensor(target, self.device)
            ref_image = to_tensor(ref_image, self.device)
        for key, n in self.norms.items():
            res = n(target, sample, self.device).value
            ref = n(target, ref_image, self.device).value
            if self.method == 'pytorch':
                res, ref = res.cpu().numpy(), ref.cpu().numpy()
            self.res[key].append([round(float(res), 3), round(float(ref), 3)])

    @deactivated
    def statistic(self):
        for key_stat, stat in self.stats.items():
            self.res_stats[key_stat] = {}
            for key, res in self.res.items():
                s = stat(np.array(res), 0)
                res_stat = s[0]
                ref_stat = s[1]
                self.res_stats[key_stat][key] = float(round(res_stat, 3))
                self.res_stats[key_stat][key + "_ref"] = float(round(ref_stat, 3))

    @deactivated
    def save(self, path):
        name = os.path.join(path, "Validation.yaml")
        stat_dict = {key: np.array(item).tolist() for key, item in self.res_stats.items()}
        res_dict = {key + '_new/' + key + '_ref': np.array(item).tolist() for key, item in self.res.items()}
        # ref_dict = {key: np.array(item).tolist() for key, item in self.ref.items()}
        with open(name, "w") as file:
            yaml.dump({"2. results": res_dict, "1. stats": stat_dict}, file)


#
# def calculate_fft(im):
#     fft_im = fft2(im)
#     fft_amp = fft_im.real ** 2 + fft_im.imag ** 2
#     fft_amp = torch.sqrt(fft_amp)
#     fft_pha = torch.atan2(fft_im.imag, fft_im.real)
#     return fft_amp, fft_pha
#
#
# def calculate_ifft(fft_amp, fft_pha):
#     imag = fft_amp * torch.sin(fft_pha)
#     real = fft_amp * torch.cos(fft_pha)
#     fft_y = torch.complex(real, imag)
#     ifft = ifft2(fft_y)
#     im = torch.real(ifft)
#     return im
