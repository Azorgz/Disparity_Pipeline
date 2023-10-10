# from torch.fft import fft2, ifft2
import os

import torch
import yaml

from module.BaseModule import BaseModule
from utils.classes import stats_dict, norms_dict
import numpy as np

from utils.misc import timeit, deactivated
from utils.transforms import ToTensor


class Validation(BaseModule):
    """
    A class which compute for each input sample the reconstruction error
    """

    def __init__(self, config):
        """
        :param config: Config from ConfigPipe
        the module compute two indices for each norm chosen:
        - The "ref" : between Ref and Old
        - The "new" : between the Ref and New
        """
        self.activated = False
        super(Validation, self).__init__(config)

    def _update_conf(self, config, *args, **kwargs):
        self.activated = config["validation"]['activated']
        if self.activated:
            self.norms = {}
            self.ref = {}
            self.res = {}
            self.stats = {}
            self.res_stats = {}
            norms = config["validation"]['indices']
            for key, value in norms.items():
                if value:
                    self.norms[key] = norms_dict[key](self.device)
            stats = config["validation"]['stats']
            for key, value in stats.items():
                if value:
                    self.stats[key] = stats_dict[key]

    def __str__(self):
        string = super().__str__()
        string += f'The validation of the results will compute the following indexes : {list(self.norms.keys())}'
        string += f'\nThe Statistic selected for those indexes are the following : {list(self.stats.keys())}'
        return string

    @deactivated
    @timeit
    def __call__(self, new, ref, old, name, *args, **kwargs):
        if name not in self.res.keys():
            self.res[name] = {}
        for key, n in self.norms.items():
            if key not in self.res[name].keys():
                self.res[name][key] = []
            res_new = n(ref, new)
            res_old = n(ref, old)
            self.res[name][key].append([round(float(res_new), 3), round(float(res_old), 3)])

    @deactivated
    def statistic(self):

        for key, norms in self.res.items():
            self.res_stats[key] = {}
            for key_norms, n in norms.items():
                self.res_stats[key][key_norms] = {}
                for key_stat, stat in self.stats.items():
                    sample = self.res[key][key_norms]
                    s = stat(np.array(sample), 0)
                    res_stat = s[0]
                    ref_stat = s[1]
                    self.res_stats[key][key_norms][key_stat] = [float(round(res_stat, 3)), float(round(ref_stat, 3))]

    def reset(self):
        self._update_conf(self.config)

    @deactivated
    def save(self, path):
        name = os.path.join(path, "Validation.yaml")
        stat_dict = {key: np.array(item).tolist() for key, item in self.res_stats.items()}
        res_dict = {key: np.array(item).tolist() for key, item in self.res.items()}
        # ref_dict = {key: np.array(item).tolist() for key, item in self.ref.items()}
        with open(name, "w") as file:
            yaml.dump({"2. results": res_dict, "1. stats": stat_dict}, file)
        self.activated = False


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
