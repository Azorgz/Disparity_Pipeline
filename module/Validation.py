# from torch.fft import fft2, ifft2
import os

import torch
import yaml

from module.BaseModule import BaseModule
from utils.classes import stats_dict, norms_dict
import numpy as np

from utils.manipulation_tools import merge_dict
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
            self.res = {}
            self.stats = {}
            self.res_stats = {}
            self.exp_name = None
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
    def __call__(self, new, ref, old, name, exp_name, *args, occlusion=None, **kwargs):
        self.exp_name = exp_name
        if name not in self.res.keys():
            self.res[name] = {}
        for key, n in self.norms.items():
            if key not in self.res[name].keys():
                self.res[name][key] = {}
            mask = new > 0
            res_new = n(ref, new, mask=mask)
            res_old = n(ref, old, mask=mask)
            if occlusion is not None:
                res_occlusion = n(ref, new, mask=~occlusion)
            else:
                res_occlusion = -1
            res = {'ref': round(float(res_old), 3),
                   'new': round(float(res_new), 4),
                   'new_occ': round(float(res_occlusion), 4)}
            if self.res[name][key] == {}:
                self.res[name][key] = res
            else:
                self.res[name][key] = merge_dict(self.res[name][key], res)

    @deactivated
    def statistic(self):
        for key, norms in self.res.items():
            self.res_stats[key] = {}
            for key_norms, n in norms.items():
                self.res_stats[key][key_norms] = {}
                for key_stat, stat in self.stats.items():
                    sample = self.res[key][key_norms]
                    res_stat = stat(np.array(sample['new']), 0).round(3)
                    ref_stat = stat(np.array(sample['ref']), 0).round(3)
                    occlusion_stat = stat(np.array(sample['new_occ']), 0).round(3)
                    self.res_stats[key][key_norms][key_stat] = {'ref': float(ref_stat),
                                                                'new': float(res_stat),
                                                                'new_occ': float(occlusion_stat)}

    def reset(self):
        self._update_conf(self.config)

    @deactivated
    def save(self, path, name=None):
        stat_dict = {key: np.array(item).tolist() for key, item in self.res_stats.items()}
        res_dict = {key: np.array(item).tolist() for key, item in self.res.items()}
        res = {"1. stats": stat_dict, "2. results": res_dict}
        if name is None:
            name = os.path.join(path, "Validation.yaml")
        else:
            name = os.path.join(path, name)
            with open(name, 'r') as file:
                validation = yaml.safe_load(file)
            res = merge_dict(validation, res)
        with open(name, "w") as file:
            yaml.dump(res, file)
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
