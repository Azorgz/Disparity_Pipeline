# from torch.fft import fft2, ifft2
import glob
import os

import oyaml
import torch
from kornia.morphology import erosion
from tqdm import tqdm

from module.BaseModule import BaseModule
from utils.ImagesCameras import ImageTensor
from utils.ImagesCameras.Metrics import stats_dict, norms_dict
import numpy as np

from utils.ImagesCameras.tools.drawing import extract_roi_from_images
from utils.misc import merge_dict
from utils.misc import timeit, deactivated


class Validation(BaseModule):
    """
    A class which compute for each input sample the reconstruction error
    """

    def __init__(self, config, *args, **kwargs):
        """
        :param config: Config from ConfigPipe
        the module compute two indices for each norm chosen:
        - The "ref" : between Ref and Old
        - The "new" : between the Ref and New
        """
        self.activated = False
        super(Validation, self).__init__(config, *args, **kwargs)

    def _update_conf(self, config, *args, **kwargs):
        self.activated = config["validation"]['activated']
        if self.activated:
            self.mask_cum = None
            self.roi = []
            self.total_roi = None
            self.norms = {}
            self.res = {}
            self.stats = {}
            self.res_stats = {}
            self.post_validation = config["validation"]['post_validation']
            self.exp_name = None
            self.path = config["output_path"]
            norms = config["validation"]['indices']
            for key, value in norms.items():
                if value:
                    self.norms[key] = norms_dict[key](self.device)
            stats = config["validation"]['stats']
            for key, value in stats.items():
                if value:
                    self.stats[key] = stats_dict[key]

    def __str__(self):
        if self.activated:
            string = super().__str__()
            string += f'The validation of the results will compute the following indexes : {list(self.norms.keys())}'
            string += f'\nThe Statistic selected for those indexes are the following : {list(self.stats.keys())}'
            return string

    @deactivated
    @timeit
    def __call__(self, new, ref, old, name, exp_name, *args, occlusion=None, roi=None, cum_roi=None, **kwargs):
        self.exp_name = exp_name
        mask = new.BINARY(threshold=0, method='gt', keepchannel=False)
        self.roi.append(extract_roi_from_images(mask)[0])
        if not self.post_validation:
            if name not in self.res.keys():
                self.res[name] = {}
            for key, n in self.norms.items():
                res = {}
                if key not in self.res[name].keys():
                    self.res[name][key] = {}
                # Compute the indices of each images cut to ROI
                if roi is not None:
                    new_roi = ImageTensor(new[:, :, roi[2]:roi[3], roi[0]:roi[1]])
                    ref_roi = ImageTensor(ref[:, :, roi[2]:roi[3], roi[0]:roi[1]])
                    old_roi = ImageTensor(old[:, :, roi[2]:roi[3], roi[0]:roi[1]])
                    mask = None
                    res_new = n(ref_roi, new_roi, mask=mask)
                    res_old = n(ref_roi, old_roi, mask=mask)
                    res.update({'ref_roi': round(float(res_old), 4),
                                'new_roi': round(float(res_new), 4)})
                # Compute the indices of each images cut to the GLOBAL ROI
                if cum_roi is not None:
                    new_cumroi = ImageTensor(new[:, :, cum_roi[2]:cum_roi[3], cum_roi[0]:cum_roi[1]])
                    ref_cumroi = ImageTensor(ref[:, :, cum_roi[2]:cum_roi[3], cum_roi[0]:cum_roi[1]])
                    old_cumroi = ImageTensor(old[:, :, cum_roi[2]:cum_roi[3], cum_roi[0]:cum_roi[1]])
                    mask = None
                    res_new = n(ref_cumroi, new_cumroi, mask=mask)
                    res_old = n(ref_cumroi, old_cumroi, mask=mask)
                    res.update({'ref_cumroi': round(float(res_old), 4),
                                'new_cumroi': round(float(res_new), 4)})
                # Compute the indices of each images without cut to ROI  but with a mask of pixel > 0
                mask = new.BINARY(threshold=0, method='gt', keepchannel=False)
                res_new = n(ref, new, mask=mask)
                res_old = n(ref, old, mask=mask)
                res.update({'ref': round(float(res_old), 4),
                            'new': round(float(res_new), 4)})
                # Compute the indices of each images without cut to ROI  but with a mask of pixel > 0  + occlusion
                if occlusion is not None:
                    res_occlusion = n(ref, new, mask=~occlusion)
                    ref_occlusion = n(ref, old, mask=~occlusion)
                    res.update({'new_occ': round(float(res_occlusion), 4)})
                    res.update({'ref_occ': round(float(ref_occlusion), 4)})
                if self.res[name][key] == {}:
                    self.res[name][key] = res
                else:
                    self.res[name][key] = merge_dict(self.res[name][key], res)

    @deactivated
    def statistic(self, path=None):
        if self.post_validation:
            roi = np.array(self.roi)
            total_left_top = roi.max(axis=0).tolist()
            total_right_bot = roi.min(axis=0).tolist()
            self.total_roi = [total_left_top[0], total_right_bot[1], total_left_top[2], total_right_bot[3]]
            path = self.path if path is None else path
            self._post_validation(path)
        else:
            for key, norms in self.res.items():
                self.res_stats[key] = {}
                for key_norms, n in norms.items():
                    self.res_stats[key][key_norms] = {}
                    for key_stat, stat in self.stats.items():
                        sample = self.res[key][key_norms]
                        if isinstance(sample[list(sample.keys())[0]], list):
                            self.res_stats[key][key_norms][key_stat] = {
                                key: float(stat(np.array(sample[key]), 0).round(3)) for key in sample.keys()}
                        else:
                            self.res_stats[key][key_norms][key_stat] = {key: float(np.array(sample[key]).round(3)) for
                                                                        key in sample.keys()}

    def reset(self):
        self._update_conf(self.config)

    @deactivated
    def save(self, name=None, path=None):
        path = self.path if path is None else path
        if self.post_validation:
            res = {"ROI": self.roi}
            if self.total_roi is not None:
                res.update({"cum_ROI": self.total_roi})
            name = os.path.join(path, "CumMask.yaml")
        else:
            stat_dict = {key: np.array(item).tolist() for key, item in self.res_stats.items()}
            res_dict = {key: np.array(item).tolist() for key, item in self.res.items()}
            res = {"1. stats": stat_dict, "2. results": res_dict}
            if name is None:
                name = os.path.join(path, "Validation.yaml")
            else:
                name = os.path.join(path, name)
                with open(name, 'r') as file:
                    validation = oyaml.safe_load(file)
                res = merge_dict(validation, res)
        with open(name, "w") as file:
            oyaml.dump(res, file)

    def _post_validation(self, path):
        self.save(path=path)
        with open(path + '/Summary_experiment.yaml', 'r') as file:
            summary = oyaml.safe_load(file)
        cam_src = summary["Wrap"]["cam_src"]
        cam_dst = summary["Wrap"]["cam_dst"]
        with open(path + '/dataset.yaml', 'r') as file:
            dataset = oyaml.safe_load(file)
        nb_sample = int(dataset["Number of sample"])
        input_src = sorted(dataset["Files"][cam_src])
        input_dst = sorted(dataset["Files"][cam_dst])
        self.post_validation = False

        def valid(s, **kwargs):
            name = f'{cam_src}_to_{cam_dst}'
            im_ref = s[cam_dst]
            image_reg = s['image_reg'].match_shape(im_ref)
            im_old = s[cam_src].match_shape(im_ref)
            occlusion = sample['occ'].BINARY(keepchannel=False) if 'occ' in s.keys() else None
            index = s['idx']
            self(image_reg, im_ref, im_old, name, path.split('/')[-1],
                 roi=self.roi[index], cum_roi=self.total_roi, occlusion=occlusion)

        with tqdm(total=nb_sample,
                  desc=f"Nombre d'itérations for {path.split('/')[-1]} - Validation : ",
                  leave=True, position=0) as bar:
            source = [input_src, input_dst]
            keys = [cam_src, cam_dst]
            if os.path.exists(path + f'/image_reg/{cam_src}_to_{cam_dst}'):
                source.append(sorted(glob.glob(path + f'/image_reg/{cam_src}_to_{cam_dst}/*')))
                keys.append('image_reg')
            if os.path.exists(path + f'/occlusion/{cam_src}_to_{cam_dst}'):
                source.append(sorted(glob.glob(path + f'/image_reg/{cam_src}_to_{cam_dst}/*')))
                keys.append('occ')
            for idx, src in enumerate(zip(*source)):
                sample = {key: ImageTensor(s, device=self.device) for (key, s) in zip(keys, src)}
                sample["idx"] = idx
                valid(sample)
                bar.update(1)
        self.statistic(path)

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
