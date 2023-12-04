import os
import cv2 as cv
import numpy as np
import oyaml as yaml
import pandas as pd
from pandas import DataFrame
from pandas._typing import ArrayLike
from utils.classes import ImageTensor
from utils.classes.Visualizer import Visualizer
from utils.manipulation_tools import merge_dict


# from utils.classes.Image import ImageTensor


class ResultFrame:
    """
    A wrapper of the Dataframe class for the result analysis
    """

    def __init__(self, path: str, *args):
        direct, _, files = os.walk(path).__next__()
        data_val = []
        timer_val = []
        dataset = {'Files': {}}
        self._visu = Visualizer(path, search_exp=False)
        for file in files:
            with open(direct + '/' + file, "r") as f:
                val = yaml.safe_load(f)
            if file.split('.')[0] == 'Validation' or file.split('_')[0] == 'Validation':
                self.exp_name = list(val['2. results'].keys())[0]
                data = val['2. results'][self.exp_name]
                self.mask_outlier = None
                for key in data.keys():
                    data[key]['delta'] = np.array(data[key]['new']) / np.array(data[key]['ref']) - 1
                    # data[key]['delta'][np.array(data[key]['ref']) == 0] = 0.
                    data[key]['delta_occ'] = np.array(data[key]['new_occ']) / np.array(data[key]['ref']) - 1
                    # data[key]['delta_occ'][data[key]['ref'] == 0] = 0.
                    if self.mask_outlier is None:
                        self.mask_outlier = np.array(data[key]['ref']) <= np.quantile(np.array(data[key]['ref']), 0.01)
                    else:
                        self.mask_outlier = (np.array(data[key]['ref']) <= np.quantile(np.array(data[key]['ref']),
                                                                                       0.01)) + self.mask_outlier
                    data[key]['delta'], data[key]['delta_occ'] = data[key]['delta'].tolist(), data[key][
                        'delta_occ'].tolist()

                data_val.append(data)
            elif file.split('.')[0] == 'dataset':
                dataset = val
            elif file.split('.')[0] == 'Execution_time' or file.split('_')[0] == 'Execution':
                self.exp_name = list(val['3. Time per module'].keys())[0]
                data = val['3. Time per module'][self.exp_name]
                timer_val.append(data)
            else:
                pass
        if len(data_val) > 1:
            data_val = merge_dict(*data_val)
        else:
            data_val = data_val[0]
        if len(timer_val) > 1:
            timer_val = merge_dict(*timer_val)
        else:
            timer_val = timer_val[0]
        self.values = ValFrame(data_val)
        self.timer = TimeFrame(timer_val)
        self.dataset = dataset['Files']

    # def plot(self, index=None):

    def show(self, index=None, show_occ=False):
        ref = dict(self.values.T.ref)
        new = dict(self.values.T.new)
        occ = dict(self.values.T.new_occ)
        res = {}
        if index is None:
            res = {f'{key}_ref': v_ref for key, v_ref in ref.items()}
            res_new = {f'{key}_new': v_new for key, v_new in new.items()}
            res = res | res_new
            if show_occ:
                res_occ = {f'{key}_occ': v_occ for key, v_occ in occ.items()}
                res = res | res_occ
        else:
            if not isinstance(index, list):
                index = [index]
            for idx in index:
                if idx in ref.keys():
                    res = res | {f'{idx}_ref': ref[idx],
                                 f'{idx}_new': new[idx]}
                    if show_occ:
                        res[f'{idx}_occ'] = occ[idx]
        ValFrame(res).plot()

    @property
    def delta(self):
        return ValFrame(dict(self.values.T.delta))[~self.mask_outlier]

    @property
    def delta_occ(self):
        return ValFrame(dict(self.values.T.delta_occ))[~self.mask_outlier]

    @property
    def delta_full(self):
        return ValFrame(dict(self.values.T.delta))

    @property
    def delta_occ_full(self):
        return ValFrame(dict(self.values.T.delta_occ))


class TimeFrame(DataFrame):
    """
    A class based on Pandas Dataframe, part of the result Frame
    """
    def __init__(self, data):
        super().__init__(data)


class ValFrame(DataFrame):
    """
    A class based on Pandas Dataframe, part of the result Frame
    """

    def __init__(self, data):
        super().__init__(data)

    def __getitem__(self, key):
        return ValFrame({k: vec[key] for k, vec in self.items()})

    def show(self, index=None, ref=0):
        idx = {f'{k} delta': vec*100 for k, vec in self.items() if k == index}
        if f'{index} delta' in idx.keys():
            idx['0%'] = ref*idx[f'{index} delta']
            m = round(idx[f'{index} delta'].mean(), 3)
            idx[f'mean : {m}%'] = m + 0*idx[f'{index} delta']
        ValFrame(idx).plot()

    def get_column(self, index: str) -> ArrayLike:
        idx = np.array([vec * 100 for k, vec in self.items() if k == index])
        return idx

