import os
import numpy as np
import oyaml as yaml

from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas._typing import ArrayLike
from utils.misc import merge_dict, extract_key_pairs, add_ext, create_function_from_string


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
        # self._visu = Visualizer(path, search_exp=False)
        for file in files:
            with open(direct + '/' + file, "r") as f:
                val = yaml.safe_load(f)
            if file.split('.')[0] == 'Validation' or file.split('_')[0] == 'Validation':
                self.exp_name = list(val['2. results'].keys())[0]
                data = val['2. results'][self.exp_name]
                self.mask_outlier = None
                for key in data.keys():
                    self.available_res = []
                    for ext, new, ref in extract_key_pairs(data[key]):
                        data[key][add_ext('delta', ext)] = np.array(data[key][new]) / (
                                    np.array(data[key][ref]) + 1e-6) - 1
                        data[key][add_ext('delta', ext)][np.array(data[key][ref]) == 0] = 0.
                        self.available_res.append(add_ext('delta', ext))
                        if self.mask_outlier is None:
                            lim = np.array(data[key][ref]).std() / 2
                            self.mask_outlier = lim > np.array(data[key][ref])
                        else:
                            lim = np.array(data[key][ref]).std() / 2
                            self.mask_outlier = (lim > np.array(data[key][ref])) + self.mask_outlier
                        data[key][add_ext('delta', ext)] = data[key][add_ext('delta', ext)].tolist()
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

    # def create_properties(self):
    #     for ext in self.available_res:
    #         name = ext
    #         func_code = f"""return ValFrame(dict(self.values.T.{ext}))[~self.mask_outlier]"""
    #         create_function_from_string(name, func_code, prop=True, self=self)
    #         name = f'{ext}_full'
    #         func_code = f"""return ValFrame(dict(self.values.T.{ext}))"""
    #         create_function_from_string(name, func_code, prop=True)

    def show(self, index=None, show_occ=False, show_roi=False, show_cumroi=False, save='', dpi=300):

        if 'delta_occ' in self.available_res and show_occ:
            ref_occ = dict(self.values.T.ref_occ)
            new_occ = dict(self.values.T.new_occ)
        else:
            show_occ = False
        if 'delta_roi' in self.available_res and show_roi:
            ref_roi = dict(self.values.T.ref_roi)
            new_roi = dict(self.values.T.new_roi)
        else:
            show_roi = False
        if 'delta_cumroi' in self.available_res and show_cumroi:
            ref_cumroi = dict(self.values.T.ref_cumroi)
            new_cumroi = dict(self.values.T.new_cumroi)
        else:
            show_cumroi = False
        res = {}
        if index is None:
            if show_occ:
                res_occ = ({f'{key}_ref': v_ref for key, v_ref in ref_occ.items()} |
                           {f'{key}_occ': v_occ for key, v_occ in new_occ.items()})
                res = res | res_occ
                title = 'Delta with occlusion'
            elif show_roi:
                res_roi = ({f'{key}_ref': v_ref for key, v_ref in ref_roi.items()} |
                           {f'{key}_roi': v_roi for key, v_roi in new_roi.items()})
                res = res | res_roi
                title = 'Delta cut to roi'
            elif show_cumroi:
                res_cumroi = ({f'{key}_ref': v_ref for key, v_ref in ref_cumroi.items()} |
                              {f'{key}_cumroi': v_cumroi for key, v_cumroi in new_cumroi.items()})
                res = res | res_cumroi
                title = 'Delta cut to cumulative roi'
            else:
                ref = dict(self.values.T.ref)
                new = dict(self.values.T.new)
                res_new = ({f'{key}_ref': v_ref for key, v_ref in ref.items()} |
                           {f'{key}_new': v_new for key, v_new in new.items()})
                res = res | res_new
                title = 'Delta'
        else:
            if not isinstance(index, list):
                index = [index]
            ref = dict(self.values.T.ref)
            new = dict(self.values.T.new)
            for idx in index:
                if idx in ref.keys():
                    if show_occ:
                        res = res | {f'{idx} ref': ref_occ[idx],
                                     f'{idx} new': new_occ[idx]}
                        title = 'Delta with occlusion'
                    elif show_roi:
                        res = res | {f'{idx} ref': ref_roi[idx],
                                     f'{idx} new': new_roi[idx]}
                        title = 'Delta cut to roi'
                    elif show_cumroi:
                        res = res | {f'{idx} ref': ref_cumroi[idx],
                                     f'{idx} new': new_cumroi[idx]}
                        title = 'Delta cut to cumulative roi'
                    else:
                        res = res | {f'{idx} ref': ref[idx],
                                     f'{idx} new': new[idx]}
                        title = 'Delta'
        fig = ValFrame(res).plot(title=title).get_figure()
        if save:
            fig.savefig(f'{save}.png', bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
        return fig

    @property
    def delta(self):
        if 'delta' in self.available_res:
            return ValFrame(dict(self.values.T.delta))[~self.mask_outlier]
        return None

    @property
    def delta_occ(self):
        if 'delta_occ' in self.available_res:
            return ValFrame(dict(self.values.T.delta_occ))[~self.mask_outlier]
        return None

    @property
    def delta_full(self):
        if 'delta' in self.available_res:
            return ValFrame(dict(self.values.T.delta))
        return None

    @property
    def delta_occ_full(self):
        if 'delta_occ' in self.available_res:
            return ValFrame(dict(self.values.T.delta_occ))
        return None

    @property
    def delta_roi(self):
        if 'delta_roi' in self.available_res:
            return ValFrame(dict(self.values.T.delta_roi))
        return None

    @property
    def delta_roi_full(self):
        if 'delta_occ' in self.available_res:
            return ValFrame(dict(self.values.T.delta_roi_full))
        return None

    @property
    def delta_cumroi(self):
        if 'delta_cumroi' in self.available_res:
            return ValFrame(dict(self.values.T.delta_cumroi))
        return None

    @property
    def delta_cumroi_full(self):
        if 'delta_cumroi' in self.available_res:
            return ValFrame(dict(self.values.T.delta_cumroi_full))
        return None

    def display(self):
        self._visu.run()


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

    def show(self, *args, title=None, index=None, ref=0, save: str = '', dpi=300, **kwargs):
        if index is not None:
            idx = {f'{k} delta': vec * 100 for k, vec in self.items() if k == index}
            if title is None:
                title = f'Delta {" - ".join([k for k in self.keys() if k == index])}'
        else:
            idx = {f'{k} delta': vec * 100 for k, vec in self.items()}
            if title is None:
                title = 'Delta'
        if f'{index} delta' in idx.keys():
            m = round(idx[f'{index} delta'].mean(), 3)
            idx[f'mean : {m}%'] = m + 0 * idx[f'{index} delta']
        if ref is not None:
            idx['0%'] = ref * list(idx.values())[0]
        fig = ValFrame(idx).plot(*args, title=title, **kwargs).get_figure()
        if save:
            fig.savefig(f'{save}.png', bbox_inches='tight', dpi=dpi)
            plt.close(fig)
        return fig

    def get_column(self, index: str) -> ArrayLike:
        idx = np.array([vec for k, vec in self.items() if k == index])[0]
        return idx

    def combine_column(self, str_combination: str, name=None):
        """
        Combine the column of the ValFrame following this layout :
        formula(index)
        """
        val_dict = {}
        name = str_combination if name is None else name
        for key in self.keys():
            if str(key) in str_combination:
                locals()[key] = np.nan_to_num(self.get_column(key))
        return ValFrame({name: eval(str_combination)})
