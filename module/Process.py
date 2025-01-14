import inspect
import ntpath
import os
from collections import OrderedDict
from types import FrameType
from typing import cast

import numpy as np
import oyaml as yaml
import torch
from utils.ImagesCameras import CameraSetup, ImageTensor
from utils.misc import path_leaf


class Process(OrderedDict):
    """
    This class is the backbone of any pipeline of processing you want to make.
    It's an Ordered Dict which contain all the experiments the Pipe will process
    """

    def __init__(self, path: str = None, process_dict: dict = None):
        super(Process, self).__init__()
        self.isInit = False
        self.path = path
        self.camera_used = []
        self.option = {}
        assert path is not None or process_dict is not None
        if process_dict is not None:
            self.path = os.getcwd()
            self._create_from_dict(process_dict)
        else:
            self._create_from_path(path)

    def __call__(self, sample, *args, **kwargs):
        for experiment in self.values():
            experiment(sample)

    def __str__(self):
        if self.isInit:
            string = f'\n############### {self.name} #################'
            for experiment in self.values():
                string += f'{experiment.__str__()}'
        else:
            string = self.name
        return string

    def _create_from_dict(self, process_dict: dict) -> None:
        self.name = "New_process_from_dict"
        for key, p in process_dict.items():
            if key.upper() == 'OPTION':
                self.option = p
            elif key.upper() == 'NAME':
                self.name = p
            else:
                self[key] = self._make_process_dict_(p)

    def _create_from_path(self, path: str) -> None:
        with open(path, 'r') as file:
            process = yaml.safe_load(file)
        self.name = ntpath.basename(path).split('.')[0]
        for key, p in process.items():
            if key.upper() == 'OPTION':
                self.option = p
            elif key.upper() == 'NAME':
                self.name = p
            else:
                self[key] = self._make_process_dict_(p)

    def init_process(self, pipe) -> None:
        self.isInit = True
        # if len(pipe.setup) > 1:
        #     setup = CameraSetup(from_file=pipe.setup[0], device=pipe.device)
        # else:
        #     [0]
        path_result = pipe.path_output
        if not os.path.exists(path_result):
            os.makedirs(path_result, exist_ok=True)
            os.chmod(path_result, 0o777)
        for Exp, p in self.items():
            if p[0][0] == 'SETUP':
                setup = CameraSetup(from_file=p[0][1], device=pipe.device)
                p.pop(0)
            else:
                setup = pipe.setup
            process = [setup]
            first_save = True
            Exp = pipe.name_experiment if Exp[:3] == 'tbd' else Exp
            path = path_result + f'/{Exp}'
            pred = {}
            res = {'image_reg': [], 'depth_reg': []}
            for idx, instruction in enumerate(p):
                key, val = instruction[0], instruction[1]
                if key == 'DISPARITY':
                    assert val['cam1'] in setup.cameras and val['cam2'] in setup.cameras, \
                        f'The chosen cameras for disparity dont exist, available cameras are :{setup.cameras.keys()}'
                    assert setup.disparity_ready(val['cam1'], val['cam2']), \
                        'The chosen cameras for disparity are not stereo rectified'
                    name_var = 'pred_depth'
                    if val['pred_bidir']:
                        cam = [val['cam1'], val['cam2']]
                    elif val['pred_right']:
                        cam = [setup.stereo_pair(val['cam1'], val['cam2']).right.id]
                    else:
                        cam = [setup.stereo_pair(val['cam1'], val['cam2']).left.id]
                    if name_var in pred.keys():
                        pred[name_var].append(*cam)
                    else:
                        pred.update({name_var: cam})
                    if 'inference_size' not in val.keys():
                        val['inference_size'] = None
                    val['setup'] = setup
                    process.append(self.disparity(pipe, **val))
                    if val['cam1'] not in self.camera_used:
                        self.camera_used.append(val['cam1'])
                    if val['cam2'] not in self.camera_used:
                        self.camera_used.append(val['cam2'])
                elif key == 'DEPTH':
                    assert val['cam1'] in setup.cameras.keys() and val['cam2'] in setup.cameras.keys(), \
                        f'The chosen cameras for depth dont exist, available cameras are :{setup.cameras.keys()}'
                    assert setup.depth_ready(val['cam1']) and setup.depth_ready(val['cam2']), \
                        'The chosen cameras for depth are not positioned'
                    assert setup.depth_pair(val['cam1'], val['cam2']) != -1, \
                        'The chosen cameras are not depth ready'
                    name_var = 'pred_depth'
                    if val['pred_bidir']:
                        cam = [val['cam1'], val['cam2']]
                    elif val['pred_right']:
                        cam = [setup.depth_pair(val['cam1'], val['cam2']).target.id]
                    else:
                        cam = [setup.depth_pair(val['cam1'], val['cam2']).ref.id]
                    if name_var in pred.keys():
                        pred[name_var].append(*cam)
                    else:
                        pred.update({name_var: cam})
                    if 'inference_size' not in val.keys():
                        val['inference_size'] = None
                    val['setup'] = setup
                    process.append(self.depth(pipe, **val))
                    if val['cam1'] not in self.camera_used:
                        self.camera_used.append(val['cam1'])
                    if val['cam2'] not in self.camera_used:
                        self.camera_used.append(val['cam2'])
                elif key == 'MONOCULAR':
                    for cam in val['cams']:
                        assert cam in setup.cameras.keys(), \
                            f'The chosen cameras for depth dont exist, available cameras are :{setup.cameras.keys()}'
                        if cam not in self.camera_used:
                            self.camera_used.append(cam)
                    if 'pred_depth' in pred.keys():
                        pred['pred_depth'].append(*val['cams'])
                    else:
                        pred.update({'pred_depth': val['cams']})
                    if 'inference_size' not in val.keys():
                        val['inference_size'] = None
                    val['setup'] = setup
                    process.append(self.monocular_depth(pipe, **val))
                elif key == 'WRAP':
                    if val['depth']:
                        assert setup.depth_ready(val['cam_src']) and setup.depth_ready(val['cam_dst']), \
                            'The chosen cameras for wrap are not positioned'
                    else:
                        assert setup.disparity_ready(val['cam_src'], val['cam_dst']), \
                            'The chosen cameras for wrap are not disparity ready'
                    assert val['cam_dst'] in pred['pred_depth'] or val['cam_src'] in pred['pred_depth'], \
                        'The chosen tensor doesnt contain the depth of the chosen camera for wrap'
                    cam_to_cam = f'{val["cam_src"]}_to_{val["cam_dst"]}'
                    if val['cam_dst'] not in pred['pred_depth']:
                        val['reverse_wrap'] = True
                    else:
                        val['reverse_wrap'] = False
                    if val['return_depth_reg']:
                        res['depth_reg'].append(val["cam_src" if not val['reverse_wrap'] else "cam_dst"])
                        if 'depth_reg' in pred.keys():
                            pred['depth_reg'].append(val["cam_src" if not val['reverse_wrap'] else "cam_dst"])
                        else:
                            pred.update({'depth_reg': [val["cam_src" if not val['reverse_wrap'] else "cam_dst"]]})
                    if val['return_occlusion']:
                        if 'occlusion' in res.keys():
                            res['occlusion'].append(cam_to_cam)
                        else:
                            res.update({'occlusion': [cam_to_cam]})
                    res['image_reg'].append(cam_to_cam)
                    process.append(self.wrap(pipe, **val))
                    if val['cam_src'] not in self.camera_used:
                        self.camera_used.append(val['cam_src'])
                    if val['cam_dst'] not in self.camera_used:
                        self.camera_used.append(val['cam_dst'])
                elif key == 'VALID':
                    assert val['cam_reg'] in setup.cameras.keys(), 'The chosen cam_reg for validation doesnt exist'
                    assert val['cam_ref'] in setup.cameras.keys(), 'The chosen cam_ref for validation doesnt exist'
                    name = f'{val["cam_reg"]}_to_{val["cam_ref"]}'
                    assert name in res['image_reg'], \
                        f'You have to wrap the image before to valid the result. ' \
                        f'{name} is not in the list of the reg images'
                    val['exp_name'] = Exp
                    process.append(self.valid(pipe, **val))
                elif key == 'SAVE':
                    if first_save:
                        first_save = False
                        if not os.path.exists(path):
                            os.mkdir(path)
                            os.chmod(path, 0o777)
                        elif os.listdir(path):
                            resp = input(
                                f'The specified output path ({path}) is not empty, do we clear the data? (y/n)')
                            if resp == "y" or resp == "Y":
                                from utils.misc import clear_folder
                                clear_folder(path)
                            else:
                                from utils.misc import update_name
                                path = update_name(path)
                                os.makedirs(path, exist_ok=True)
                                os.chmod(path, 0o777)
                                print(f'Directory created for results at : {os.path.abspath(path)}')
                    path_res = f'{path}/{val["variable_name"]}'
                    if not os.path.exists(path_res) and val["variable_name"] is not None:
                        os.makedirs(path_res, exist_ok=True)
                        os.chmod(path_res, 0o777)
                    if val["variable_name"] == 'inputs':
                        for cam in setup.cameras.keys():
                            if not os.path.exists(path_res + f'/{cam}'):
                                os.mkdir(path_res + f'/{cam}')
                                os.chmod(path_res + f'/{cam}', 0o777)
                    elif val["variable_name"] == 'pred_depth':
                        if val["variable_name"] in pred.keys():
                            for cam in pred[val["variable_name"]]:
                                path_used = path_res + f'/{cam}'
                                if not os.path.exists(path_used):
                                    os.mkdir(path_used)
                                    os.chmod(path_used, 0o777)
                        else:
                            val["variable_name"] = None
                    elif val["variable_name"] == 'occlusion':
                        if val["variable_name"] in res.keys():
                            for cam in res[val["variable_name"]]:
                                path_used = path_res + f'/{cam}'
                                if not os.path.exists(path_used):
                                    os.mkdir(path_used)
                                    os.chmod(path_used, 0o777)
                        else:
                            val["variable_name"] = None
                    elif val["variable_name"] == 'image_reg':
                        if val["variable_name"] in res.keys():
                            for cam in res[val["variable_name"]]:
                                path_res_ = path_res + f'/{cam}'
                                if not os.path.exists(path_res_):
                                    os.mkdir(path_res_)
                                    os.chmod(path_res_, 0o777)
                        else:
                            val["variable_name"] = None
                    if val["variable_name"] is not None:
                        process.append(self.save(pipe, val["variable_name"], path))
                    # else:
                    #     p.pop(idx - pop_count)
                    #     pop_count += 1
            self[Exp] = Experiment(process, path)
        pipe.dataloader.camera_used = self.camera_used

    @staticmethod
    def _make_process_dict_(process) -> list:
        proc = []
        for key, p in process.items():
            key = key.upper()
            if key == 'DISPARITY' or key == 'DEPTH':
                assert len(p['cameras']) == 2, \
                    f'The {key} instructions need 2 cameras : cam_src, cam_dst'
                option = {'cam1': p['cameras'][0], 'cam2': p['cameras'][1], 'pred_bidir': False, 'pred_right': False,
                          'cut_roi_max': False, 'cut_roi_min': False}
                if 'option' in p:
                    for p_ in p['option']:
                        if isinstance(p_, dict):
                            assert 'inference_size' in p_.keys(), f'The option given in {key} doesnt exist'
                            option['inference_size'] = p_['inference_size']
                        elif p_.upper() == 'PRED_BIDIR':
                            option['pred_bidir'] = True
                        elif p_.upper() == 'PRED_RIGHT':
                            option['pred_right'] = True
                        elif p_.upper() == 'CUT_ROI_MAX':
                            option['cut_roi_max'] = True
                        elif p_.upper() == 'CUT_ROI_MIN':
                            option['cut_roi_min'] = True
                        else:
                            pass
                proc.append([key, option])
            elif key == 'MONOCULAR_DEPTH' or key == 'MONOCULAR' or key == 'MONO':
                assert len(p['cameras']) >= 1, \
                    f'The {key} instructions need at least one camera'
                option = {'cams': p['cameras']}
                if 'option' in p:
                    for p_ in p['option']:
                        if isinstance(p_, dict):
                            assert 'inference_size' in p_.keys(), f'The option given in {key} doesnt exist'
                            option['inference_size'] = p_['inference_size']
                        else:
                            pass
                proc.append(['MONOCULAR', option])
            elif key == 'WRAP':
                assert len(p['cameras']) == 2, \
                    'The WRAP instruction needs 2 cameras : cam_src, cam_dst or random'
                option = {'cam_src': p['cameras'][0],
                          'cam_dst': p['cameras'][0] if p['cameras'][1] == 'random' else p['cameras'][1],
                          'depth': True if p['method'] == 'depth' else False,
                          'return_depth_reg': False, 'return_occlusion': False}
                if 'option' in p:
                    opt = np.array(p['option'])
                    option['return_depth_reg'] = True if 'return_depth_reg' in opt else False
                    option['return_occlusion'] = True if 'return_occlusion' in opt else False
                    opt_upsample = opt[["upsample" in p_ for p_ in opt]]
                    option['upsample'] = eval(opt_upsample[0].split('=')[-1]) if len(opt_upsample) >= 1 else 1
                proc.append([key, option])
            elif key == 'VALID':
                assert len(p) == 2, 'The VALID instruction needs 2 positional argument : cam_reg, cam_ref'
                option = {'cam_reg': p[0], 'cam_ref': p[1]}
                proc.append([key, option])
            elif key == 'SAVE':
                assert len(p) >= 0, 'The SAVE instruction needs at least 1 positional argument : name_variable'
                if 'all' in p:
                    p = ['inputs', 'pred_depth', 'image_reg', 'depth_reg', 'occlusion']
                for p_ in p:
                    assert p_ in ['inputs', 'pred_depth', 'image_reg', 'depth_reg', 'occlusion'], \
                        'The chosen variable has to be in this list: ' \
                        '[pred_depth, image_reg, depth_reg, inputs, occlusion]'
                    option = {'variable_name': p_}
                    proc.append([key, option])
            elif key == 'SETUP':
                assert len(
                    p) >= 0, 'The SETUP instruction needs at least 1 positional argument : path to the setup file'
                assert os.path.exists(os.getcwd() + '/' + p['path']), f'The setup file {p["path"]}'
                proc.insert(0, [key, os.getcwd() + '/' + p['path']])
            else:
                return NotImplemented
        if 'SAVE' not in [k.upper() for k in process.keys()]:
            proc.append(['SAVE', {'variable_name': None}])
        return proc

    @staticmethod
    def disparity(pipe, cam1, cam2, pred_bidir, pred_right, cut_roi_max, cut_roi_min, inference_size, setup, **kwargs0):
        descr = [f'Using {cam1} and {cam2}']
        summary = {"cam_1": cam1, "cam_2": cam2}

        def _disparity(sample, res, **kwargs):
            pipe.network.update_pred_bidir(activate=pred_bidir)
            pipe.network.update_pred_right(activate=pred_right)
            if inference_size is not None:
                pipe.network.update_size(inference_size, task='disparity')
            else:
                pipe.network.update_size(pipe.config['disparity_network']["network_args"].inference_size)
            setup_ = kwargs['setup'] if 'setup' in kwargs.keys() else setup
            setup_ = setup_.stereo_pair(cam1, cam2)
            new_sample = setup_(sample, cut_roi_min=cut_roi_min, cut_roi_max=cut_roi_max, return_image=True)
            output = pipe.network(new_sample, task='disparity')
            output_ = setup_(output, reverse=True, return_depth=True)
            res['pred_depth'].update(setup_.disparity_to_depth(output_, return_depth=True))

        return _disparity, descr, summary

    @staticmethod
    def depth(pipe, cam1, cam2, pred_bidir, pred_right, inference_size, setup, **kwargs0):
        descr = [f'Using {cam1} and {cam2}']
        summary = {"cam_1": cam1, "cam_2": cam2}

        def _depth(sample, res, **kwargs):
            pipe.network.update_pred_bidir(activate=pred_bidir)
            pipe.network.update_pred_right(activate=pred_right)
            if inference_size is not None:
                pipe.network.update_size(inference_size, task='depth')
            else:
                pipe.network.update_size(pipe.config['depth_network']["network_args"].inference_size, task='depth')
            setup_ = kwargs['setup'] if 'setup' in kwargs.keys() else setup
            setup_ = setup_.depth_pair(cam1, cam2)
            new_sample = setup_(sample)
            output = pipe.network(**new_sample, task='depth')
            res['pred_depth'].update(setup_(output, reverse=True))

        return _depth, descr, summary

    @staticmethod
    def monocular_depth(pipe, cams, inference_size, setup, **kwargs0):

        def _monocular_estimation(sample, res, **kwargs):
            if inference_size is not None:
                pipe.network.update_size(inference_size, task='monocular')
            else:
                pipe.network.update_size(pipe.config['monocular_network']["network_args"].inference_size,
                                         task='monocular')
            setup_ = kwargs['setup'] if 'setup' in kwargs.keys() else setup
            new_sample = {'sample': {cam: sample[cam] for cam in cams},
                          'focal': [setup_.cameras[cam].f for cam in cams],
                          'intrinsics': [setup_.cameras[cam].intrinsics for cam in cams]}
            output = pipe.network(**new_sample, task='monocular')
            res['pred_depth'].update(output)

        return _monocular_estimation, cams, cams

    @staticmethod
    def wrap(pipe, cam_src, cam_dst, depth, return_depth_reg, return_occlusion, reverse_wrap, **kwargs0):
        descr = [f'from {cam_src} to {cam_dst} ' + ('with depth' if depth else 'with disparity')]
        summary = {"cam_src": cam_src, "cam_dst": cam_dst, "reverse": reverse_wrap}

        def _wrap(sample, res, **kwargs):
            wrapper = kwargs['wrapper'] if 'wrapper' in kwargs.keys() else pipe.wrapper
            result = wrapper(res['pred_depth'], sample, cam_src, cam_dst,
                             depth=depth,
                             return_depth_reg=return_depth_reg,
                             return_occlusion=return_occlusion,
                             reverse_wrap=reverse_wrap,
                             **(kwargs0 | kwargs))

            cam_to_cam = f'{cam_src}_to_{cam_dst}'
            if return_depth_reg:
                cam = cam_dst if reverse_wrap else cam_src
                res['depth_reg'].update({cam: result['depth_reg']})
            if return_occlusion:
                res['occlusion'].update({cam_to_cam: result['occlusion']})
            res['image_reg'].update({cam_to_cam: result['image_reg']})

        return _wrap, descr, summary

    @staticmethod
    def valid(pipe, cam_reg, cam_ref, exp_name, **kwargs0):
        descr = [f'With {cam_ref} as ref and {cam_reg} as new image']
        summary = {"cam_ref": cam_ref, "cam_reg": cam_reg}

        def _valid(sample, res, **kwargs):
            name = f'{cam_reg}_to_{cam_ref}'
            im_ref = sample[cam_ref]
            im_reg = res['image_reg'][name].match_shape(im_ref)
            im_old = sample[cam_reg].match_shape(im_ref)
            if name in res['occlusion'].keys():
                occlusion = res['occlusion'][name].to(torch.float32).match_shape(im_ref).to(torch.bool)
            else:
                occlusion = None
            pipe.validation.activated = True
            pipe.validation(im_reg.clone(), im_ref.clone(), im_old.clone(), name, exp_name, occlusion=occlusion)

        return _valid, descr, summary

    @staticmethod
    def save(pipe, variable_name, path, **kwargs0):

        def _save(sample, res, **kwargs):
            path_res = f'{path}/{variable_name}'
            if variable_name == 'inputs':
                var = sample
            else:
                var = res[variable_name]
            pipe.saver(var, path_res)

        return _save, [variable_name], variable_name

    @property
    def isInit(self):
        return self._isInit

    @isInit.setter
    def isInit(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'init_process' or name == '__new__' or name == '__init__':
            self._isInit = value


class Experiment(list):

    def __init__(self, instructions, path, name=None):
        super(Experiment, self).__init__(instructions[1:])
        self.setup = instructions[0]
        self.path = path
        if name is None:
            self.name = path_leaf(path)
        else:
            self.name = name

    def __str__(self):
        space = ' '
        return f'\n{self.name} : {" / ".join([f"{instruction[0].__name__[1:].capitalize()} {space.join(instruction[1])}" for instruction in self])}'

    def __call__(self, sample, *args, **kwargs):
        res = {'pred_depth': {}, 'image_reg': {}, 'depth_reg': {}, 'occlusion': {}}
        for instruction in self:
            instruction[0](sample, res, **kwargs)

    def summary(self):
        summary = {}
        for instruction in self:
            name = f'{instruction[0].__name__[1:].capitalize()}'
            temp = {name: instruction[2]}
            if name in summary:
                if isinstance(summary[name], list):
                    summary[name].append(temp[name])
                else:
                    summary[name] = [summary[name], temp[name]]
            else:
                summary.update(temp)
        return summary


if __name__ == '__main__':
    Process('/home/godeta/PycharmProjects/Disparity_Pipeline/Process.txt')
    print('Done !')
