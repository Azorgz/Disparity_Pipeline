import os
import sys
import stat
import shutil
import json
import time

import torch


# def read_text_lines(filepath):
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
#     lines = [l.rstrip() for l in lines]
#     return lines


def name_generator(idx, max_number=10e6):
    k_str = str(idx)
    while idx + 0.5 < max_number:
        k_str = '0' + k_str
        max_number /= 10
    return k_str


def time2str(t, optimize_unit=True):
    if not optimize_unit:
        return str(round(t, 3)) + ' sec'
    else:
        unit = 0
        unit_dict = {-1: " h", 0: " s", 1: " ms", 2: " us", 3: " ns"}
        while t < 1:
            t *= 1000
            unit += 1

        if t > 3600:
            t /= 3600
            unit = -1
            str_time = str(int(t)) + unit_dict[unit] + str(t % 1) + unit_dict[unit + 1]
        else:
            str_time = str(round(t, 3)) + unit_dict[unit]
        return str_time


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def make_writable(folder_path):
    os.chmod(folder_path, stat.S_IRWXU)
    dirs = os.listdir(folder_path)
    for d in dirs:
        os.chmod(os.path.join(folder_path, d), stat.S_IRWXU)


def update_name(path):
    i = 0
    path_exp = path + f"({i})"
    path_ok = not os.path.exists(path_exp)
    while not path_ok:
        i += 1
        path_exp = path + f"({i})"
        path_ok = not os.path.exists(path_exp)
    return path_exp


def count_parameter(model):
    num_params = sum(p.numel() for p in model.parameters())
    return f'Number of trainable parameters: {num_params}'


def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def timeit(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if hasattr(self, "timeit"):
            if isinstance(self.timeit, list):
                start = time.time()
                res = func(*args, **kwargs)
                self.timeit.append(time.time() - start)
                return res
            else:
                res = func(*args, **kwargs)
                return res
        else:
            res = func(*args, **kwargs)
            return res

    return wrapper


def deactivated(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        if hasattr(self, "activated"):
            if self.activated:
                res = func(*args, **kwargs)
                return res
            else:
                pass
        else:
            res = func(*args, **kwargs)
            return res

    return wrapper


def form_cloud_data(sample, pred_disp, image_reg, new_disp, config):
    if config['pointsCloud']['disparity']:
        if config['dataset']['pred_bidir_disp']:
            if config['dataset']['proj_right']:
                cloud_disp = pred_disp[1].copy()
            else:
                cloud_disp = pred_disp[0].copy()
        else:
            cloud_disp = pred_disp.copy()
        cloud_sample = {key: im.copy() for key, im in sample.items()}
        cloud_sample['other'] = image_reg
    else:
        cloud_disp = {}
        if config['pointsCloud']['mode'] == 'stereo' or config['pointsCloud']['mode'] == 'both':
            if config['dataset']['pred_bidir_disp']:
                cloud_disp = {'left': pred_disp[0], 'right': pred_disp[1]}
            elif config['dataset']['pred_right_disp']:
                cloud_disp = {'right': pred_disp.copy()}
            else:
                cloud_disp = {'left': pred_disp.copy()}
        if config['pointsCloud']['mode'] == 'other' or config['pointsCloud']['mode'] == 'both':
            cloud_disp['other'] = new_disp.copy()
        cloud_sample = {key: im.copy() for key, im in sample.items()}
    return cloud_sample, cloud_disp


# def save_command(save_path, filename='command_train.txt'):
#     check_path(save_path)
#     command = sys.argv
#     save_file = os.path.join(save_path, filename)
#     # Save all training commands when resuming training
#     with open(save_file, 'a') as f:
#         f.write(' '.join(command))
#         f.write('\n\n')
#
#
# def save_args(args, filename='args.json'):
#     args_dict = vars(args)
#     check_path(args.checkpoint_dir)
#     save_path = os.path.join(args.checkpoint_dir, filename)
#
#     # save all training args when resuming training
#     with open(save_path, 'a') as f:
#         json.dump(args_dict, f, indent=4, sort_keys=False)
#         f.write('\n\n')
