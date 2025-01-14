import collections.abc
import math
import ntpath
import os
import sys
import stat
import shutil
import json
import time
import torch
import os
import struct


# Names managing functions ##################################
def name_generator(idx, max_number=10e4):
    k_str = str(idx)
    digits = 1 if max_number < 10 else int(math.log10(max_number)) + 1
    current_digits = 1 if idx < 10 else int(math.log10(idx)) + 1
    for i in range(digits - current_digits):
        k_str = '0' + k_str
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


def update_name_tree(sample: dict, suffix):
    for im in sample.values():
        im.name += '-' + suffix
    return sample


def path_leaf(path):
    if os.path.isdir(path):
        res = ntpath.split(path)[-1]
    else:
        res = ntpath.split(path)[-1].split(".")[:-1]
    if isinstance(res, list):
        if len(res) > 1:
            res = ''.join(res)
        else:
            res = res[0]
    return res


def add_ext(word, ext):
    if ext != '':
        return f'{word}_{ext}'
    else:
        return word


def create_function_from_string(func_name, function_code, prop=False, **kwargs):
    # Define the function code as a string
    var_name = list(kwargs.keys())
    for k in var_name:
        exec(f'{k} = kwargs["{k}"]')
    func_code = f"""
{'@property' if prop else ''}
def {func_name}({', '.join(var_name)}):
    {function_code}
    """
    exec(func_code)
    print(func_code)


def extract_key_pairs(data: dict) -> list[tuple]:
    """
    Function that extracts key pairs from a dictionary and return it as a list of tuples
    :param data: dict to process
    :return: tuples of (ext, new, ref)
    """
    keys_list = [k.replace('new', '').replace('_', '') for k in data.keys() if 'new' in k]
    return [(k, add_ext('new', k), add_ext('ref', k)) for k in keys_list]


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def count_parameter(model):
    if model is not None:
        num_params = sum(p.numel() for p in model.parameters())
        return f'Number of trainable parameters: {num_params}'
    else:
        return 'Not loaded'


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


# Dict manipulation ##################################
def list_to_dict(list_of_dict):
    res = {}
    for d in list_of_dict:
        if d.keys() == res.keys():
            for key in d.keys():
                res[key].append(d[key])
        else:
            for key in d.keys():
                res[key] = [d[key]]
    return res


def merge_dict(dict1: dict, dict2: dict, *args):
    if not dict1.keys() == dict2.keys():
        res = dict1 | dict2
        if args:
            res = merge_dict(res, *args)
    else:
        res = dict1.copy()
        for k in res.keys():
            if isinstance(res[k], dict) and isinstance(dict2[k], dict):
                res[k] = merge_dict(res[k], dict2[k])
            elif isinstance(res[k], list) and isinstance(dict2[k], list):
                if isinstance(res[k][0], list):
                    res[k] = [*res[k], dict2[k]]
                else:
                    res[k] = [res[k], dict2[k]]
                # for idx, (r1, r2) in enumerate(zip(res[k], dict2[k])):
                #     if isinstance(r1, dict) and isinstance(r2, dict):
                #         res[k][idx] = merge_dict(r1, r2)
                #     elif isinstance(r1, list) and isinstance(r2, float):
                #         res[k][idx] = [*r1, r2]
                #     elif isinstance(r1, float) and isinstance(r2, list):
                #         res[k][idx] = [r1, *r2]
                #     else:
                #         res[k][idx] = [r1, r2]
            elif (isinstance(res[k], list) and
                  (isinstance(dict2[k], float) or isinstance(dict2[k], int) or isinstance(dict2[k], str))):
                res[k] = [*res[k], dict2[k]]
            elif ((isinstance(res[k], float) or isinstance(res[k], int) or isinstance(res[k], str)) and
                  isinstance(dict2[k], list)):
                res[k] = [res[k], *dict2[k]]
            else:
                res[k] = [res[k], dict2[k]]
        if args:
            res = merge_dict(res, *args)
    return res


def flatten_dict(x):
    result = []
    if isinstance(x, dict):
        x = x.values()
    for el in x:
        if isinstance(el, dict) and not isinstance(el, str):
            result.extend(flatten_dict(el))
        else:
            result.append(el)
    return result


def map_dict_level(d: dict, level=0, map_of_dict=[], map_of_keys=[]):
    if len(map_of_dict) <= level:
        map_of_dict.append([len(d)])
        map_of_keys.append(list(d.keys()))
    else:
        map_of_dict[level].append(len(d))
    for idx, res in d.items():
        if isinstance(res, dict):
            map_of_dict, map_of_keys = map_dict_level(res, level + 1, map_of_dict, map_of_keys)
        else:
            return map_of_dict, map_of_keys
    if level == 0:
        map_of_dict.pop(0)
    return map_of_dict, map_of_keys


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


class ClassAnalyzer:
    def __init__(self, c):
        self.call = 0
        self.c = c
        self.cum_time = 0

    def __call__(self, *args, **kwargs):
        self.call += 1
        t = time.time()
        res = self.c(*args, **kwargs)
        self.cum_time += time.time() - t
        return res

    @property
    def average(self):
        if self.c != 0:
            return self.cum_time / self.c
        else:
            return 0

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
