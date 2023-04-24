import os
import shutil
import sys
import json
import time


# def read_text_lines(filepath):
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
#     lines = [l.rstrip() for l in lines]
#     return lines


def name_generator(idx, max_number=10e6):
    k_str = str(idx)
    while idx < max_number:
        k_str = '0' + k_str
        max_number /= 10
    return k_str


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)  # explicitly set exist_ok when multi-processing


def update_name(path):
    i = 0
    path_exp = os.path.join(path, f"({i})")
    path_ok = not os.path.exists(path_exp)
    while not path_ok:
        i += 1
        path_exp = os.path.join(path, f"({i})")
        path_ok = not os.path.exists(path_exp)
    return path_exp


def count_parameter(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of trainable parameters: {num_params}')


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
        if isinstance(self.timeit, list):
            start = time.time()
            res = func(*args, **kwargs)
            self.timeit.append(time.time() - start)
            return res
        else:
            res = func(*args, **kwargs)
            return res
    return wrapper


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
