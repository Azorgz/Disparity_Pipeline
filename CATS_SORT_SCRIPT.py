# CATS -------------------------------- #
import os
import shutil
from glob import glob

paths = []
INDOORS = {'color': {'left': [], 'right': []}, 'thermal': {'left': [], 'right': []}}
OUTDOORS = {'color': {'left': [], 'right': []}, 'thermal': {'left': [], 'right': []}}
GT = {'color': {'left': [], 'right': []}, 'thermal': {'left': [], 'right': []}}
new_folder = '/media/godeta/T5 EVO/Datasets/CATS_Sorted'


def find_raw_images_folders(path):
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)) and d.startswith('raw'):
            paths.append(os.path.join(path, d))
        elif os.path.isdir(os.path.join(path, d)):
            find_raw_images_folders(os.path.join(path, d))
        else:
            pass


def build_images_list(path):
    for f in sorted(glob(path + '/*.png')):
        in_out, im_name = f.split('/')[-5], f.split('/')[-1]
        im_side, mod, _ = im_name.split('.')[0].split('_')
        if in_out == 'INDOOR':
            INDOORS[mod][im_side].append(f)
        else:
            OUTDOORS[mod][im_side].append(f)


def generate_name_from_path(im_path):
    im_path = im_path.split('/')
    obj, scene, special = im_path[-4], im_path[-3].split('scene')[-1], im_path[-1].split('_')[-1]
    return f"{obj}_{scene}_{special}"


def save_dict(dict_2_save, path_save):
    for key, val in dict_2_save.items():
        if not os.path.exists(path_save + f'/{key}'):
            os.makedirs(path_save + f'/{key}', exist_ok=True)
        if isinstance(val, dict):
            save_dict(val, path_save + f'/{key}')
        elif isinstance(val, list):
            for image_path in val:
                name = generate_name_from_path(image_path)
                shutil.copy(image_path, f"{path_save}/{key}/{name}")


find_raw_images_folders("/media/godeta/T5 EVO/Datasets/CATS_Release")
for path in sorted(paths):
    build_images_list(path)

if not os.path.exists(new_folder):
    os.makedirs(new_folder + '/OUTDOORS', exist_ok=True)
    os.makedirs(new_folder + '/INDOORS', exist_ok=True)
save_dict(INDOORS, new_folder + '/INDOORS')
save_dict(OUTDOORS, new_folder + '/OUTDOORS')
