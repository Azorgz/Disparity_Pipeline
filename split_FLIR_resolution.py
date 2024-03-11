import glob
import os

import torch
import tqdm

from utils.classes import ImageTensor


perso = '/home/aurelien/Images/Images_FLIR/FLIR_ADAS_1_3_full/'
pro = '/home/godeta/PycharmProjects/'

p = pro if 'godeta' in os.getcwd() else perso

p_train_RGB = p + 'FLIR_ADAS_1_3_train/train/RGB'
p_val_RGB = p + 'FLIR_ADAS_1_3_val/val/RGB'
p_train_IR = p + 'FLIR_ADAS_1_3_train/train/thermal_8_bit'
p_val_IR = p + 'FLIR_ADAS_1_3_val/val/thermal_8_bit'
n_p_RGB = p + '/RGB_processed'
n_p_IR = p + '/IR_processed'

files_RGB = glob.glob(p_train_RGB + '/*.jpg') + glob.glob(p_val_RGB + '/*.jpg')


for idx, RGB in enumerate(tqdm.tqdm(files_RGB)):
    im_name = RGB.split('/')[-1].split('.')[0] + '.jpeg'
    im_path = RGB.split('/')[-3]
    path_ir = p_val_IR + '/' + im_name if im_path == 'val' else p_train_IR + '/' + im_name
    try:
        ir = ImageTensor(path_ir)
    except FileNotFoundError:
        continue
    rgb = ImageTensor(RGB)
    l = torch.tensor(rgb.shape[-2:]).numpy().tolist()
    l.reverse()
    l = [str(t) for t in l]
    folder_res = 'x'.join(l)
    path = f'{n_p_RGB}/{folder_res}'
    rgb.save(path)
    path = f'{n_p_IR}/{folder_res}'
    ir.save(path)


