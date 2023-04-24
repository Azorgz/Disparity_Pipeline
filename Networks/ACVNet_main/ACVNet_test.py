import pathlib
import time
import sys
import os
import numpy as np
import torch
from torch import nn

from Stereo_matching.NeuralNetwork.ACVNet_main.models import __models__
from Stereo_matching.NeuralNetwork.ACVNet_main.ACVNet_main import test_sample


def ACVNet_test(sample, verbose=False, post_process=0, time_of_day='Day', im_type='visible', clean=False, path_save=None, colormap='inferno'):
    start = time.time()
    model = initialize_model(verbose=False)
    print(f"    Setup for inference done in {round(time.time() - start,2)} seconds")
    start = time.time()
    if not(sample is None):
        image_outputs = test_sample(sample, model)
        m, M = np.array(image_outputs['disp_est']).min(), np.array(image_outputs['disp_est']).max()
        if verbose:
            print(f"        Max disparity = {M}\n"
                  f"        Min disparity = {m}")
        image_outputs['disp_est'] = (image_outputs['disp_est'] - m) / (M - m)
        print(f"    Inference done in {round(time.time() - start, 2)} seconds !")
        return image_outputs['imgL'], image_outputs['imgR'], image_outputs['disp_est'], m, M
    else:
        test_sample(sample, model, threshold=post_process, verbose=False, time_of_day=time_of_day, im_type=im_type,
                    clean=clean, path_save=path_save, colormap=colormap)
    print(f"    Inference done in {round(time.time() - start,2)} seconds !")


def initialize_model(verbose=False):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    model = __models__['acvnet'](192, False, False)
    model = nn.DataParallel(model)
    model.cuda()
    if verbose:
        print(f"    loading model {'pretrained_model_sceneflow.ckpt'}")
    base_path = os.path.dirname(os.path.dirname(pathlib.Path().resolve()))
    state_dict = torch.load(
        base_path + '/Stereo_matching/NeuralNetwork/ACVNet_main/pretrained_model/pretrained_model_sceneflow.ckpt')
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    return model
