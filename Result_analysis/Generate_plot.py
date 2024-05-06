import os
import time
import pandas as pd
from matplotlib import pyplot as plt

from Result_analysis.ResultFrame import ResultFrame


def create_directory(directory):
    try:
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    except OSError as e:
        print(f"Error: {e}")


metrics = ['rmse', 'psnr', 'ssim', 'ms_ssim', 'nec']
folder = "methods_comparison"
base_path = os.getcwd() + "/../results/"
for f in reversed(os.listdir(base_path + folder)):
    res = ResultFrame(base_path + folder + '/' + f)
    path = f'{os.getcwd()}/Images/{folder}/{f}'
    if not os.path.exists(path):
        create_directory(path)
    for key in metrics:
        res.show(key, save=f'{path}/{key.upper()}')
        res.delta.show(key, save=f'{path}/{key.upper()}_delta')
        res.delta.show(key, save=f'{path}/{key.upper()}_delta_occ')
    comb = '+'.join([(f'{key}' if key != "rmse" else f'-{key}') for key in metrics])
    comb = res.delta.combine_column(comb, name=f'Sum of Delta for {f} method')
    fig = comb.show()
    fig.savefig(f'{path}/delta_comb.png', bbox_inches='tight', dpi=100)
    plt.close(fig)
