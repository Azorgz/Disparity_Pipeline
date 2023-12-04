import os
import shutil
import warnings
import oyaml as yaml
from Disparity_Pipeline import Pipe
from config.Config import ConfigPipe
from module.Process import Process
from utils.classes.Visualizer import Visualizer


def full_process(script_path):
    warnings.filterwarnings('ignore')
    process = Process(os.getcwd() + '/' + script_path)
    config = ConfigPipe(process.option)
    pipe = Pipe(config)
    pipe.run(process)


if __name__ == '__main__':
    full_process('Process_resolution.yaml')
