import os
import warnings
from Disparity_Pipeline import Pipe
from config.Config import ConfigPipe
from module.Process import Process


def full_process(script_path):
    warnings.filterwarnings('ignore')
    process = Process(os.getcwd() + '/' + script_path)
    config = ConfigPipe(process.option)
    pipe = Pipe(config)
    pipe.run(process)


if __name__ == '__main__':
    full_process('Process/Process_methods.yaml')
    # full_process('Process/Process_dataset_creation.yaml')
