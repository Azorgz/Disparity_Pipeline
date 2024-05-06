import os
import shutil
import warnings

import oyaml as yaml
from Disparity_Pipeline import Pipe
from config.Config import ConfigPipe
from module.Process import Process
from utils.classes.Vizualisation.Visualizer import Visualizer


def quick_process(idx, script_path):
    warnings.filterwarnings('ignore')
    if not os.path.exists(script_path):
        script_path = os.getcwd() + '/' + script_path
        assert os.path.exists(script_path)
    with open(script_path, 'r') as file:
        process_dict = yaml.safe_load(file)

    process_dict['Option']["output_path"] = '/'
    process_dict['Option']["name_experiment"] = 'temp'
    if idx is not None:
        process_dict['Option']["dataset"]["indexes"] = idx if isinstance(idx, list) else [idx]
    else:
        process_dict['Option']["dataset"]["indexes"] = None
    for k in process_dict.keys():
        if k.upper() != 'OPTION' and k.upper() != 'NAME':
            proc = process_dict[k]
            proc['save'] = 'all'
            proc['wrap']['option'].append('return_depth_reg')
    process = Process(process_dict=process_dict)
    config = ConfigPipe(process.option)
    config['validation']['stats']['mean'] = False
    config['validation']['stats']['std'] = False
    pipe = Pipe(config)
    pipe.run(process=process)
    path = os.getcwd() + "/temp"
    Visualizer(path, search_exp=True).run()
    shutil.rmtree(path)


if __name__ == '__main__':
    quick_process(None, 'Process/Process_test.yaml')
    #[1324, 2159, 2543, 3493]
