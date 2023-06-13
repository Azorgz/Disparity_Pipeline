import warnings
from Config.Config import ConfigPipe
from Disparity_Pipeline import Pipe


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = ConfigPipe()
    pipe = Pipe(config)
    pipe.run()
    print('Done !')
