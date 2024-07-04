from pathlib import Path
from typing import Union
from module.BaseModule import BaseModule
from utils.misc import timeit


class ImageSaver(BaseModule):

    def __init__(self, config, *args, **kwargs):
        super(ImageSaver, self).__init__(config, *args, **kwargs)

    def _update_conf(self, config, *args, **kwargs):
        self.device = config["device"]["device"]

    def __str__(self):
        return ''

    @timeit
    def __call__(self, var: dict, path: Union[str, Path], *args, **kwargs):
        for key, im in var.items():
            # name = f'{im.im_name}.png'
            im.save(path + f'/{key}', depth=8)
