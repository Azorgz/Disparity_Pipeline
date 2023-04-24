from utils.misc import timeit
from utils.transforms import Resize


class Preprocessing:
    """
    A class which define apply the preprocessing necessary for each Network
    """
    def __init__(self, config):
        self.transforms = config["transforms"]
        # If the DisparityPipe need to measure the execution time of each block, this parameter will be set to True.
        if config["timeit"]:
            self.timeit = []

    @timeit
    def __call__(self, sample, *args, **kwargs):
        if self.transforms is not None:
            sample = self.transforms(sample)
        self.inference_size = None
        self.ori_size = None
        for t in self.transforms.transforms:
            if isinstance(t, Resize):
                self.inference_size = t.inference_size
                self.ori_size = t.ori_size
        return sample
