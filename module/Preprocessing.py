from module.PostProcessing import PostProcessing
from utils.ImagesCameras.Geometry.transforms import Resize, Pad, DispSide, Compose


class Preprocessing:
    """
    A class which define apply the preprocessing necessary for each Network
    """

    def __init__(self, transform, device, task='disparity', pred_right=False, pred_bidir=False):
        if isinstance(transform[-1], DispSide):
            transform = transform[:-1]
        transform.append(DispSide(pred_right, pred_bidir))
        self.config = {'pred_right': pred_right, 'pred_bidir': pred_bidir}
        self.transforms = Compose(transform, device)
        self.task = task
        self.pad = None
        self.resize = None
        self.device = device
        for t in self.transforms.transforms:
            if isinstance(t, Resize):
                self.resize = t
            if isinstance(t, Pad):
                self.pad = t
        self.postprocessing = None

    @property
    def inference_size(self):
        if self.resize is not None:
            return self.resize.inference_size
        elif self.pad is not None:
            return self.pad.inference_size
        else:
            return None

    @inference_size.setter
    def inference_size(self, size):
        if self.resize is not None and self.resize.inference_size != size:
            self.resize.inference_size = size
        elif self.pad is not None:
            self.pad.inference_size = size
        if self.postprocessing is not None:
            self.postprocessing = None

    @property
    def ori_size(self):
        if self.resize is not None:
            return self.resize.ori_size
        elif self.pad is not None:
            return self.pad.ori_size
        else:
            return None

    def __str__(self):
        return ""

    def __call__(self, sample, reverse=False, *args, **kwargs):
        if not reverse:
            for key, im in sample.items():
                if im.modality == 'Any':
                    sample[key] = im.RGB('gray')
            if self.transforms is not None:
                sample = self.transforms(sample, **kwargs)
            if self.postprocessing is None:
                self.postprocessing = PostProcessing(self.config,
                                                     self.device,
                                                     task=self.task,
                                                     pad=self.pad,
                                                     resize=self.resize,
                                                     post_process=None)
        else:
            for key, im in sample.items():
                if im.modality == 'Any':
                    sample[key] = im.GRAY()
            sample = self.postprocessing(sample, **kwargs)
        return sample
