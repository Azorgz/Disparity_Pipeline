from utils.misc import timeit


class Fusion:
    """
    A module which purpose is to fusion two images with the given methods
    """

    def __init__(self, config, *args, **kwargs):
        super(Fusion, self).__init__(config, *args, **kwargs)
        self.methods = {'alpha': self._alpha_fusion}

    def __str__(self):
        return ''

    @timeit
    def __call__(self, sample, *args, method='alpha', alpha=0.5, **kwargs):
        return self.methods[kwargs['method']](sample, *args, **kwargs)

    def _alpha_fusion(self, im1, im2, *args, **kwargs):
        alpha = kwargs['alpha']
        return im1 * alpha + im2 * (1 - alpha)
