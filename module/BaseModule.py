

class BaseModule:
    """
    A class which is the basic module with the basic implementation of each attribute the modules need
    """

    def __init__(self, config, *args, verbose=True, **kwargs):
        self.activated = True
        if config["timeit"]:
            self.timeit = []
        self.config = config
        self.device = config['device']['device']
        self._update_conf(config, *args, **kwargs)
        if self.config['print_info'] and verbose and self.activated:
            print(self)

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        return f'\n############# {self.__class__.__name__.upper()} ######\n'

    def _update_conf(self, config, *args, **kwargs):
        pass