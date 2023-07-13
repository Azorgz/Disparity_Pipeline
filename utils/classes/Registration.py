from module.SuperNetwork import SuperNetwork
from utils.registration_tools import manual_registration, manual_position_calibration, manual_position_calibration_v2, \
    manual_registration_v2


class Registration:
    """
    A Class instanced with a CameraSetup to tend to the cameras registrations when needed
    """
    _model = None
    perspective_transform = None
    manual_calibration_available = False

    def __init__(self, device=None, model: SuperNetwork = None, distance_ref=5):
        """
        :param device: Device cuda or cpu
        :param model: Disparity Network to compute the disparity (needed for manual registration)
        :param distance_ref: distance where the camera FOV is the same
        """
        self.device = device
        self.model = model
        self.distance_ref = distance_ref
        if model is not None:
            try:
                assert model.device == device
            except AssertionError:
                print('The given model is not on the same device as the Setup')
            self.manual_calibration_available = True

    def __call__(self, left, right, position_ref=False, manual=True):
        inverse, pos = False, 0
        if manual:
            funct = manual_registration_v2
            if not position_ref:
                inverse = manual_position_calibration_v2(left, right)
                if inverse:
                    left, right = right, left
                    position_ref = 0
                else:
                    position_ref = 1
            else:
                position_ref = 1 if position_ref == 'left' else 0
        # else:
        #     funct = keypoints_registration
        matrix, crop = funct(left, right, self.model, position_ref)
        return matrix, crop, matrix[0, 2]*self.distance_ref

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.manual_calibration_available = True

    @model.deleter
    def model(self):
        self._model = None
