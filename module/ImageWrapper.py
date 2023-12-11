from utils.classes import ImageTensor
from utils.classes.DepthWrapper import DepthWrapper
from utils.classes.DisparityWrapper import DisparityWrapper
from module.SetupCameras import CameraSetup
from kornia.geometry import relative_transformation
import warnings

from config.Config import ConfigPipe
from module.BaseModule import BaseModule
from utils.classes.Image import DepthTensor
from utils.misc import timeit

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)


class ImageWrapper(BaseModule):
    """
    This block implement the reconstruction methods following the reconstruction options
    """

    def __init__(self, config: ConfigPipe, setup: CameraSetup, *args, **kwargs):
        super(ImageWrapper, self).__init__(config, *args, **kwargs)
        self.setup = setup
        # If the DisparityPipe need to measure the execution time of each block, this parameter will be set to True.
        self.depth_wrapper = DepthWrapper(self.device)
        self.disparity_wrapper = DisparityWrapper(self.device, setup)

    def _update_conf(self, config, *args, **kwargs):
        self.device = config["device"]["device"]
        self.save_disp = config["save_disp"]
        self.remove_occlusion = config['reconstruction']['remove_occlusion']
        if self.remove_occlusion:
            self.post_process_image = config['reconstruction']['post_process_image']
            self.post_process_depth = config['reconstruction']['post_process_depth']
        else:
            self.post_process_image = False
            self.post_process_depth = False

    def __str__(self):
        return ''

    @timeit
    def __call__(self, depth_tensors: dict, sample: dict, cam_src: str, cam_dst: str, *args,
                 depth=False,
                 return_depth_reg=False,
                 return_occlusion=False,
                 **kwargs) -> (ImageTensor, DepthTensor):
        """
        :param depth_tensors: Depth tensor computed using network
        :param sample: frames of the different cameras
        :param cam_src: name of the src camera (source of the image to be projected)
        :param cam_dst: name of the dst camera (destination of the image to be projected)
        :param args: other cameras to be used to complete the occluded pixels
        :param depth: if the wrapper use Depth wrapping or disparity wrapping
        :param kwargs: Further implementation?
        :return: ImageTensor of the projected image
        """
        if depth:
            res = self.depth_wrapper(sample[cam_src].clone(),
                                     depth_tensors[cam_dst].clone(),
                                     self.setup.cameras[cam_src].intrinsics[:, :3, :3],
                                     self.setup.cameras[cam_dst].intrinsics[:, :3, :3],
                                     relative_transformation(
                                         self.setup.cameras[cam_src].extrinsics.inverse(),
                                         self.setup.cameras[cam_dst].extrinsics.inverse()),
                                     *args,
                                     return_depth_reg=return_depth_reg,
                                     return_occlusion=return_occlusion,
                                     post_process_image=self.post_process_image,
                                     post_process_depth=self.post_process_depth,
                                     **kwargs)
        else:
            res = self.disparity_wrapper(sample.copy(),
                                         depth_tensors.copy(),
                                         cam_src,
                                         cam_dst,
                                         *args,
                                         return_depth_reg=return_depth_reg,
                                         return_occlusion=return_occlusion,
                                         post_process_image=self.post_process_image,
                                         post_process_depth=self.post_process_depth,
                                         **kwargs)
        res['image_reg'].pass_attr(sample[cam_src])
        return res
