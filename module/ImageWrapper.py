import torch

from utils.ImagesCameras import ImageTensor, CameraSetup, DepthTensor
from kornia.geometry.epipolar import scale_intrinsics
from utils.ImagesCameras.Wrappers.DepthWrapper import DepthWrapper
from utils.ImagesCameras.Wrappers.DisparityWrapper import DisparityWrapper
from kornia.geometry import relative_transformation, axis_angle_to_quaternion, quaternion_to_rotation_matrix, \
    Rt_to_matrix4x4
import warnings

from config.Config import ConfigPipe
from module.BaseModule import BaseModule
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
        self.remove_occlusion = config['reconstruction']['remove_occlusion']
        self.random_projection = config['reconstruction']['random_projection']
        self.relative_poses = None
        self.scales = None
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
                 reverse_wrap=False,
                 **kwargs) -> (ImageTensor, DepthTensor):
        """
        :param depth_tensors: Depth tensor computed using network
        :param sample: frames of the different cameras
        :param cam_src: name of the src camera (source of the image to be projected)
        :param cam_dst: name of the dst camera (destination of the image to be projected)
        :param args: other cameras to be used to complete the occluded pixels
        :param depth: if the wrapper use Depth wrapping or disparity wrapping
        :param inverse_wrap: If True the image will be projected from dst to src
        :param kwargs: Further implementation?
        :return: ImageTensor of the projected image
        """
        if self.random_projection is not None:
            scale, tx, ty, tz, *r = (torch.rand([7], device=self.device) - 0.5) * torch.tensor(self.random_projection,
                                                                                               device=self.device)
            quaternion = axis_angle_to_quaternion(torch.deg2rad(torch.stack(r, dim=-1)))
            rotations = quaternion_to_rotation_matrix(quaternion)[None]
            translations = torch.stack([tx, ty, tz], dim=0)[None, :, None]  # N 3 1
            relative_pose = Rt_to_matrix4x4(rotations, translations)
            cam_src_intrinsic = self.setup.cameras[cam_src].intrinsics[:, :3, :3]
            cam_dst_intrinsic = scale_intrinsics(self.setup.cameras[cam_dst].intrinsics[:, :3, :3], 1 - scale)
            if self.relative_poses is None:
                self.relative_poses = [relative_pose.cpu().numpy().tolist()]
                self.scales = [float(scale.cpu())]
            else:
                self.relative_poses.append(relative_pose.cpu().numpy().tolist())
                self.scales.append(float(scale.cpu()))
        else:
            cam_src_intrinsic = self.setup.cameras[cam_src].intrinsics[:, :3, :3]
            cam_dst_intrinsic = self.setup.cameras[cam_dst].intrinsics[:, :3, :3]
            relative_pose = relative_transformation(
                self.setup.cameras[cam_src].extrinsics.inverse(),
                self.setup.cameras[cam_dst].extrinsics.inverse())
        if depth:
            res = self.depth_wrapper(sample[cam_src].clone(),
                                     sample[cam_dst].clone(),
                                     depth_tensors[cam_dst].clone() if not reverse_wrap else depth_tensors[cam_src].clone(),
                                     cam_src_intrinsic,
                                     cam_dst_intrinsic,
                                     relative_pose,
                                     *args,
                                     return_depth_reg=return_depth_reg,
                                     return_occlusion=return_occlusion,
                                     post_process_image=self.post_process_image,
                                     post_process_depth=self.post_process_depth,
                                     reverse_wrap=reverse_wrap,
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
                                         reverse_wrap=reverse_wrap,
                                         **kwargs)
        return res
