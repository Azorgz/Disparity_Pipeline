import os
import kornia
import open3d as o3d
import torch
import yaml
from kornia.geometry import StereoCamera, PinholeCamera
from module.BaseModule import BaseModule
from utils.misc import timeit, deactivated


class PointCloudTransformer(BaseModule):
    """
    A class which is instanced if a PointCloud is demanded in the options
    Two methods of computing are presented : One using the Stereo system, the disparity and the baseline,
    the other creating PointsClouds from image + depth and Camera pose.
    """

    def __init__(self, config):
        self.activated = config['pointsCloud']['activated']
        if self.activated:
            super(PointCloudTransformer, self).__init__(config)

    def _update_conf(self, config):
        self.path = os.path.join(config["output_path"], "pointsCloud")
        self.data_type = config['dataset']["type"]
        self.disparity_mode = config['pointsCloud']['disparity']
        self.visualisation = config['pointsCloud']['visualisation']
        self.save = config['pointsCloud']['save']
        self.min_disparity = config['pointsCloud']['min_disparity']
        self.setup = {"position_setup": config['dataset']["position_setup"],
                      "pred_bidir_disp": config['dataset']["pred_bidir_disp"],
                      "proj_right": config['dataset']["proj_right"],
                      "pred_right_disp": config['dataset']["pred_right_disp"]}
        self.mode = config['pointsCloud']['mode']
        self.image = []
        if self.disparity_mode:
            if self.mode == 'stereo' or self.mode == 'both':
                if self.setup['proj_right']:
                    self.image.append('right')
                else:
                    self.image.append('left')
            if self.mode == 'other' or self.mode == 'both':
                self.image.append('other')
        else:
            if self.mode == 'stereo' or self.mode == 'both':
                if self.setup['pred_bidir_disp'] and config['pointsCloud']['use_bidir']:
                    self.image = ['left', 'right']
                elif self.setup['proj_right']:
                    self.image = ['right']
                else:
                    self.image = ['left']
            if self.mode == 'other' or self.mode == 'both':
                self.image.append('other')
        self._configure_cameras(config)
        if config['pointsCloud']['save']:
            self._save_cameras(config)

    def __str__(self):
        string = super().__str__()
        string += f'The PointCloud will be computed using ' \
                  f'{"the disparity and a stereo cameras setup" if self.disparity_mode else "the depth and individual pinhole cameras"}\n'
        return string

    @timeit
    def _cloud_from_disparity(self, disp, sample) -> list:
        disp_ = disp.unsqueeze(0).unsqueeze(0).permute(0, 3, 2, 1).type(torch.DoubleTensor).to(
            self.device)
        min_disp = (1 - self.min_disparity) * disp_.max()
        disp_[disp_ < min_disp] = min_disp
        disp_ /= disp.max()
        pc = []
        pointCloud = self.camera['stereo'].reproject_disparity_to_3D(disp_)  # 1xHxWx3
        pointCloud[0, :, :, 1] *= -1
        pointCloud[0, :, :, 2] *= -1
        flat_pointCloud = torch.flatten(pointCloud, start_dim=1, end_dim=2).squeeze().cpu().numpy()  # HWx3
        for im in self.image:
            image = sample[im]
            if image.shape[0] == 3:
                image = torch.tensor(image).permute(2, 1, 0)  # HxWx3
            else:
                image = torch.tensor(image).permute(1, 0, 2)  # HxWx3
            if image.max() > 1:
                image = image / 255
            flat_image = torch.flatten(image, start_dim=0, end_dim=1).cpu().numpy()  # HWx3
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(flat_pointCloud)
            pcd.colors = o3d.utility.Vector3dVector(flat_image)
            pc.append(pcd)
        return pc

    @timeit
    def _cloud_from_depth(self, disp, sample) -> list:
        pc = []
        for im in self.image:
            pcd = o3d.geometry.PointCloud()
            if disp[im].dtype not in [torch.float64, torch.float32]:
                disp_ = torch.tensor(disp[im])
            else:
                disp_ = disp[im]
            h, w = disp_.shape[-2:]
            min_disp = (1 - self.min_disparity) * disp_.max()
            disp_[disp_ < min_disp] = min_disp
            grid_reg = torch.flatten(kornia.utils.create_meshgrid(h, w, normalized_coordinates=False),
                                     start_dim=1, end_dim=2).type(torch.DoubleTensor).to(self.device).squeeze()  # HWx2
            depth = torch.flatten(
                self.matrix_intrinsic[im][0][0][0] * self.baseline[im] / disp_.type(torch.DoubleTensor).to(
                    self.device)).unsqueeze(-1)
            pointCloud = self.camera[im].unproject(grid_reg, depth)  # 1xHWx3
            pointCloud = torch.reshape(pointCloud, [1, h, w, 3])
            print(f'xmin:{pointCloud[:, :, :, 0].min()}, xmax:{pointCloud[:, :, :, 0].max()}, '
                  f'ymin:{pointCloud[:, :, :, 1].min()}, ymax:{pointCloud[:, :, :, 1].max()}, '
                  f'zmin:{pointCloud[:, :, :, 2].min()}, zmax:{pointCloud[:, :, :, 2].max()}')
            pointCloud[0, :, :, 1] *= -1
            pointCloud[0, :, :, 2] *= -1
            flat_pointCloud = torch.flatten(pointCloud, start_dim=1, end_dim=2).squeeze().cpu().numpy()  # HWx3
            image = sample[im]
            if image.shape[0] == 3:
                image = torch.tensor(image).permute(1, 2, 0)  # HxWx3
            else:
                image = torch.tensor(image).permute(0, 1, 2)  # HxWx3
            if image.max() > 1:
                image = image / 255

            flat_image = torch.flatten(image, start_dim=0, end_dim=1).cpu().numpy()  # HWx3
            pcd.points = o3d.utility.Vector3dVector(flat_pointCloud)
            pcd.colors = o3d.utility.Vector3dVector(flat_image)
            o3d.visualization.draw_geometries(pcd,
                                              mesh_show_wireframe=True,
                                              window_name="_pointCloud_",
                                              point_show_normal=True,
                                              mesh_show_back_face=True)
            pc.append(pcd)
        return pc

    def _configure_cameras(self, config) -> None:
        F = 777
        if self.disparity_mode:
            if config['cameras']['left']:
                self.matrix_left = torch.tensor(config['cameras']['left'], dtype=torch.double).unsqueeze(0).to(
                    self.device)
            else:
                self.matrix_left = torch.tensor([[F, 0, int(config['dataset']["ori_size"][1] / 2), 0],
                                                 [0, F, int(config['dataset']["ori_size"][0] / 2), 0],
                                                 [0, 0, 1, 0]], dtype=torch.double).unsqueeze(0).to(self.device)
            if config['cameras']['right']:
                self.matrix_right = torch.tensor(config['cameras']['right'], dtype=torch.double).unsqueeze(0).to(
                    self.device)
            else:
                self.matrix_right = torch.tensor(
                    [[F, 0, int(config['dataset']["ori_size"][1] / 2), -1],
                     [0, F, int(config['dataset']["ori_size"][0] / 2), 0],
                     [0, 0, 1, 0]], dtype=torch.double).unsqueeze(0).to(self.device)
            self.camera = {'stereo': StereoCamera(self.matrix_left, self.matrix_right)}
            self.function = self._cloud_from_disparity
        else:
            self.matrix_intrinsic = {}
            self.matrix_extrinsic = {}
            self.camera = {}
            self.baseline = {'left': torch.tensor(1).to(self.device), 'right': torch.tensor(1).to(self.device),
                             'other': torch.tensor(1).to(self.device)}
            if self.mode == 'stereo' or self.mode == 'both':

                if config['cameras']['left']:
                    self.matrix_intrinsic['left'] = torch.tensor(config['cameras']['left']['intrinsic'],
                                                                 dtype=torch.double).unsqueeze(0).to(self.device)
                    self.matrix_extrinsic['left'] = torch.tensor(config['cameras']['left']['extrinsic'],
                                                                 dtype=torch.double).unsqueeze(0).to(self.device)
                else:
                    self.matrix_intrinsic['left'] = \
                        torch.tensor([[F, 0, int(config['dataset']["ori_size"][1] / 2), 0],
                                      [0, F, int(config['dataset']["ori_size"][0] / 2), 0], [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=torch.double).unsqueeze(0).to(self.device)
                    self.matrix_extrinsic['left'] = torch.tensor(
                        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        dtype=torch.double).unsqueeze(0).to(self.device)
                if config['cameras']['right']:
                    self.matrix_intrinsic['right'] = torch.tensor(config['cameras']['right']['intrinsic'],
                                                                  dtype=torch.double).unsqueeze(0).to(self.device)
                    self.matrix_extrinsic['right'] = torch.tensor(config['cameras']['right']['extrinsic'],
                                                                  dtype=torch.double).unsqueeze(0).to(self.device)
                else:
                    self.matrix_intrinsic['right'] = \
                        torch.tensor([[F, 0, int(config['dataset']["ori_size"][1] / 2), 0],
                                      [0, F, int(config['dataset']["ori_size"][0] / 2), 0], [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=torch.double).unsqueeze(0).to(self.device)
                    self.matrix_extrinsic['right'] = torch.tensor(
                        [[1, 0, 0, -0.341], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        dtype=torch.double).unsqueeze(0).to(self.device)
                self.camera['left'] = PinholeCamera(self.matrix_intrinsic['left'], self.matrix_extrinsic['left'],
                                                    torch.tensor(config['dataset']["ori_size"][0]).unsqueeze(0).to(
                                                        self.device),
                                                    torch.tensor(config['dataset']["ori_size"][1]).unsqueeze(0).to(
                                                        self.device))
                self.camera['right'] = PinholeCamera(self.matrix_intrinsic['right'], self.matrix_extrinsic['right'],
                                                     torch.tensor(config['dataset']["ori_size"][0]).unsqueeze(0).to(
                                                         self.device),
                                                     torch.tensor(config['dataset']["ori_size"][1]).unsqueeze(0).to(
                                                         self.device))
            if self.mode == 'other' or self.mode == 'both':
                if config['cameras']['other']:
                    self.matrix_intrinsic['other'] = torch.tensor(config['cameras']['other']['intrinsic'],
                                                                  dtype=torch.double).unsqueeze(0).to(self.device)
                    self.matrix_extrinsic['other'] = torch.tensor(config['cameras']['other']['extrinsic'],
                                                                  dtype=torch.double).unsqueeze(0).to(self.device)
                else:
                    self.matrix_intrinsic['other'] = \
                        torch.tensor([[F, 0, int(config['dataset']["ori_size"][1] / 2), 0],
                                      [0, F, int(config['dataset']["ori_size"][0] / 2), 0], [0, 0, 1, 0],
                                      [0, 0, 0, 1]], dtype=torch.double).unsqueeze(0).to(self.device)
                    self.matrix_extrinsic['other'] = \
                        torch.tensor([[1, 0, 0, -0.127], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                     dtype=torch.double).unsqueeze(0).to(self.device)
                self.camera['other'] = PinholeCamera(self.matrix_intrinsic['other'], self.matrix_extrinsic['other'],
                                                     torch.tensor(config['dataset']["ori_size"][0]).unsqueeze(0).to(
                                                         self.device),
                                                     torch.tensor(config['dataset']["ori_size"][1]).unsqueeze(0).to(
                                                         self.device))
            self.function = self._cloud_from_depth
            for key in self.matrix_extrinsic.keys():
                self.baseline[key] = -self.matrix_extrinsic[key][0][0][-1]
            self.baseline['left'] = self.baseline['right']

    def _save_cameras(self, config) -> None:
        camera_dict = {'disparity': self.disparity_mode,
                       'mode': self.mode}
        if self.disparity_mode:
            camera_dict["left"] = self.matrix_left.squeeze().cpu().numpy().tolist()
            camera_dict["right"] = self.matrix_right.squeeze().cpu().numpy().tolist()
        else:
            for key in self.matrix_extrinsic.keys():
                camera_dict[key] = {}
                camera_dict[key]["intrinsic"] = self.matrix_intrinsic[key].squeeze().cpu().numpy().tolist()
                camera_dict[key]["extrinsic"] = self.matrix_extrinsic[key].squeeze().cpu().numpy().tolist()

        name = os.path.join(config["dataset"]['path'], "dataset.yaml")
        with open(name, "r") as file:
            dataset_conf = yaml.safe_load(file)
        dataset_conf['3. Position and alignment']['camera'] = camera_dict
        with open(name, "w") as file:
            yaml.dump(dataset_conf, file)

    @deactivated
    def __call__(self, disp, sample, name, *args):
        """
        if the mode = 'both', only one disparity (the reference disparity) will be used on the main stereo image
        and the other modality reg_image
        :param disp: disparity of the chosen mode
        :param sample: sample containing the image(s) to create the pointcloud(s)
        :return: None, save and display pointsCloud
        """
        pcd = self.function(disp, sample)
        if self.visualisation:
            for idx, p in enumerate(pcd):
                p = p.voxel_down_sample(voxel_size=0.01)
                cl, ind = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
                pcd[idx] = p.select_by_index(ind)
            o3d.visualization.draw_geometries(pcd,
                                              mesh_show_wireframe=True,
                                              window_name=name + "_pointCloud_",
                                              point_show_normal=True,
                                              mesh_show_back_face=True)

    @staticmethod
    def display_inlier_oulier(cloud, ind):
        """
        :param cloud: The cloud of point to be visualized
        :param ind: the index of inlier
        :return: None, just visualization purpose
        """
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          mesh_show_wireframe=True,
                                          point_show_normal=True,
                                          mesh_show_back_face=True)
