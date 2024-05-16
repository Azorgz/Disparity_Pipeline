# Copyright 2023 Toyota Research Institute.  All rights reserved.
import time
import numpy as np
import torch.nn.functional as F
import os
import sys
import torch
from torch import Tensor
import utils.classes.Vizualisation.camviz.camviz as cv
from Networks.Depth_anything.metric_depth.zoedepth.models.builder import build_model
from Networks.Depth_anything.metric_depth.zoedepth.utils.config import get_config
from config.Config import configure_parser
from module.SetupCameras import CameraSetup
from utils.classes.Vizualisation.camviz.camviz.objects.quaternion import pose_from_extrinsics
from utils.classes import ImageTensor

sys.path.append(os.getcwd() + '/Networks/Depth_anything/metric_depth')
device = torch.device('cuda:0')
time_it = False
# cmap = cv.utils.cmaps.depth_clr
cmap = cv.utils.cmaps.jet

R = CameraSetup(from_file=os.getcwd() + "/Setup_Camera/Lynred_day&night.yaml")
parser = get_config('zoedepth', "infer")
config = configure_parser(parser,
                          None,
                          path_config=os.getcwd() + '/Networks/Depth_anything/config_Depth_anything.yml',
                          dict_vars=None)
config.pretrained_resource = config.path_checkpoint
# Depth
NN = build_model(config).eval().to(device)


def scaled(s, size=None, im=None):
    if size is not None:
        return size[-2] * s, size[1] * s
    if im is not None:
        return im.shape[-2] * s, im.shape[-1] * s
    else:
        return s


####### INIT ##########
im_rgb = R.cameras['RGB'].__getitem__(0).RGB('gray')
im_ir = R.cameras['IR'].__getitem__(0).RGB('gray')
scale = 2

matrix_rgb = R.cameras['RGB'].intrinsics[:, :3, :3]
matrix_ir = R.cameras['IR'].intrinsics[:, :3, :3]

# Get image resolution
wh_rgb = im_rgb.shape[-2:][1], im_rgb.shape[-2:][0]
wh_ir = im_ir.shape[-2:][1], im_ir.shape[-2:][0]

# Create draw tool with specific width and height window dimensions
draw = cv.Draw(wh=(wh_rgb[0] * 1.5, wh_rgb[1]), title='CamViz Pointcloud Demo')

# Create image screen to show the Fusion image
draw.add2Dimage('fus', luwh=(1 / 3, 0, 1.00, 1.00), res=wh_rgb)

# Create image screen to show the depth visualization
draw.add2Dimage('depth', luwh=(0.00, 0.50, 1 / 3, 1.00), res=scaled(scale, size=wh_ir))

pose_rgb = pose_from_extrinsics(R.cameras['RGB'].extrinsics)
pose_ir = pose_from_extrinsics(R.cameras['IR'].extrinsics)

# Create world screen at specific position inside the window (% left/up/right/down)
draw.add3Dworld('wld', luwh=(0.00, 0.00, 1 / 3, 0.50), pose=pose_rgb, K=matrix_rgb[0], wh=wh_rgb)
# pose=(7.25323, -3.80291, -5.89996, 0.98435, 0.07935, 0.15674, 0.01431))

# Create camera from intrinsics and image dimensions (width and height)
camera_rgb = cv.objects.Camera(K=matrix_rgb[0], wh=wh_rgb, pose=pose_rgb)
camera_ir = cv.objects.Camera(K=matrix_ir[0], wh=wh_ir, pose=pose_ir)

color_mode = 0
pose_mode = 0

for i in range(100):
    # Load Data
    start = time.time()
    im_rgb, idx = R.cameras['RGB'].random_image()
    im_rgb = im_rgb.RGB('gray')
    im_ir = R.cameras['IR'].__getitem__(idx).RGB('gray')
    load = time.time() - start
    start = time.time()

    depth = F.interpolate(NN(Tensor(im_ir), focal=matrix_ir[0, 0, 0])['metric_depth'].clip(0, 30),
                          scaled(scale, im=im_ir)).squeeze().detach().cpu().numpy()
    depth_pred = time.time() - start
    start = time.time()
    # Project depth maps from image (i) to camera (c) coordinates
    points = camera_ir.i2w(depth, scaled=scale)
    im_ir = im_ir.match_shape(scaled(scale, im=im_ir)).squeeze(0).permute([1, 2, 0]).cpu().numpy()
    depth_clr = cmap(depth).reshape(*im_ir.shape)
    im_rgb = im_rgb.squeeze(0).permute([1, 2, 0]).cpu().numpy()

    # Create pointcloud colors
    ir_clr = im_ir.reshape(-1, 3)  # Depth visualization colors
    hgt_clr = cmap(-points[:, 1])  # Height colors

    # Create RGB and visualization textures
    draw.addTexture('fus', im_rgb)  # Create texture buffer to store rgb image
    draw.addTexture('depth', depth_clr)  # Create texture buffer to store visualization image

    # Create buffers to store data for display
    draw.addBufferf('pts', points)  # Create data buffer to store depth points
    draw.addBufferf('ir', ir_clr)  # Create data buffer to store pointcloud heights
    draw.addBufferf('hgt', hgt_clr)  # Create data buffer to store pointcloud heights

    # Color & Pose dictionary
    color_dict = {0: 'ir', 1: 'hgt'}
    pose_dict = {0: pose_ir, 1: pose_rgb}
    proj_dict = {0: matrix_ir[0], 1: matrix_rgb[0]}
    wh_dict = {0: wh_ir, 1: wh_rgb}
    # Display loop
    while draw.input():
        # If RETURN is pressed, switch color mode
        if draw.RETURN:
            color_mode = (color_mode + 1) % len(color_dict)
            time.sleep(0.1)
        if draw.SPACE:
            pose_mode = (pose_mode + 1) % len(pose_dict)
            draw.scr('wld').viewer.setPose(pose_dict[pose_mode])
            draw.scr('wld').K = proj_dict[pose_mode]
            draw.scr('wld').wh = wh_dict[pose_mode]
            draw.scr('wld').calibrate()
            time.sleep(0.1)
        # Clear window
        draw.clear()
        # Draw points and colors from buffer
        draw['wld'].size(1).points('pts', color_dict[color_mode])
        # Draw image textures on their respective screens
        proj = ImageTensor(draw.to_image('wld'))
        im_fus = im_rgb / 2 + (proj.match_shape(im_rgb.shape[:2]).GRAYSCALE().RGB('inferno') / 2).to_numpy(
            datatype=np.float32)
        draw.addTexture('fus', im_fus)  # Create texture buffer to store fus image
        draw['fus'].image('fus')
        draw['depth'].image('depth')
        draw_buffers = time.time() - start
        # Draw camera with texture as image
        # draw['wld'].object(camera_rgb)
        # draw['wld'].object(camera_ir)
        # Update window
        if time_it:
            print(
                f'load data : {round(load, 4)} sec, process depth : {round(depth_pred, 4)} sec, draw buffers : {round(draw_buffers, 4)} sec, ')
        draw.update(1)
