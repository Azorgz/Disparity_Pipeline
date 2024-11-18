#!/usr/bin/env python
import math

import cv2
import numpy as np
import os
import glob

import oyaml
from matplotlib import pyplot as plt
from scipy.stats import stats
from tqdm import tqdm

from utils.ImagesCameras import ImageTensor

######### CALIBRATION #########
# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
nb_matrix = 20
world_scaling = 0.15


# Functions for calibration
def calculate_dist_in_checker_points(corners, h=0, w=0, verbose=False):
    c = corners.copy()
    c[:, :, 1] = h - c[:, :, 1]
    dist_in_checker_points = []
    if verbose:
        plt.plot(c[:, 0, 0], c[:, 0, 1], 'o', color='k')
    for i in range(CHECKERBOARD[1]):
        idx = np.array([k for k in range(CHECKERBOARD[0])]) + CHECKERBOARD[0] * i
        line = np.array([c[id][0] for id in idx])
        res = np.polyfit(line[:, 1], line[:, 0], deg=1, full=True)
        if verbose:
            plt.plot(c[idx, 0, 0], c[idx, 0, 1], 'o')
            plt.plot(np.array([0, w]), -res[0][1] / res[0][0] + 1 / res[0][0] * np.array([0, w]))
        dist_in_checker_points.append(int(res[1][0]))
    for j in range(CHECKERBOARD[0]):
        idx = np.array([k for k in range(CHECKERBOARD[1])]) * CHECKERBOARD[0] + j
        line = np.array([c[id][0] for id in idx])
        res = np.polyfit(line[:, 0], line[:, 1], deg=1, full=True)
        if verbose:
            plt.plot(c[idx, 0, 0], c[idx, 0, 1], 'o')
            plt.plot(np.array([0, w]), res[0][1] + res[0][0] * np.array([0, w]))
        dist_in_checker_points.append(res[1][0])
    if verbose:
        plt.ylim([0, h])
        plt.xlim([0, w])
        plt.show()
    return np.array(dist_in_checker_points).sum()


def plot_corner(corners, w, h, img: list = None):
    # Draw and display the corners
    if img is not None:
        assert len(img) == len(corners)
    nrows, ncols = int(math.sqrt(len(corners))), int(math.sqrt(len(corners))) + 1
    if nrows * ncols < len(corners):
        nrows += 1
    fig = plt.figure(1)
    for idx, corner in enumerate(corners):
        corner[:, 1] = h - corner[:, 1]
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        if img is not None:
            ax.imshow(img[idx], extent=(0, w, 0, h))
        ax.plot(corner[:, 0], corner[:, 1], 'o', color='k')
        for i in range(CHECKERBOARD[1]):
            idx = np.array([k for k in range(CHECKERBOARD[0])]) + CHECKERBOARD[0] * i
            line = np.array([corner[id] for id in idx])
            res = np.polyfit(line[:, 1], line[:, 0], deg=1, full=True)
            ax.plot(corner[idx, 0], corner[idx, 1], 'o')
            ax.plot(np.array([0, w]), -res[0][1] / res[0][0] + 1 / res[0][0] * np.array([0, w]))
        for j in range(CHECKERBOARD[0]):
            idx = np.array([k for k in range(CHECKERBOARD[1])]) * CHECKERBOARD[0] + j
            line = np.array([corner[id] for id in idx])
            res = np.polyfit(line[:, 0], line[:, 1], deg=1, full=True)
            ax.plot(corner[idx, 0], corner[idx, 1], 'o')
            ax.plot(np.array([0, w]), res[0][1] + res[0][0] * np.array([0, w]))
        plt.ylim([0, h])
        plt.xlim([0, w])
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * world_scaling
cam_guess = {'visible_1': np.array([[1739.0, 0.0, 480.0], [0.0, 1739.0, 640.0], [0, 0, 1]], dtype=np.float32),
             'visible_2': np.array([[1739.0, 0.0, 480.0], [0.0, 1739.0, 640.0], [0, 0, 1]], dtype=np.float32),
             'infrared_1': np.array([[853.0, 0.0, 240.0], [0.0, 853.0, 320.0], [0, 0, 1]], dtype=np.float32),
             'infrared_2': np.array([[853.0, 0.0, 240.0], [0.0, 853.0, 320.0], [0, 0, 1]], dtype=np.float32)}
# Extracting path of individual image stored in a given directory
for i in [1, 3, 4]:
    new_path_infrared_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/infrared_1/"
    new_path_infrared_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/infrared_2/"
    new_path_visible_1 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/visible_1/"
    new_path_visible_2 = f"/media/godeta/T5 EVO/Datasets/Lynred/sequence_{i}/calib/visible_2/"
    paths = [new_path_infrared_1, new_path_infrared_2, new_path_visible_1, new_path_visible_2]

    ###############################################################
    res = {}
    objpoints_dict = {}
    imgpoints_dict = {}
    distance_tot = []
    sizes = {}

    for path, n in zip(paths, ['infrared_1', 'infrared_2', 'visible_1', 'visible_2']):
        # Creating vector to store vectors of 3D points for each checkerboard image

        # Creating vector to store vectors of 2D points for each checkerboard image
        imgpoints = []
        # Creating vector to store the index where the checkerboard was found
        idx_valid = []
        # Creating vector to store the distance from a perfect grid
        distance = []

        # prev_img_shape = None
        for idx, fname in tqdm(enumerate(sorted(os.listdir(path)))):
            img = ImageTensor(f'{path}/{fname}')
            if 'infrared' in n:
                img = img.pyrUp()
            gray = img.GRAY().to_opencv(datatype=np.uint8)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            """
            If desired number of corner are detected,
            we refine the pixel coordinates and display
            them on the images of checker board
            """
            # objpoints.append(objp)
            if ret == True:
                # refining pixel coordinates for given 2d points.
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                if 'infrared' in n:
                    corners /= 2
                    w, h = int(gray.shape[1] / 2), int(gray.shape[0] / 2)
                else:
                    w, h = int(gray.shape[1]), int(gray.shape[0])
                sizes[n] = (w, h)
                imgpoints.append(corners)
                distance.append(calculate_dist_in_checker_points(corners, h, w))
                idx_valid.append(idx)
            else:
                imgpoints.append(0)
                # if idx % 100 == 0:
                #     # Draw and display the corners
                #     img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
                #     cv2.imshow('img', img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
        """
        Performing camera calibration by
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the
        detected corners (imgpoints)
        """
        if 'infrared' in n:
            img = img.pyrDown()
        imgpoints_dict[n] = imgpoints
        l = len(os.listdir(path))
        if not distance_tot:
            distance_tot = ({i: dist for i, dist in zip(idx_valid, distance)} |
                            {i: 1000 for i in range(l) if i not in idx_valid})
        else:
            new_dist = ({i: dist for i, dist in zip(idx_valid, distance)} |
                        {i: 1000 for i in range(l) if i not in idx_valid})
            distance_tot = {i: distance_tot[i] + new_dist[i] for i in range(l)}
    distance = np.argsort([distance_tot[i] for i in range(l)])[:min(l - 1, nb_matrix)]
    objpoints = np.array([objp[0]] * len(distance))
    for path, n in zip(paths, ['infrared_1', 'infrared_2', 'visible_1', 'visible_2']):
        # distance = np.argsort(distance)[:min(len(distance), nb_matrix)]
        print(f"{len(imgpoints_dict[n])} checkerboard detected, {len(distance)} best kept for calibration...")
        imgpoints = np.array([imgpoints_dict[n][d] for d in distance])[:, :, 0, :]
        images = [cv2.imread(path + im) for im in np.array(sorted(os.listdir(path)))[distance.astype(int)]]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, sizes[n], cam_guess[n],
                                                           np.array([[0, 0, 0, 0, 0]], dtype=np.float32),
                                                           flags=cv2.CALIB_ZERO_TANGENT_DIST +
                                                                 cv2.CALIB_USE_INTRINSIC_GUESS +
                                                                 cv2.CALIB_FIX_S1_S2_S3_S4 +
                                                                 cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 +
                                                                 cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6)
        plot_corner(imgpoints, *sizes[n], images)
        print(mtx)
        res[n] = {'mtx': mtx.tolist(),
                  'dist': dist.tolist(),
                  'rvecs': [r.tolist() for r in rvecs],
                  'tvecs': [t.tolist() for t in tvecs]}
    res['valid_idx'] = distance.tolist()
    with open(f'/home/godeta/PycharmProjects/Disparity_Pipeline/res{i}.yaml', "w") as file:
        oyaml.dump(res, file)

    ###############################################################
    # with open(f'/home/godeta/PycharmProjects/Disparity_Pipeline/res{i}.yaml', "r") as file:
    #     res = oyaml.safe_load(file)
    # infrared1_R = 0
    # infrared1_T = 0
    # infrared2_R = 0
    # infrared2_T = 0
    # visible2_R = 0
    # visible2_T = 0
    # for num in range(len(res['visible_2']['tvecs'])):
    #     infrared1_R += np.array(res['infrared_1']['rvecs'][num]) - np.array(res['visible_1']['rvecs'][num])
    #     infrared1_T += np.array(res['infrared_1']['tvecs'][num]) - np.array(res['visible_1']['tvecs'][num])
    #     infrared2_R += np.array(res['infrared_2']['rvecs'][num]) - np.array(res['visible_1']['rvecs'][num])
    #     infrared2_T += np.array(res['infrared_2']['tvecs'][num]) - np.array(res['visible_1']['tvecs'][num])
    #     visible2_R += np.array(res['visible_2']['rvecs'][num]) - np.array(res['visible_1']['rvecs'][num])
    #     visible2_T += np.array(res['visible_2']['tvecs'][num]) - np.array(res['visible_1']['tvecs'][num])
    # infrared1_R /= len(res['visible_2']['tvecs'])
    # infrared1_T /= len(res['visible_2']['tvecs'])
    # infrared2_R /= len(res['visible_2']['tvecs'])
    # infrared2_T /= len(res['visible_2']['tvecs'])
    # visible2_R /= len(res['visible_2']['tvecs'])
    # visible2_T /= len(res['visible_2']['tvecs'])
    #
    # infrared1 = np.vstack([np.hstack([cv2.Rodrigues(infrared1_R)[0], infrared1_T]), np.array([0, 0, 0, 1])])
    # infrared2 = np.vstack([np.hstack([cv2.Rodrigues(infrared2_R)[0], infrared2_T]), np.array([0, 0, 0, 1])])
    # visible2 = np.vstack([np.hstack([cv2.Rodrigues(visible2_R)[0], visible2_T]), np.array([0, 0, 0, 1])])
    #
    # ######## CAMERA SETUP CREATION #################################
    # import torch
    # from utils.ImagesCameras import Camera
    # from utils.ImagesCameras import CameraSetup
    #
    # name_path = '/'
    # perso = '/home/aurelien/Images/Images_LYNRED/'
    # pro = '/media/godeta/T5 EVO/Datasets/Lynred/'
    #
    # p = pro if 'godeta' in os.getcwd() else perso
    #
    # path_RGB = p + f'/sequence_{i}/visible_1/'
    # path_RGB2 = p + f'/sequence_{i}/visible_2/'
    # path_IR = p + f'/sequence_{i}/infrared_1/'
    # path_IR2 = p + f'/sequence_{i}/infrared_2/'
    #
    # IR = Camera(path=path_IR, device=torch.device('cuda'), id='IR', f=14, name='SmartIR640',
    #             intrinsics=np.array(res['infrared_1']['mtx']))
    # IR2 = Camera(path=path_IR2, device=torch.device('cuda'), id='IR2', f=14, name='subIR',
    #              intrinsics=np.array(res['infrared_2']['mtx']))
    # RGB = Camera(path=path_RGB, device=torch.device('cuda'), id='RGB', f=6, name='mainRGB',
    #              intrinsics=np.array(res['visible_1']['mtx']))
    # RGB2 = Camera(path=path_RGB2, device=torch.device('cuda'), id='RGB2', f=6, name='subRGB',
    #               intrinsics=np.array(res['visible_2']['mtx']))
    #
    # R = CameraSetup(RGB, IR, IR2, RGB2, print_info=True)
    #
    # R.update_camera_relative_position('IR', extrinsics=torch.tensor(infrared1, dtype=torch.double))
    # R.update_camera_relative_position('RGB2', extrinsics=torch.tensor(visible2, dtype=torch.double))
    # R.update_camera_relative_position('IR2', extrinsics=torch.tensor(infrared2, dtype=torch.double))
    # R.calibration_for_stereo('RGB', 'RGB2')
    # R.stereo_pair('RGB', 'RGB2').show_image_calibration()
    #
    # R.calibration_for_stereo('IR', 'RGB')
    # R.calibration_for_stereo('IR', 'RGB2')
    # R.stereo_pair('RGB', 'IR').show_image_calibration()
    #
    # R.calibration_for_stereo('IR', 'IR2')
    # R.stereo_pair('IR', 'IR2').show_image_calibration()
    #
    # R.calibration_for_depth('IR', 'IR2')
    # R.calibration_for_depth('RGB', 'RGB2')
    #
    # print(R)
    #
    # perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/Setup_Camera/'
    # pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/Setup_Camera/'
    # p = pro if 'godeta' in os.getcwd() else perso
    # path_result = p + name_path
    # R.save(path_result, f'Lynred_seq{i}.yaml')
