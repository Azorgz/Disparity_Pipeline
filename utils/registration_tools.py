import os
import cv2 as cv
import timeit
import numpy as np

from utils.classes import ImageTensor
from utils.classes.Geometry.transforms import ToTensor, ToFloatTensor, Resize
from utils.visualization import vis_disparity


def automatic_keypoints_selection(im_src: ImageTensor, ref: ImageTensor, pts_ref=None) -> tuple:
    delta = [0, 200, 0, 5]

    height, width = im_src.shape[-2:]
    pts_src, pts_dst = SIFT(ref, im_src, MIN_MATCH_COUNT=4, matcher='FLANN', lowe_ratio=0.5,
                            nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6,
                            toggle_evaluate=False, verbose=True, delta=delta)
    warp_matrix_rotation, _ = cv.findHomography(pts_dst, pts_src)
    CutY = -int(warp_matrix_rotation[0, 2])
    CutZ = -int(warp_matrix_rotation[1, 2])
    CutTop, CutLeft, CutBot, CutRight = AllSideCrop_from_2(CutY, CutZ)

    imL_aligned = cv.warpPerspective(imL, warp_matrix_rotation, (width, height),
                                     flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    imR_aligned = ref
    imL_aligned, imR_aligned = crop_from_cut(imL_aligned, imR_aligned, (CutTop, CutLeft, CutBot, CutRight), idx=2)
    # cv.imwrite(os.path.join(path_save, 'calib', 'left.png'), np.uint8(imL_aligned))
    # cv.imwrite(os.path.join(path_save, 'calib', 'right.png'), np.uint8(imR_aligned))
    return warp_matrix_rotation, [CutTop, CutLeft, CutBot, CutRight]


def manual_position_calibration_v2(imL, imR):
    imL = imL.RGB().numpy()[:, :, [2, 1, 0]]
    imR = imR.RGB().numpy()[:, :, [2, 1, 0]]
    name_window = "Position Calibration: Please align the BACKGROUND, valid with Escape"
    create_position_windows(name_window)
    while True:
        smpl = [imL.copy(), imR.copy()]
        # Updating the parameters based on the trackbar positions
        matrix_translation = update_warp_matrix_translation(name_window)
        smpl = warp_case(smpl, 1, np.float64(matrix_translation), 0, 0, 0, 0)
        final = smpl[0] / 2 + smpl[1] / 2
        cv.imshow(name_window, final)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()
    T_ref = matrix_translation[0][2]
    name_window = "Position Calibration: Please align the FOREGROUND, valid with Escape"
    create_position_windows(name_window)
    while True:
        smpl = [imL.copy(), imR.copy()]
        # Updating the parameters based on the trackbar positions
        matrix_translation = update_warp_matrix_translation(name_window)
        smpl = warp_case(smpl, 1, np.float64(matrix_translation), 0, 0, 0, 0)
        final = smpl[0] / 2 + smpl[1] / 2
        cv.imshow(name_window, final)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()
    T = matrix_translation[0][2] - T_ref
    inverted = True if T < 0 else False
    return inverted


def manual_registration_v2(imL, imR, model, idx):
    imL = imL.RGB().numpy()[:, :, [2, 1, 0]]
    imR = imR.RGB().numpy()[:, :, [2, 1, 0]]
    name_window = "Registration: Please align the background, valid with Escape"
    create_registration_windows(name_window, imL.shape[0], imL.shape[1])
    disp_vis = True if model is not None else False
    line_nb = 10
    line_activate = True
    while True:
        smpl = [imL.copy(), imR.copy()]
        # Updating the parameters based on the trackbar positions
        warp_matrix_rotation, cut, center = update_warp_matrix_rotation(name_window)
        smpl = warp_case_v2(smpl, int(idx), warp_matrix_rotation, *cut, center)
        if disp_vis:
            disp = disparity_for_display(smpl[0], smpl[1], model)
            final = np.hstack([smpl[0] / 2 + smpl[1] / 2, disp])
        else:
            final = smpl[0] / 2 + smpl[1] / 2
        if line_activate:
            interline = final.shape[0] / line_nb
            line = [[(0, int(interline*(i + 0.5))), (final.shape[1], int(interline * (i + 0.5)))] for i in range(line_nb)]
            for l in line:
                final = cv.line(final, l[0], l[1], (0, 0, 255), 3)
        cv.imshow(name_window, final)
        key = cv.waitKey(1)
        if key == 27:
            break
        if (key == 118 or key == 100) and model is not None:  # d or v to switch disparity display
            disp_vis = not disp_vis
        if key == 108:  # l
            line_activate = not line_activate
        if key == 43:  # +
            line_nb += 1
        if key == 45:  # 1
            line_nb -= 1
            line_nb = max(line_nb, 1)
    cv.destroyAllWindows()
    return warp_matrix_rotation, [*cut]


def manual_registration(sample, model, process, cut):
    imL, imR = sample[process[0]], sample[process[1]]
    idx = process[2]
    if imL.cmap != 'GRAYSCALE':
        imL = np.float32(imL.BGR() / 255)
    else:
        imL = np.float32(imL.copy() / 255)
    if imR.cmap != 'GRAYSCALE':
        imR = np.float32(imR.BGR() / 255)
    else:
        imR = np.float32(imR.copy() / 255)
    name_window = "Registration: Please align the background, valid with Escape"
    create_registration_windows(name_window)
    while True:
        smpl = [imL.copy(), imR.copy()]
        # Updating the parameters based on the trackbar positions
        warp_matrix_rotation, CutTop, CutLeft, CutBot, CutRight = update_warp_matrix_rotation(name_window)
        smpl = warp_case(smpl, int(idx), warp_matrix_rotation,
                         CutTop + cut[0], CutLeft + cut[1], CutBot + cut[2], CutRight + cut[3])
        disp = disparity_for_display(smpl[0], smpl[1], model)
        final = np.hstack([smpl[0] / 2 + smpl[1] / 2, disp])
        cv.imshow(name_window, final)
        if cv.waitKey(1) == 27:
            break
    cv.destroyAllWindows()
    return warp_matrix_rotation, [CutTop + cut[0], CutLeft + cut[1], CutBot + cut[2], CutRight + cut[3]]


def manual_position_calibration(imL, imR, pos):
    if imL.cmap != 'GRAYSCALE':
        imL = np.float32(imL.BGR() / 255)
    else:
        imL = np.float32(imL.copy() / 255)
    if imR.cmap != 'GRAYSCALE':
        imR = np.float32(imR.BGR() / 255)
    else:
        imR = np.float32(imR.copy() / 255)
    name_window = "Position Calibration: Please align the BACKGROUND, valid with Escape"
    create_position_windows(name_window)
    while True:
        smpl = [imL.copy(), imR.copy()]
        # Updating the parameters based on the trackbar positions
        matrix_translation = update_warp_matrix_translation(name_window)
        smpl = warp_case(smpl, 1, np.float64(matrix_translation), 0, 0, 0, 0)
        final = smpl[0] / 2 + smpl[1] / 2
        cv.imshow(name_window, final)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()
    T_ref = matrix_translation[0][2]
    name_window = "Position Calibration: Please align the FOREGROUND, valid with Escape"
    create_position_windows(name_window)
    while True:
        smpl = [imL.copy(), imR.copy()]
        # Updating the parameters based on the trackbar positions
        matrix_translation = update_warp_matrix_translation(name_window)
        smpl = warp_case(smpl, 1, np.float64(matrix_translation), 0, 0, 0, 0)
        final = smpl[0] / 2 + smpl[1] / 2
        cv.imshow(name_window, final)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()
    T = matrix_translation[0][2] - T_ref
    inverted = True if T < 0 else False
    if pos[0] == 0:
        pos[0] = T
    else:
        pos[1] = T
    return pos, inverted


def disparity_for_display(imL, imR, model):
    if model.name == "unimatch":
        convert = ToTensor(no_normalize=True)
    elif model.name == "acvNet":
        convert = ToFloatTensor(no_normalize=True)
    else:
        convert = ToTensor()
    device = next(model.model.parameters()).device
    resize = Resize([256, 320], 0)
    smpl = {"left": imL, "right": imR}
    smpl = resize(convert(smpl, device))
    disp = model(smpl["left"], smpl["right"])
    resize_back = Resize_disp([resize.ori_size[0], resize.ori_size[1]])
    disp = resize_back(disp, device)
    return vis_disparity(disp.squeeze().cpu().numpy()) / 255


def create_position_windows(name):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 1280, 720)
    cv.createTrackbar('Translation Y', name, 250, 500, nothing)


def update_warp_matrix_translation(name):
    T = cv.getTrackbarPos('Translation Y', name) - 250
    return [[1, 0, T], [0, 1, 0], [0, 0, 1]]


def create_registration_windows(name, *args):
    shape_x, shape_y = args[0], args[1]
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 1280, 720)
    cv.createTrackbar('Rotation Z', name, 250, 500, nothing)
    cv.createTrackbar('Rotation X', name, 250, 500, nothing)
    cv.createTrackbar('Rotation Y', name, 250, 500, nothing)
    cv.createTrackbar('Translation Z', name, 250, 500, nothing)
    cv.createTrackbar('Translation Y', name, 250, 500, nothing)
    cv.createTrackbar('Translation X', name, 250, 500, nothing)
    cv.createTrackbar('Cut Top', name, 0, 100, nothing)
    cv.createTrackbar('Cut Left', name, 0, 500, nothing)
    cv.createTrackbar('Cut Bottom', name, 0, 100, nothing)
    cv.createTrackbar('Cut Right', name, 0, 500, nothing)
    cv.createTrackbar('Center X', name, 0, shape_x, nothing)
    cv.createTrackbar('Center Y', name, 0, shape_y, nothing)


def update_warp_matrix_rotation(name):
    Rz = (cv.getTrackbarPos('Rotation Z', name) - 250) / 10 ** 6
    Rx = (cv.getTrackbarPos('Rotation X', name) - 250) / 10 ** 6
    Ry = (cv.getTrackbarPos('Rotation Y', name) - 250) / 10 ** 4
    Tz = cv.getTrackbarPos('Translation Z', name) - 250
    Ty = cv.getTrackbarPos('Translation Y', name) - 250
    Tx = (cv.getTrackbarPos('Translation X', name)) / 500 + 0.5
    CutTop = cv.getTrackbarPos('Cut Top', name)
    CutLeft = cv.getTrackbarPos('Cut Left', name)
    CutBot = cv.getTrackbarPos('Cut Bottom', name)
    CutRight = cv.getTrackbarPos('Cut Right', name)
    center_x = cv.getTrackbarPos('Center X', name)
    center_y = cv.getTrackbarPos('Center Y', name)
    warp_matrix_rotation, _ = cv.Rodrigues(np.array([Rx, Rz, Ry]))
    warp_matrix_rotation[0, 0] = Tx
    warp_matrix_rotation[1, 1] = Tx
    warp_matrix_rotation[0, 2] = Ty
    warp_matrix_rotation[1, 2] = Tz
    return warp_matrix_rotation, (CutTop, CutLeft, CutBot, CutRight), (center_x, center_y)


def warp_case(smpl, process_idx: int, matrix, CutTop, CutLeft, CutBot, CutRight):
    height, width = smpl[process_idx].shape[:2]
    smpl[process_idx] = cv.warpPerspective(smpl[process_idx], matrix, (width, height), flags=cv.INTER_LINEAR)
    smpl = crop_from_cut(smpl[0], smpl[1], (CutTop, CutLeft, CutBot, CutRight))
    return smpl


def warp_case_v2(smpl, process_idx: int, matrix, CutTop, CutLeft, CutBot, CutRight, center):
    im = smpl[process_idx]
    height, width = smpl[process_idx].shape[:2]
    x = np.linspace(0, width, width)
    y = np.linspace(0, height, height)
    mapX, mapY = np.meshgrid(x, y)
    mapZ = np.ones_like(mapX)
    mapX, mapY = mapX - center[1], mapY - center[0]
    grid = np.stack([mapX, mapY, mapZ])
    grid_transformed = (matrix @ (np.transpose(grid, (1, 0, 2)))).transpose((1, 0, 2))
    a = grid_transformed[2].copy()
    grid_transformed = grid_transformed / a

    mapX, mapY = cv.convertMaps(grid_transformed[0].astype(np.float32), grid_transformed[1].astype(np.float32),
                                cv.CV_32FC1)
    mapX, mapY = mapX + center[1], mapY + center[0]
    smpl[process_idx] = cv.remap(im, mapX, mapY, cv.INTER_LINEAR)
    smpl = crop_from_cut(smpl[0], smpl[1], (CutTop, CutLeft, CutBot, CutRight))
    return smpl


def crop_from_cut(imL, imR, cut: tuple or list):
    """
    :param imL_aligned: /
    :param imR_aligned: /
    :param idx: 1 for the first window cut, 2 for the second one
    :param cut: a Tuple, CutY, CutZ if idx 1, Top, Left, Bottom, Right if index 2
    :return:
    """
    if len(cut) == 2:
        CutY, CutZ = cut[0], cut[1]
        CutTop, CutLeft, CutBot, CutRight = AllSideCrop_from_2(CutY, CutZ)
    else:
        CutTop, CutLeft, CutBot, CutRight = cut[0], cut[1], cut[2], cut[3]
    # imL ####
    if imL is not None:
        if CutTop > 0:
            imL = imL[CutTop:, :]
        if CutLeft > 0:
            imL = imL[:, CutLeft:]
        if CutBot > 0:
            imL = imL[:-CutBot, :]
        if CutRight > 0:
            imL = imL[:, :-CutRight]
    # imR ####
    if imR is not None:
        if CutTop > 0:
            imR = imR[CutTop:, :]
        if CutLeft > 0:
            imR = imR[:, CutLeft:]
        if CutBot > 0:
            imR = imR[:-CutBot, :]
        if CutRight > 0:
            imR = imR[:, :-CutRight]
    if imL is not None and imR is not None:
        return imL, imR
    elif imL is not None:
        return imL
    elif imR is not None:
        return imR
    else:
        return None


def AllSideCrop_from_2(CutY, CutZ):
    if CutZ == 0:
        CutBot, CutTop = 0, 0
    elif CutZ > 0:
        CutBot, CutTop = 0, CutZ
    else:
        CutBot, CutTop = -CutZ, 0
    if CutY == 0:
        CutLeft, CutRight = 0, 0
    elif CutY > 0:
        CutLeft, CutRight = CutY, 0
    else:
        CutLeft, CutRight = 0, -CutY
    return CutTop, CutLeft, CutBot, CutRight


# Semi-Automatic method

def automatic_registration(sample, path_save, data_type):
    delta = [0, 200, 0, 5]

    if data_type == '2vis':
        imL = sample['left'].BGR()
        imR = sample['right'].BGR()
    else:
        imL = sample['left'].copy()
        imR = sample['right'].copy()
    height, width = imL.shape[:2]
    pts_src, pts_dst = SIFT(imR, imL, MIN_MATCH_COUNT=4, matcher='FLANN', lowe_ratio=0.5,
                            nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6,
                            toggle_evaluate=False, verbose=True, delta=delta)
    warp_matrix_rotation, _ = cv.findHomography(pts_dst, pts_src)
    CutY = -int(warp_matrix_rotation[0, 2])
    CutZ = -int(warp_matrix_rotation[1, 2])
    CutTop, CutLeft, CutBot, CutRight = AllSideCrop_from_2(CutY, CutZ)

    imL_aligned = cv.warpPerspective(imL, warp_matrix_rotation, (width, height),
                                     flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
    imR_aligned = imR
    imL_aligned, imR_aligned = crop_from_cut(imL_aligned, imR_aligned, (CutTop, CutLeft, CutBot, CutRight), idx=2)
    cv.imwrite(os.path.join(path_save, 'calib', 'left.png'), np.uint8(imL_aligned))
    cv.imwrite(os.path.join(path_save, 'calib', 'right.png'), np.uint8(imR_aligned))
    return warp_matrix_rotation, [CutTop, CutLeft, CutBot, CutRight]


def SIFT(image_dst, image_src, MIN_MATCH_COUNT=4, matcher='FLANN', lowe_ratio=0.5,
         nfeatures=0, nOctaveLayers=4, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6, toggle_evaluate=False,
         verbose=False, delta=None):
    if delta is None:
        delta_x_min = 0
        delta_x_max = 1000
        delta_y_min = 0
        delta_y_max = 1000
    else:
        delta_x_min = delta[0]
        delta_x_max = delta[1]
        delta_y_min = delta[2]
        delta_y_max = delta[3]
    ## Image in grayscale
    gray_dst = image_dst.GRAYSCALE()
    gray_src = image_src.GRAYSCALE()
    # Initiate SIFT detector
    sift = cv.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                          edgeThreshold=edgeThreshold, sigma=sigma)
    # find the keypoints and descriptors with SIFT
    tic = timeit.default_timer()
    kp_dst, des_dst = sift.detectAndCompute(gray_dst, None)
    kp_src, des_src = sift.detectAndCompute(gray_src, None)
    toc = timeit.default_timer()
    if toggle_evaluate:
        t1 = round(toc - tic, 2)
    elif verbose:
        print(
            f"Number of keypoints found in source image : {len(kp_src)}, in the destination image : {len(kp_dst)} in {round(toc - tic, 2)} secondes")
    ## Initialize the matcher and match the keypoints
    tic = timeit.default_timer()
    matcher = init_matcher(method='SIFT', matcher=matcher, trees=5)
    matches = matcher.knnMatch(des_dst, des_src, k=2)
    toc = timeit.default_timer()
    if toggle_evaluate:
        t2 = round(toc - tic, 2)
    elif verbose:
        print(f"Matches computed in {round(toc - tic, 2)} secondes")
    # store all the good matches as per Lowe's ratio test.
    good = []
    src_pts = []
    dst_pts = []
    dist = []
    if lowe_ratio == 0:
        ratio_matches = 0.4
        k = 0.005
        while len(good) < MIN_MATCH_COUNT and ratio_matches < 0.95:
            good = []
            ratio_matches += k
            for m, n in matches:
                if m.distance < ratio_matches * n.distance:
                    src_pts.append(np.float32(kp_src[m.trainIdx].pt).reshape(-1, 2))
                    dst_pts.append(np.float32(kp_dst[m.queryIdx].pt).reshape(-1, 2))
    else:
        ratio_matches = lowe_ratio
        for m, n in matches:
            if m.distance < ratio_matches * n.distance:
                src_temp = np.float32(kp_src[m.trainIdx].pt).reshape(-1, 2)  # (x, y)
                dst_temp = np.float32(kp_dst[m.queryIdx].pt).reshape(-1, 2)  # (x, y)
                dist_x = abs(src_temp[0][0] - dst_temp[0][0])
                dist_y = abs(src_temp[0][1] - dst_temp[0][1])
                if delta_y_max >= dist_y > delta_y_min and delta_x_max >= dist_x > delta_x_min:
                    good.append(m)
                    src_pts.append(src_temp)
                    dst_pts.append(dst_temp)
                    dist.append(dist_x)
        idx = []
        factor = 0.98
        while len(idx) < MIN_MATCH_COUNT and factor > 0.9:
            idx = dist > max(dist) * factor
            factor = factor - 0.01
        dist, src_pts, dst_pts, good = np.array(dist)[idx], np.array(src_pts)[idx], np.array(dst_pts)[idx], \
            np.array(good)[idx]
    pts_src = np.int32(src_pts)
    pts_dst = np.int32(dst_pts)
    if not len(src_pts) >= MIN_MATCH_COUNT:
        #     src_pts = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 2)
        #     dst_pts = np.float32([kp_dst[m.queryIdx].pt for m in good]).reshape(-1, 2)
        #     pts_src = np.int32(src_pts)
        #     pts_dst = np.int32(dst_pts)
        # else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")

    if toggle_evaluate:
        return len(kp_src), len(kp_dst), len(src_pts), t1, t2
    elif verbose:
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=None,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(image_dst, kp_dst, image_src, kp_src, good, None, **draw_params)
        cv.putText(img3, f'Threshold min : {delta_x_min}, threshold max :{delta_x_max}', (10, 920),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 0, 0), 0)
        if img3.shape[0] > 500:
            img3 = cv.pyrDown(img3)
        cv.imshow('matched points', img3)
        cv.waitKey(0)
    return pts_src, pts_dst


def init_matcher(method='SIFT', matcher='FLANN', trees=5):
    FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_COMPOSITE = 3
    FLANN_INDEX_KDTREE_SINGLE = 4
    FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6
    FLANN_INDEX_SAVED = 254
    FLANN_INDEX_AUTOTUNED = 255

    if method == 'SIFT':
        norm = cv.NORM_L2
    elif method == 'RootSIFT':
        norm = cv.NORM_L2
    elif method == 'ORB':
        norm = cv.NORM_HAMMING
    elif method == 'BRISK':
        norm = cv.NORM_HAMMING

    if matcher == 'FLANN':
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=trees)
        else:
            flann_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12  The number of hash tables to use
                                key_size=6,  # 20  The length of the key in the hash tables
                                multi_probe_level=0)  # 2 Number of levels to use in multi-probe (0 for standard LSH)
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    elif matcher == 'BF':
        matcher = cv.BFMatcher(norm)
    else:
        flann_params = dict(algorithm=FLANN_INDEX_LINEAR)
        matcher = cv.FlannBasedMatcher(flann_params, {})
    return matcher


def nothing(x):
    pass
