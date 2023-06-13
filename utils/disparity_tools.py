import time
import cv2 as cv
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from kornia.geometry import hflip
from scipy.signal._signaltools import medfilt2d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, normalized_mutual_information


def reconstruction_from_disparity(image, disparity_matrix,
                                  inpainting=False, median=False, post_process=False, verbose=False):
    """
    Reconstruction function from disparity map.
    If the DISPARITY matrix is POSITIVE the IMAGE will be projected to the RIGHT (so the CAMERA to the LEFT)
    :return: reconstruct image.
    """
    start = time.time()
    # Disparity map hold between 0 and 1 with the min/max  disparity pass as arguments.
    # Left2right is TRUE if the IMAGE is projected to the right
    if abs(disparity_matrix).min() >= disparity_matrix.max():
        min_disp = disparity_matrix.max()
        max_disp = -(abs(disparity_matrix).max())
    else:
        min_disp = disparity_matrix.min()
        max_disp = disparity_matrix.max()
    disp = np.round(disparity_matrix.copy())
    if not (max_disp == min_disp):
        left2right = max_disp > min_disp
        dx = int(abs(max_disp))
        if image.max() <= 1:
            image_proc = np.uint8(image * 255)
        else:
            image_proc = np.uint8(image)
        if len(image_proc.shape) > 2:
            new_image = np.zeros([image_proc.shape[0], image_proc.shape[1] + dx, 3])
            disp_temp = np.stack([disp, disp, disp], axis=-1)
        else:
            new_image = np.ones([image_proc.shape[0], image_proc.shape[1] + dx]) * -1000
            disp_temp = disp.copy()
        if left2right:
            step = 1
        else:
            step = -1
        for k in range(int(min_disp), int(max_disp), step):
            mask = disp_temp == k
            if left2right:
                if k == dx:
                    new_image[:, dx:][mask] = image_proc[mask]
                else:
                    new_image[:, k:-dx + k][mask] = image_proc[mask]
            else:
                if k == 0:
                    new_image[:, dx:][mask] = image_proc[mask]
                else:
                    new_image[:, dx + k:k][mask] = image_proc[mask]
            if verbose:
                cv.imshow('New image', new_image / new_image.max())
                cv.waitKey(int(1000 / abs(max_disp)))
        if left2right:
            new_image_cropped = new_image[:, :-dx]
        else:
            new_image_cropped = new_image[:, dx:]
    else:
        new_image_cropped = image.copy()
    if verbose:
        print(f"    Reconstruction done in : {time.time() - start} seconds")
    '''
    Image post processing options
    '''
    if post_process:
        disparity_post_process(new_image_cropped, 0, 0, 200, verbose=False)
    elif median:
        start = time.time()
        new_image_cropped[new_image_cropped == -1000] = 0
        new_image_cropped[new_image_cropped == 0] = medfilt2d(new_image_cropped, kernel_size=5, )[
            new_image_cropped == 0]
        if verbose:
            print(f"    Median filtering done in : {round(time.time() - start, 2)} seconds")
    elif inpainting:
        start = time.time()
        if len(new_image_cropped.shape) > 2:
            mask = np.uint8(np.array(((cv.cvtColor(new_image_cropped.astype(np.uint8), cv.COLOR_BGR2GRAY) > 253) +
                                      (cv.cvtColor(new_image_cropped.astype(np.uint8), cv.COLOR_BGR2GRAY) < 1)) * 255))
        else:
            mask = np.uint8(np.array(((new_image_cropped > 253) + (new_image_cropped < 1)) * 255))
        new_image_cropped = cv.inpaint(np.uint8(new_image_cropped), mask, 15, cv.INPAINT_NS)
        if verbose:
            print(f"    Inpainting done in : {round(time.time() - start, 2)} seconds")
    if verbose:
        print(new_image_cropped.min(), new_image_cropped.max())
        cv.imshow('new image', new_image_cropped / new_image_cropped.max())
        cv.waitKey(0)
        cv.destroyAllWindows()

    return new_image_cropped


def error_estimation(image_translated, ref, ignore_undefined=True):
    if ignore_undefined:
        im1 = image_translated[image_translated > 0]
        im2 = ref[image_translated > 0]
    else:
        im1 = image_translated
        im2 = ref
    ssim_result, grad, s = ssim(im1, im2, data_range=im2.max() - im2.min(), gradient=True, full=True)
    rmse_result = mean_squared_error(im1, im2)
    nmi_result = normalized_mutual_information(im1, im2, bins=100)
    print(f"    Structural Similarity Index : {ssim_result}")
    print(f"    Root Mean squared error: {rmse_result}")
    print(f"    Normalized mutual information : {nmi_result}")
    if ignore_undefined:
        print(
            f"    Percentage of the image defined = {round(len(im1) / (image_translated.shape[0] * image_translated.shape[1]) * 100, 2)}%")
    cv.destroyAllWindows()


def disparity_post_process(disparity_matrix, min_disp, max_disp, threshold, verbose=False, median=3):
    start = time.time()
    if min_disp == 0 and max_disp == 0:
        disp = disparity_matrix.copy()
    else:
        disp = np.round(disparity_matrix * (max_disp - min_disp) + min_disp)
    if threshold == 0:
        threshold = 70
        mask_false_values = ((disp > threshold) + (disp < 1 + min_disp))
        disp[mask_false_values] = 0
    else:
        mask_false_values = ((disp > threshold) + (disp < 1 + min_disp))
        good_values = (disp * (mask_false_values == 0))
        sum_rows = good_values.sum(1)
        nb_values_sum = (good_values != 0).sum(1)
        nb_values_sum[nb_values_sum == 0] = 1
        ref = sum_rows / nb_values_sum
        ref = np.expand_dims(ref, axis=1) * np.ones([1, disparity_matrix.shape[1]])
        disp[mask_false_values] = ref[mask_false_values]
    if median:
        disp = medfilt2d(disp, kernel_size=median * 2 + 1)
    return disp


def find_occluded_pixel(disparity, device, upsample_factor=1, min_delta_disp=0):
    # disparity = disparity[1]

    while len(disparity.shape) < 4:
        disparity = disparity.unsqueeze(-1)  # [B H W 1]
    if disparity.shape[0] > 2:
        disparity = disparity.permute(2, 3, 0, 1)   # [B 1 H W]
    else:
        disparity = disparity.permute(0, 3, 1, 2)  # [B 1 H W]
    disp = (F.interpolate(disparity, scale_factor=upsample_factor,
                         mode='bilinear', align_corners=True) * upsample_factor).permute(0, 2, 3, 1)   # [B H' W' 1]
    b, h, w, _ = disp.shape
    grid_reg = kornia.utils.create_meshgrid(h, w, normalized_coordinates=False, device=device).to(
        disp.dtype)  # [1 H' W' 2]
    if b > 1:
        grid_reg = grid_reg.repeat(b, 1, 1, 1)  # [B H' W' 2]
    for idx in range(b):
        if disparity[idx].max() < 0:
            disp[idx] = hflip(disp[idx].unsqueeze(0).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(0)  # [H' W' 1]
        else:
            disp[idx] = -disp[idx]
    im = hflip(grid_reg[:, :, :, 0].unsqueeze(-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [B H' W' 1]
    grid_reg = torch.round(grid_reg[:, :, :, 0] + disp.squeeze(-1)).unsqueeze(-1)  # [B H' W' 1]
    # unique = torch.unique(grid_reg, sorted=False, dim=2)  # [B H' W' 1]
    # unique = F.interpolate(unique.permute(0, 3, 1, 2), scale_factor=1 / upsample_factor,
    #                      mode='bilinear', align_corners=True) # [B 1 H W]

    mask = torch.zeros_like(disp)  # [B H' W' 1]
    h_t = torch.tensor(range(h), device=device)
    h_t = h_t.repeat(b, 1)  # [B H']
    for i in range(w):
        tmp = (im * (grid_reg == i)).view(b, h, w, 1)   # [B H' W' 1]
        indice_tensor = torch.argmax(tmp, 2, keepdim=True).view(b, h)  # [B H']
        for j in range(b):
            h_tensor = h_t[j, (indice_tensor+i > i)[j, :]]  # [H']
            id_t = indice_tensor[j, (indice_tensor+i > i)[j, :]]  # [B H']
            mask[j, h_tensor, id_t, 0] = 1  # [B H' W' 1]
    mask = F.interpolate(mask.permute(0, 3, 1, 2), scale_factor=1/upsample_factor,
                         mode='bilinear', align_corners=True)  # [B 1 H W]
    for idx in range(b):
        if disparity[idx].max() < 0:
            mask[idx] = hflip(mask[idx].unsqueeze(0)).squeeze()
    return mask.squeeze()
    # if b > 1:
    #     mask_0 = hflip(mask[0].unsqueeze(0)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
    #     cv.imshow('mask0', np.uint8(mask_0 == 0) * 255)
    #     disp_im0 = hflip(disparity[0].unsqueeze(0)).permute(0, 2, 3, 1).squeeze().cpu().numpy()
    #     cv.imshow('disparity0', (mask_0 * disp_im0) / disp_im0.max())
    #     mask_1 = mask[1].squeeze().cpu().numpy()
    #     cv.imshow('mask1', np.uint8(mask_1 == 0)*255)
    #     disp_im1 = disparity[1].squeeze().cpu().numpy()
    #     cv.imshow('disparity1', (mask_1 * disp_im1)/disp_im1.max())
    # else:
    #     mask_0 = mask[0].squeeze().cpu().numpy()
    #     cv.imshow('mask0', np.uint8(mask_0 == 0) * 255)
    #     disp_im0 = disparity[0].squeeze().cpu().numpy()
    #     cv.imshow('disparity0', (mask_0 * disp_im0) / disp_im0.max())
    # cv.waitKey(0)



