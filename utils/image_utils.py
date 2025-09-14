#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim # pip install scikit-image
from lpips import lpips  # pip install lpips

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def iou(pred, gt):
    intersection = np.sum(np.logical_and(gt, pred))
    union = np.sum(np.logical_or(gt, pred))
    iou = (np.sum(intersection) / np.sum(union))
    return iou

def calculate_ssim(img_fake, img_real):
    """
    img_fake 和 img_real 是PIL.Image
    """
    img_fake = np.array(img_fake)
    img_real = np.array(img_real)
    ssim_value, _ = ssim(img_fake, img_real, full=True, channel_axis=2)
    return ssim_value

def calculate_lpips(lpips_model, img_pred, img_gt):
    """
    img_pred 和 img_gt 是PIL.Image
    """
    img_fake = lpips.im2tensor(np.array(img_pred)).to("cuda")
    img_real = lpips.im2tensor(np.array(img_gt)).to("cuda")
    lpips_value = lpips_model(img_fake, img_real).item()
    return lpips_value

def calculate_iou(pred, gt):
    """
    pred 和 gt 是PIL.Image
    """
    pred = np.array(pred)
    gt = np.array(gt)
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calcluate_psnr(pred, gt):
    """
    pred 和 gt 是PIL.Image
    """
    pred = np.array(pred)
    gt = np.array(gt)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

