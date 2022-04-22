"""
Collection of metrics.
"""

import torch


def get_psnr(image_gt, image_pred):
    """
    Returns the psnr for an image. TODO: add masking
    """
    mean = torch.mean((image_gt - image_pred) ** 2)
    return -10 * torch.log10(mean)
