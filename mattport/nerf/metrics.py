"""
Collection of metrics.
"""

import torch
from pytorch_msssim import ssim, ms_ssim
from torchtyping import TensorType


def get_psnr(image_gt: TensorType, image_pred: TensorType) -> TensorType:
    """
    Returns the PSNR (peak signal-to-noise ratio) for an image. TODO: add masking

    Args:
        image_gt (TensorType): Ground truth image
        image_pred (TensorType): Predicted image
    Returns:
        float: PSNR
    """
    mean = torch.mean((image_gt - image_pred) ** 2)
    return -10 * torch.log10(mean)


def get_ssim(image_gt: TensorType, image_pred: TensorType) -> TensorType:
    """
    Returns the SSIM (structural similarty) for an image.

    Args:
        image_gt (TensorType): Ground truth image
        image_pred (TensorType): Predicted image
    Returns:
        float: SSIM
    """
    assert image_gt.shape[-1] in [1, 3]
    assert image_pred.shape[-1] in [1, 3]
    assert image_gt.shape == image_pred.shape
    if image_gt.dim() == 3:
        image_gt = image_gt[None, ...]
    if image_pred.dim() == 3:
        image_pred = image_pred[None, ...]

    return ssim(image_gt, image_pred, data_range=1.0)


def get_ms_ssim(image_gt: TensorType, image_pred: TensorType) -> TensorType:
    """
    Returns the MS_SSIM (multi-scale structural similarty) for an image.

    Args:
        image_gt (TensorType): Ground truth image
        image_pred (TensorType): Predicted image
    Returns:
        float: MS-SSIM
    """
    assert image_gt.shape[-1] in [1, 3]
    assert image_pred.shape[-1] in [1, 3]
    assert image_gt.shape == image_pred.shape
    if image_gt.dim() == 3:
        image_gt = image_gt[None, ...]
    if image_pred.dim() == 3:
        image_pred = image_pred[None, ...]

    return ms_ssim(image_gt, image_pred, data_range=1.0)
