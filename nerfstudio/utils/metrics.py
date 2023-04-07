"""
This file contains NeRF metrics with masking capabilities.
"""

from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# TODO: masked PSNR, SSIM, LPIPS