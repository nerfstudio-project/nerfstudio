from typing import Literal, Tuple

import torch


def get_best_device() -> Tuple[Literal["cpu", "cuda", "mps"], str]:
    """Determine the best available device to run nerfstudio inference.

    Returns:
        tuple: (device_type, reason) where device_type is the selected device and
        reason is an explanation of why it was chosen
    """
    if torch.cuda.is_available():
        return "cuda", "CUDA GPU available - using for optimal performance"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "Apple Metal (MPS) available - using for accelerated performance"
    else:
        return "cpu", "No GPU/MPS detected - falling back to CPU"
