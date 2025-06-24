"""
Test if temporal distortions run properly
"""

import torch

from nerfstudio.field_components.temporal_distortions import DNeRFDistortion


def test_dnerf_distortion():
    """Test dnerf distortion"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distortion = DNeRFDistortion().to(device)

    num_rays = 1024
    num_samples = 256
    positions = torch.rand((num_rays, num_samples, 3), dtype=torch.float32, device=device)
    times = torch.rand_like(positions[..., :1])
    out = distortion.forward(positions, times)
    assert out is not None
    assert out.shape == positions.shape


if __name__ == "__main__":
    test_dnerf_distortion()
