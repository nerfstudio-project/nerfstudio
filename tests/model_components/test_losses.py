"""
Test losses
"""

import torch

from nerfstudio.model_components.losses import tv_loss


def test_tv_loss():
    """Test total variation loss"""

    grids = torch.ones([3, 64, 128, 128])
    assert tv_loss(grids).item() == 0

    grids = torch.zeros([1, 10, 100, 100])
    for i in range(100):
        for j in range(100):
            grid_location = (i + j) % 2
            for k in range(10):
                grids[0, k, i, j] = grid_location

    # TV_row = 1, TV_col = 1. Total tv should be 2 * (TV_row + TV_col)
    assert tv_loss(grids).item() == 4.0


if __name__ == "__main__":
    test_tv_loss()
