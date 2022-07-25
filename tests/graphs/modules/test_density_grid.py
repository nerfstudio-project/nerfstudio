"""
Code to test the density grid.
"""

# import torch

# from nerfactory.fields.density_fields.density_grid import DensityGrid


def test_density_grid():
    """For testing the density grid."""
    # TODO(ruilongli): currently the DensityGrid only supports GPU init.
    # after we support for CPU init then we re-enable this test

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_cascades = 1
    # resolution = 128
    # aabb = torch.tensor([[-1, -1, -1], [1, 1, 1]], dtype=torch.float32).to(device)
    # grid = DensityGrid(num_cascades, resolution, aabb)
    # grid.to(device)
    # density_eval_func = lambda x: x[..., 0:1]  # xyz (..., 3) => density (...,)
    # grid.update_density_grid(density_eval_func=density_eval_func, step=0)


if __name__ == "__main__":
    test_density_grid()
