"""
Test the fields
"""
import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.fields.instant_ngp_field import TCNNInstantNGPField


def test_tcnn_instant_ngp_field():
    """Test the tiny-cuda-nn field"""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=unused-import
    try:
        import tinycudann as tcnn
    except ImportError as e:
        # tinycudann module doesn't exist
        print(e)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aabb_scale = 1.0
    aabb = torch.tensor(
        [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
    ).to(device)
    field = TCNNInstantNGPField(aabb)
    num_rays = 1024
    num_samples = 256
    positions = torch.rand((num_rays, num_samples, 3), dtype=torch.float32, device=device)
    directions = torch.rand_like(positions)
    frustums = Frustums(
        origins=positions,
        directions=directions,
        starts=torch.zeros((*directions.shape[:-1], 1), device=device),
        ends=torch.zeros((*directions.shape[:-1], 1), device=device),
        pixel_area=torch.ones((*directions.shape[:-1], 1), device=device),
    )
    ray_samples = RaySamples(frustums=frustums)
    field.forward(ray_samples)


if __name__ == "__main__":
    test_tcnn_instant_ngp_field()
