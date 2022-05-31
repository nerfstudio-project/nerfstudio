"""
Test the fields
"""
import torch

from pyrad.nerf.field_modules.field_heads import FieldHeadNames
from pyrad.nerf.fields.instant_ngp_field import TCNNInstantNGPField
from pyrad.structures.rays import PointSamples


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
    point_samples = PointSamples(positions=positions, directions=directions)
    field_outputs = field.forward(point_samples)

    positions_dtype = positions.dtype
    field_outputs_density_dtype = field_outputs[FieldHeadNames.DENSITY].dtype
    assert (
        positions_dtype == field_outputs_density_dtype
    ), f"The input and output dtypes do not match: {positions_dtype} vs. {field_outputs_density_dtype}"


if __name__ == "__main__":
    test_tcnn_instant_ngp_field()
