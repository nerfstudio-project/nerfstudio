"""
Encoding Tests
"""

import pytest
import torch

from nerfstudio.field_components import encodings


def test_scaling_and_offset():
    """Test scaling and offset encoder"""
    in_dim = 4
    in_tensor = torch.ones((2, 3, in_dim))

    scaling = 2.0
    offset = 4.5
    encoder = encodings.ScalingAndOffset(in_dim=in_dim, scaling=scaling, offset=offset)

    assert encoder.get_out_dim() == in_dim
    encoded = encoder(in_tensor)
    assert encoded.shape[-1] == in_dim
    assert in_tensor * 6.5 == pytest.approx(encoded)

    with pytest.raises(ValueError):
        encodings.ScalingAndOffset(in_dim=-1)


def test_nerf_encoder():
    """Test NeRF encoder"""
    in_dim = 4
    out_dim = 24
    in_tensor = torch.ones((2, 3, in_dim))

    num_frequencies = 3
    min_freq_exp = 0
    max_freq_exp = 3
    encoder = encodings.NeRFEncoding(
        in_dim=in_dim, num_frequencies=num_frequencies, min_freq_exp=min_freq_exp, max_freq_exp=max_freq_exp
    )
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.ones((2, 3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape[-1] == out_dim
    assert torch.max(encoded) == 1

    in_tensor = torch.zeros((2, 3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape[-1] == out_dim
    assert torch.min(encoded) == 0

    # Test integrated posenc
    covs = torch.ones((2, 3, in_dim, in_dim))
    encoded = encoder(in_tensor, covs=covs)


def test_rff_encoder():
    """Test RFF encoder"""
    in_dim = 3
    out_dim = 24
    in_tensor = torch.ones((2, 3, in_dim))

    num_frequencies = 12
    scale = 5
    encoder = encodings.RFFEncoding(in_dim=in_dim, num_frequencies=num_frequencies, scale=scale)
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.ones((2, 3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape[-1] == out_dim

    # Test integrated encoding
    covs = torch.ones((2, 3, in_dim, in_dim))
    encoded = encoder(in_tensor, covs=covs)


def test_tensor_vm_encoder():
    """Test TensorVM encoder"""

    num_components = 24
    resolution = 32

    in_dim = 3
    out_dim = 3 * num_components

    encoder = encodings.TensorVMEncoding(num_components=num_components, resolution=resolution)
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.ones((3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape == (3, out_dim)

    in_tensor = torch.ones((6, 3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape == (6, 3, out_dim)

    encoder.upsample_grid(resolution=64)
    encoder.upsample_grid(resolution=128)


def test_tensor_cp_encoder():
    """Test TensorCP encoder"""

    num_components = 24
    resolution = 32

    in_dim = 3
    out_dim = num_components

    encoder = encodings.TensorCPEncoding(num_components=num_components, resolution=resolution)
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.ones((3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape == (3, out_dim)

    in_tensor = torch.ones((6, 3, in_dim))
    encoded = encoder(in_tensor)
    assert encoded.shape == (6, 3, out_dim)

    encoder.upsample_grid(resolution=64)


def test_tensor_sh_encoder():
    """Test Spherical Harmonic encoder"""

    levels = 5
    out_dim = levels**2

    with pytest.raises(ValueError):
        encoder = encodings.SHEncoding(levels=6)

    encoder = encodings.SHEncoding(levels=levels)
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.zeros((10, 3))
    in_tensor[..., 1] = 1
    encoded = encoder(in_tensor)
    assert encoded.shape == (10, out_dim)


def test_tensor_hash_encoder():
    """Test Spherical Harmonic encoder"""

    num_levels = 4
    features_per_level = 4
    out_dim = num_levels * features_per_level

    encoder = encodings.HashEncoding(
        num_levels=num_levels,
        features_per_level=features_per_level,
        log2_hashmap_size=5,
        implementation="torch",
    )
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.rand((10, 3))
    encoded_torch = encoder(in_tensor)
    assert encoded_torch.shape == (10, out_dim)

    encoder = encodings.HashEncoding(
        num_levels=num_levels,
        features_per_level=features_per_level,
        log2_hashmap_size=5,
        implementation="tcnn",
    )
    encoded_tcnn = encoder(in_tensor).float().cpu()
    assert encoded_tcnn.shape == (10, out_dim)


def test_kplane_encoder():
    """Test K-Planes encoder"""

    out_dim = 8

    encoder = encodings.KPlanesEncoding(
        resolution=(32, 32, 32, 16),
        num_components=out_dim,
        reduce="product",
    )
    assert encoder.get_out_dim() == out_dim

    in_tensor = torch.randn((10, 4))
    encoded = encoder(in_tensor)
    assert encoded.shape == (10, out_dim)

    encoder = encodings.KPlanesEncoding(
        resolution=(32, 32, 32),
        num_components=out_dim,
        reduce="product",
    )
    in_tensor = torch.randn((10, 3))
    encoded = encoder(in_tensor)
    assert encoded.shape == (10, out_dim)


if __name__ == "__main__":
    test_scaling_and_offset()
    test_nerf_encoder()
    test_rff_encoder()
    test_tensor_vm_encoder()
    test_tensor_cp_encoder()
    test_tensor_sh_encoder()
    test_tensor_hash_encoder()
