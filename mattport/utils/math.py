""" Math Helper Functions """

import torch
from torchtyping import TensorType


def components_from_spherical_harmonics(levels: int, directions: TensorType[..., 3]) -> TensorType[..., "components"]:
    """
    Args:
        levels (int): Number of spherical harmonic levels to compute.
        coeffs (TensorType[..., &quot;coeffs&quot;]): Spherical hamonic coefficients

    Returns:
        TensorType[..., "components"]: Value for each component.
    """
    num_components = levels**2
    components = torch.zeros((*directions.shape[:-1], num_components)).to(directions.device)

    assert levels > 0 and levels <= 5, f"SH levels must be in [1,4], got {levels}"
    assert directions.shape[-1] == 3, f"Direction input should have three dimensions. Got {directions.shape[-1]}"

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # l0
    components[..., 0] = 0.28209479177387814

    # l1
    if levels > 1:
        components[..., 1] = 0.4886025119029199 * y
        components[..., 2] = 0.4886025119029199 * z
        components[..., 3] = 0.4886025119029199 * x

    # l2
    if levels > 2:
        components[..., 4] = 1.0925484305920792 * x * y
        components[..., 5] = 1.0925484305920792 * y * z
        components[..., 6] = 0.9461746957575601 * zz
        components[..., 7] = 1.0925484305920792 * x * z
        components[..., 8] = 0.5462742152960396 * (xx - yy)

    # l3
    if levels > 3:
        components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        components[..., 10] = 2.890611442640554 * x * y * z
        components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        components[..., 14] = 1.445305721320277 * z * (xx - yy)
        components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l4
    if levels > 4:
        components[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        components[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        components[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        components[..., 19] = 0.6690465435572892 * y * (7 * zz - 3 * z)
        components[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        components[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        components[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        components[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        components[..., 24] = 0.4425326924449826 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return components
