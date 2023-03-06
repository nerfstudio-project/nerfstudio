"""
Test misc data utils
"""

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat


def test_scalar_first_scalar_last_quaternions():
    """Many nerfstudio datasets use quaternions for pose rotations.  Some use
    scalar-first format and others use scalar-last.  However internally
    nerfstudio uses rotation *matrices* / projection matrices.  This test
    helps document differences in quaternion encodings & conversion to
    rotation matrices.

    FMI see call sites in nerfstudio where the various libraries are used. You
    might also consult this detailed reference:
     * "Why and How to Avoid the Flipped Quaternion Multiplication"
        Sommer et al.
        https://arxiv.org/abs/1801.07478
    """

    # Pick an arbitrary euler rotation; note that different communities
    # also use different rotation axes (e.g. Euler vs Tait-Bryan ...)
    rot = Rotation.from_euler("xyz", np.array([10.0, 20.0, 30.0]), degrees=True)

    # scalar-last
    xyzw = rot.as_quat()
    assert np.allclose(xyzw, np.array([0.03813458, 0.18930786, 0.23929834, 0.95154852]))

    # scalar-first
    wxyz = xyzw[[3, 0, 1, 2]]
    assert np.allclose(wxyz, np.array([0.95154852, 0.03813458, 0.18930786, 0.23929834]))

    # Expected Rotation matrix
    # fmt: off
    R_expected = np.array( # pylint: disable=invalid-name
        [
            [ 0.81379768, -0.44096961,  0.37852231],
            [ 0.46984631,  0.88256412,  0.01802831],
            [-0.34202014,  0.16317591,  0.92541658]
        ]
    )
    # fmt: on

    # Record3D / scipy
    R = Rotation.from_quat(xyzw).as_matrix()
    assert np.allclose(R, R_expected)

    # Nuscenes / pyquaternion
    R = Quaternion(wxyz).rotation_matrix
    assert np.allclose(R, R_expected)

    # COLMAP
    # TODO(1480) use pycolmap
    # R = pycolmap.qvec_to_rotmat(wxyz)
    R = qvec2rotmat(wxyz)
    assert np.allclose(R, R_expected)
