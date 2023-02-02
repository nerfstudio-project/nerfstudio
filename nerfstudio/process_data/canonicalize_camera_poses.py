import matplotlib.pyplot as plt
import numpy as np


def canonicalize_camera_poses(frames):
    """Scale rotate and translate camera poses so that the "bottom" ring is centered at the origin
    and oriented in the xy plane."""
    # This utilizes the fact that images were taken on a turntable 15 degrees in order.
    # It uses ICP ("ground truth" is a circle equally spaced every 15 degrees)
    frames = sorted(frames, key=lambda frame: frame['file_path'])

    Ts = np.array([frame['transform_matrix'] for frame in frames])
    ts = Ts[:, :3, 3]

    # Find duplicate images
    dists = np.sqrt(np.sum(np.square(ts[:, :, None] - ts.T[None, :, :]), axis=1))
    dists /= np.max(dists)  # noralize
    np.fill_diagonal(dists, 1)  # Ignore diagonal
    threshold = np.pi * (15 / 360) / 2  # approximate distance to nearest turntable image
    duplicates = np.argwhere(dists < threshold)

    # Indices for non-duplicate images
    to_take = list(range(len(frames)))
    for _, dup in filter(lambda dup: dup[0] < dup[1], duplicates):
        to_take.remove(dup)

    # The images come in 2 rings.  Find which is the bigger ring
    if dists[to_take[0], to_take[11]] > dists[to_take[24], to_take[35]]:
        big_circle = to_take[:24]
    else:
        big_circle = to_take[24:]

    # The turntable turned CCW, so the camera moves in a CW circle
    theta = -(np.arange(0, 360, 15) * np.pi / 180).reshape(-1, 1)
    expected = np.hstack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))

    # Now run ICP
    _, sR, t = ICP_transform_with_scale(ts[big_circle], expected)

    # Correct poses and export
    T = np.zeros((4, 4))
    T[:3, :3] = sR
    T[:3, 3] = t
    T[3, 3] = 1

    for frame in frames:
        frame['transform_matrix'] = (T @ np.array(frame['transform_matrix'])).tolist()

    return frames


def center_roi(json_object):
    """Transforms the frames in the object so that the subject is centered and scaled to unit box"""

    Ts = np.array([frame['transform_matrix'] for frame in json_object['frames']])
    ray_dirs, ts = Ts[:, :3, 2], Ts[:, :3, 3]
    fx, fy, cx, cy = json_object['fl_x'], json_object['fl_y'], json_object['cx'], json_object['cy']

    # Find the point where all the cameras are pointing at.
    # Assume that the point lies on the z-axis.
    # Then we want to find the closest point on the ray to the z-axis.
    x_xy = -ts[:, :2]
    ray_xy = ray_dirs[:, :2]
    v_xy = np.sum(x_xy * ray_xy, axis=1)[:, None] * ray_xy / np.sum(np.square(ray_xy), axis=1)[:,
                                                                                               None]
    ratios = v_xy / ray_xy
    np.testing.assert_allclose(*ratios.T, err_msg='v_xy is not a projection of v onto xy')
    dz_height = ray_dirs[:, 2] * ratios[:, 0]
    # v_xy = np.hstack((v_xy, dz_height[:, None])) # this is just for debugging and we don't actually need it
    z_height = dz_height + ts[:, 2]
    z_height = np.mean(z_height)

    ts[:, 2] -= z_height

    # Now we want to scale the image so that the subject is in the box [-1, 1] centered at origin
    radius = 1
    max_x, max_y = radius * cx / fx, radius * cy / fy
    scale_factor = 1 / max(max_x, max_y)
    ts *= scale_factor

    for frame, t in zip(json_object['frames'], ts):
        T = np.array(frame['transform_matrix'])
        T[:3, 3] = t
        frame['transform_matrix'] = T.tolist()

    return json_object


# Copied from https://github.gatech.edu/borglab/hydroponics/blob/4b0cd43e20a0adcdec72fabb70705c1dfd63958c/analysis/align_camera_poses.py#L72
def ICP_transform_with_scale(xyz_colmap, xyz_exp, save_folder=None):
    '''
    Compute the optimal transformation sR, t such that xyz_exp = sR * xyz_colmap + t + error

    Arguments:
        xyz_colmap: Nx3 array of unscaled camera positions (in need of scale disambiguation)
        xyz_exp: Nx3 array of metrically correct camera positions
        save_folder: if provided, save the transformation to a file
    Returns: transformed_xyz_colmap, sR, t
        where transformed_xyz_colmap = sR * xyz_exp + t =~= xyz_exp

    From icp literature of transforming (pre-associated) pointclouds to match eachother,
      it can be proven that finding the optimal transformation: s,R,t of:
          min (xyz_exp - (s.R.xyz_colmap + t))
      can be done by first optimizing t, then R, then s.
      The intuition is that, in estimating R, we used SVD and ignore the Diagonal matrix, and s
        would just scale the diagonal matrix.
    See also:
        http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2005/Zinsser05-PSR.pdf
        https://johnwlambert.github.io/icp/
    '''
    # zero-out centroids (translation, but in the wrong frame)
    xyz2_colmap = xyz_colmap - np.mean(xyz_colmap, axis=0)
    xyz2_commanded = xyz_exp - np.mean(xyz_exp, axis=0)
    # Align R (rotation)
    H = np.einsum('tj,tk->jk', xyz2_commanded, xyz2_colmap)
    U, S, VT = np.linalg.svd(H)
    R = U @ VT
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ VT
    # Align s (scale)
    xyz3_colmap = (R @ xyz2_colmap.T).T
    s = np.sum(xyz2_commanded * xyz3_colmap) / np.sum(np.square(xyz3_colmap))
    # compute final t (translation in commanded frame, after rotation/scale)
    t = np.mean(xyz_exp, axis=0) - s * R @ np.mean(xyz_colmap, axis=0)

    # save and return
    if save_folder is not None:
        np.savez(f'{save_folder}/transformation_to_correct_dimensions.npz', R=s * R, t=t)
    return s * (R @ xyz_colmap.T).T + t, s * R, t
