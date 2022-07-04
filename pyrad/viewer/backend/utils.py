import numpy as np
import torch


def get_chunks(lst, num_chunks=None, size_of_chunk=None):
    """Returns list of n elements, constaining a sublist."""
    if num_chunks:
        assert not size_of_chunk
        size = len(lst) // num_chunks
    if size_of_chunk:
        assert not num_chunks
        size = size_of_chunk
    chunks = []
    for i in range(0, len(lst), size):
        chunks.append(lst[i : i + size])
    return chunks


def get_intrinsics_matrix_and_camera_to_world_h(camera_object, image_height):
    """Returns the camera intrinsics matrix and the camera to world homogeneous matrix.
    Args:
        camera_object: a Camera object.
        image_size: the size of the image (height, width)
    """
    # intrinsics
    fov = camera_object["fov"]
    aspect = camera_object["aspect"]
    image_width = aspect * image_height
    pp_w = image_width / 2.0
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    intrinsics_matrix = torch.tensor([[focal_length, 0, pp_w], [0, focal_length, pp_h], [0, 0, 1]]).float()

    # extrinsics
    camera_to_world_h = torch.tensor(get_chunks(camera_object["matrix"], size_of_chunk=4)).T.float()

    return intrinsics_matrix, camera_to_world_h
