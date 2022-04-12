"""
Test ray generation.
"""
import torch

from mattport.nerf.dataset.blender_dataset import load_blender_data
from mattport.nerf.modules.ray_generator import RayGenerator
from mattport.structures.rays import RayBundle
from mattport.utils.io import get_absolute_path


def test_ray_generator():
    """Test the ray generation works given camera, row, and col indices as input."""

    # load data
    datadir = get_absolute_path("data/lego")
    image_filenames, poses, focal_length, image_height, image_width = load_blender_data(
        datadir, half_res=False, testskip=1
    )
    cx = image_width / 2.0
    cy = image_height / 2.0
    camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform
    num_cameras = len(image_filenames)
    num_intrinsics_params = 3
    intrinsics = torch.ones((num_cameras, num_intrinsics_params), dtype=torch.float32)
    intrinsics *= torch.tensor([focal_length, cx, cy])

    # Initialize the ray generator with the cameras.
    ray_generator = RayGenerator(intrinsics, camera_to_world)

    # Random ray indices.
    print("Random ray indexing.")
    num_rays = 1024
    ray_indices = torch.floor(torch.rand((num_rays, 3)) * torch.tensor([num_cameras, image_height, image_width])).long()
    ray_bundle = ray_generator(ray_indices)
    assert isinstance(ray_bundle, RayBundle)

    # TODO(ethan): visualize the RayBundle object
    # TODO(ethan): add other test cases? e.g., origin and camera_indices checks

    # check that the origins are correct
    c = ray_indices[:, 0]
    assert torch.allclose(ray_bundle.origins, camera_to_world[c, :3, 3])

    # Ray indices for one camera.
    coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
    coords = torch.stack(coords, dim=-1)
    coords = coords.view(-1, 2)
    num_rays = coords.shape[0]
    camera_index = 0
    ray_indices = torch.cat([torch.ones_like(coords[:, :1]) * camera_index, coords], dim=-1).long()
    ray_bundle = ray_generator(ray_indices)

    print(ray_bundle.camera_indices)

    # assert False
