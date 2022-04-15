"""
Ray generator.
"""
from torch import nn
from torchtyping import TensorType
from mattport.structures.rays import RayBundle
from mattport.structures.cameras import get_camera_model


class RayGenerator(nn.Module):
    """torch.nn Module for generating rays.
    This class will store the intrinsics and extrinsics parameters of the cameras."""

    def __init__(
        self,
        intrinsics: TensorType["num_cameras", "num_intrinsics_params"],
        camera_to_world: TensorType["num_cameras", 3, 4],
    ) -> None:
        """_summary_

        Args:
            intrinsics (TensorType[&quot;num_cameras&quot;, &quot;num_intrinsics_params&quot;]):
                The intrinsics parameters.
            camera_to_world (TensorType[&quot;num_cameras&quot;, 3, 4]): Camera to world transformation matrix.
        """
        super().__init__()
        self.num_cameras, self.num_intrinsics_params = intrinsics.shape
        assert self.num_cameras >= 0
        self.intrinsics = nn.Parameter(intrinsics, requires_grad=False)
        self.camera_to_world = nn.Parameter(camera_to_world, requires_grad=False)
        # TODO(ethan): add learnable parameters that are deltas on the intrinsics and camera_to_world parameters

    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        """Index into the cameras to generate the rays.

        Args:
            x (_type_): _description_
        """
        c = ray_indices[:, 0]  # camera indices
        i = ray_indices[:, 1]  # row indices
        j = ray_indices[:, 2]  # col indices
        intrinsics = self.intrinsics[c]
        camera_to_world = self.camera_to_world[c]
        # NOTE(ethan): we currently assume all images have the same height and width
        camera_index = 0
        camera_class = get_camera_model(self.num_intrinsics_params)
        camera = camera_class(*self.intrinsics[camera_index].tolist())
        image_coords = camera.get_image_coords()
        coords = image_coords[i, j]

        ray_bundle = camera_class.generate_rays(intrinsics=intrinsics, camera_to_world=camera_to_world, coords=coords)
        ray_bundle.camera_indices = c
        return ray_bundle
