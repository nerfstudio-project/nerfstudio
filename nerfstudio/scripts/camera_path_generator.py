import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup


def calculate_camera_to_world_matrix(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    forward = (eye - target) / torch.norm(target - eye)
    right = torch.cross((up - eye), forward)
    right /= torch.norm(right)
    up = torch.cross(forward, right)
    # Create rotation matrices
    rotation_matrices = torch.stack([right, up, forward], dim=-1)

    # Combine rotation and translation matrices
    camera_to_world_matrices = torch.eye(4)  # Initialize with identity matrices
    camera_to_world_matrices[:3, :3] = rotation_matrices
    camera_to_world_matrices[:3, 3] = eye
    return camera_to_world_matrices[:3, :].unsqueeze(0)


# Function to create a rotation matrix around the y-axis
def rotation_matrix_y(angles: torch.Tensor) -> torch.Tensor:
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    rotation_matrices = torch.zeros((angles.shape[0], 3, 3), device=angles.device)
    rotation_matrices[:, 0, 0] = cos_angles
    rotation_matrices[:, 0, 2] = sin_angles
    rotation_matrices[:, 1, 1] = 1
    rotation_matrices[:, 2, 0] = -sin_angles
    rotation_matrices[:, 2, 2] = cos_angles
    return rotation_matrices


def rotate_camera_to_world_matrices(
    camera_to_worlds: torch.Tensor, start_angle: float, end_angle: float, intermediate_steps: int, camera: Cameras
) -> Cameras:
    device = camera.device
    # Generate views by rotating around the y-axis in the world coordinate system and transform these rotated targets into the NeRF coordinate system afterwards
    start_angle = np.deg2rad(start_angle)
    end_angle = np.deg2rad(end_angle)
    angles = torch.linspace(start_angle, end_angle, intermediate_steps, device=device)
    # Create rotation matrices for all angles in parallel
    rotation_matrices = rotation_matrix_y(angles)
    # Transpose the rotation matrices to get the clockwise rotation
    rotation_matrices = rotation_matrices.transpose(1, 2)
    # Number of cameras and rotations
    num_cameras = camera_to_worlds.shape[0]
    num_rotations = rotation_matrices.shape[0]

    # Repeat the camera-to-world matrices for each rotation
    # Shape: (num_cameras, num_rotations, 3, 4)
    expanded_camera_matrices = camera_to_worlds.unsqueeze(1).repeat(1, num_rotations, 1, 1)
    # Repeat the rotation matrices for each camera
    # Shape: (num_cameras, num_rotations, 3, 3)
    expanded_rotation_matrices = rotation_matrices.unsqueeze(0).repeat(num_cameras, 1, 1, 1)

    # Apply the rotation matrices to the rotation part of the camera-to-world matrices
    # Shape: (num_cameras, num_rotations, 3, 3)
    rotated_matrices = torch.matmul(expanded_camera_matrices[..., :3], expanded_rotation_matrices)
    # Combine the rotated parts with the original translation parts
    # Shape: (num_cameras, num_rotations, 3, 4)
    result_matrices = torch.cat((rotated_matrices, expanded_camera_matrices[..., 3:]), dim=-1)

    # Reshape to the desired shape (num_cameras*num_rotations, 3, 4)
    result_matrices = result_matrices.view(-1, 3, 4)
    cameras = Cameras(
        camera_to_worlds=result_matrices,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
        width=camera.width,
        height=camera.height,
        distortion_params=camera.distortion_params,
        camera_type=camera.camera_type,
        times=camera.times,
        metadata=camera.metadata,
    )
    return cameras


def get_pose_view_camera_matrix(
    transformation_matrix: list, scale_factor: float, eye: list, target: list, device: torch.device
) -> torch.Tensor:
    transformation_matrix = torch.tensor(transformation_matrix, device=device)
    transformation_matrix = torch.cat(
        (transformation_matrix, torch.tensor([[0, 0, 0, 1]], device=device)), dim=0
    )  # Homogeneous coordinate
    # Apply scale to transformation matrix
    scaled_transformation_matrix = transformation_matrix * scale_factor

    eye = torch.tensor(eye, device=device)
    target = torch.tensor(target, device=device)

    # Transform the eye into NeRF coordinate system
    eye_nerf = torch.cat((eye, torch.tensor([1.0], device=device)))  # Homogeneous coordinate
    eye_nerf = scaled_transformation_matrix @ eye_nerf
    eye_nerf = eye_nerf[:-1]  # Remove homogeneous component

    target_nerf = torch.cat((target, torch.tensor([1.0], device=device)))  # Homogeneous coordinate
    target_nerf = scaled_transformation_matrix @ target_nerf
    target_nerf = target_nerf[:-1]  # Remove homogeneous component

    up = torch.tensor(
        [eye[0], -1, eye[2]], device=device
    )  # [0, -1, 0] in world coordinate system becomes the view towards the ceiling in the camera coordinate system
    up = torch.cat((up, torch.tensor([1.0], device=device)))
    up_nerf = scaled_transformation_matrix @ up
    up_nerf = up_nerf[:-1]  # Remove homogeneous component
    camera_to_world_matrix = calculate_camera_to_world_matrix(eye_nerf, target_nerf, up_nerf)
    return camera_to_world_matrix


@dataclass
class GeneratePoseView:
    """Generate a camera path by rotating around a pose and output it to a JSON file."""

    load_config: Path
    """Path to config YAML file."""
    output_path: Path = Path("camera_path.json")
    """Path to output camera path JSON file."""
    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    viewpoints: Optional[int] = None
    """Number of viewpoints to sample from the camera path"""
    eye: Optional[str] = None
    """Eye of the pose to render of shape "x y z" in the original data coordinate system."""
    target: Optional[str] = None
    """Target of the pose to render of shape "x y z" in the original data coordinate system. If None, they eye targets the origin in parallel to the floor."""
    start_angle: float = 0.0
    """Start angle of the perspective relative to the origin."""
    end_angle: float = 345.0
    """End angle of the perspective relative to the origin."""
    intermediate_steps: int = 20
    """Number of intermediate steps of the arc that will be rendered."""
    load_dataparser_transforms: Optional[Path] = None
    """Path to dataparser_transforms JSON file."""

    def main(self) -> None:
        """Main function."""
        self._setup_pipeline()

        if self.eye:
            assert (
                self.viewpoints is None
            ), "Either use eye and target point or viewpoints sampled uniformly from the camera path to render views."
            pose_view_cameras = self._render_eye_view()
        elif self.viewpoints:
            assert (
                self.eye is None
            ), "Either use eye and target point or viewpoints sampled uniformly from the camera path to render views."
            assert (
                self.target is None
            ), "Either use eye and target point or viewpoints sampled uniformly from the camera path to render views."
            assert (
                self.load_dataparser_transforms is None
            ), "Either use eye and target point or viewpoints sampled uniformly from the camera path to render views."
            pose_view_cameras = self._render_viewpoints()
        else:
            raise ValueError("Either eye and target or viewpoints must be specified.")
        assert self.start_angle <= self.end_angle, "The start angle must be less than or equal to the end angle."
        assert self.start_angle > -360, "The start angle must not exceed -360 degrees."
        assert self.end_angle < 360, "The end angle must not exceed 360 degrees."

        camera_path = self._generate_camera_path(pose_view_cameras)
        self._save_camera_path(camera_path)

    def _setup_pipeline(self) -> None:
        """Sets up the pipeline and necessary components."""
        _, self.pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="test",
        )

        if self.pose_source == "eval":
            assert self.pipeline.datamanager.eval_dataset is not None
            self.cameras = self.pipeline.datamanager.eval_dataset.cameras
        else:
            assert self.pipeline.datamanager.train_dataset is not None
            self.cameras = self.pipeline.datamanager.train_dataset.cameras

        self.camera = self.cameras[0]  # Use first camera object to get intrinsics and distortion params

    def _render_eye_view(self) -> None:
        """Renders views by rotating around a specified eye and target point."""
        assert self.load_dataparser_transforms, "dataparser_transforms.json path must be provided."
        assert (
            self.load_dataparser_transforms.is_file()
        ), f"dataparser_transforms.json could not be found in {self.load_dataparser_transforms}."

        with open(self.load_dataparser_transforms) as f:
            transforms = json.load(f)
            assert (
                "transform" in transforms
            ), f"Transformation matrix could not be found in {self.load_dataparser_transforms}."
            assert "scale" in transforms, f"Scale factor could not be found in {self.load_dataparser_transforms}."
            transform = transforms["transform"]
            scale = transforms["scale"]
            assert len(transform) == 3 and len(transform[0]) == 4, "Transformation matrix must be of shape [3, 4]."
            assert isinstance(scale, float), "Scale factor must be a scalar."

        eye = [float(x) for x in self.eye.split()]
        target = [float(x) for x in self.target.split()] if self.target else [0, eye[1], 0]
        self._validate_eye_target(eye, target)
        pose_view_camera_matrix = get_pose_view_camera_matrix(transform, scale, eye, target, self.camera.device)
        pose_view_cameras = rotate_camera_to_world_matrices(
            pose_view_camera_matrix, self.start_angle, self.end_angle, self.intermediate_steps, self.camera
        )
        return pose_view_cameras

    def _render_viewpoints(self) -> None:
        """Renders views by rotating around sampled viewpoints."""
        step_size = int(np.ceil(len(self.cameras) / self.viewpoints))
        camera_indices = torch.arange(0, len(self.cameras), step_size, device=self.cameras.device)
        pose_view_cameras = self.cameras[camera_indices]
        pose_view_cameras = rotate_camera_to_world_matrices(
            pose_view_cameras.camera_to_worlds, self.start_angle, self.end_angle, self.intermediate_steps, self.camera
        )
        return pose_view_cameras

    def _validate_eye_target(self, eye, target) -> None:
        """Validates the eye and target points."""
        assert len(eye) == 3, "Eye must be of shape 'x y z' in the original data coordinate system."
        assert len(target) == 3, "Target must be of shape 'x y z' in the original data coordinate system."

    def _generate_camera_path(self, cameras: Cameras) -> dict:
        """Generates a camera path from the given cameras."""
        camera_path = {
            "camera_type": cameras.camera_type[0].value,
            "cameras": [],
            "seconds": self.intermediate_steps / 24.0,  # Assuming 24 FPS
        }

        for idx, camera_to_world in enumerate(cameras.camera_to_worlds):
            camera_path["cameras"].append(
                {
                    "camera_to_worlds": camera_to_world.cpu().numpy().tolist(),
                    "fx": cameras.fx[idx].item(),
                    "fy": cameras.fy[idx].item(),
                    "cx": cameras.cx[idx].item(),
                    "cy": cameras.cy[idx].item(),
                    "width": cameras.width[idx].item(),
                    "height": cameras.height[idx].item(),
                    "distortion_params": cameras.distortion_params[idx].cpu().numpy().tolist(),
                    "metadata": cameras.metadata[idx],
                }
            )

        return camera_path

    def _save_camera_path(self, camera_path: dict) -> None:
        """Saves the generated camera path to a JSON file."""
        with open(self.output_path, "w") as f:
            json.dump(camera_path, f, indent=4)
        print(f"Camera path saved to {self.output_path}")


Commands = tyro.conf.FlagConversionOff[Literal["pose-view"]]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(GeneratePoseView).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
