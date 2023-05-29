import copy
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from block_nerf.block_nerf import transform_camera_path


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_json(path: Union[Path, str]):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Union[Path, str], data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


class RenderSideBySide:
    def __init__(
        self,
        exp_dir: Path,
        cameras_in_rig: int,
        target_camera_path_path: Path,
        render_dir: Path,
        images_dir: Optional[Path],
        camera_offset: int = 0,
        fps: int = 24,
    ) -> None:
        self.exp_dir = exp_dir
        self.images_dir = images_dir if images_dir else exp_dir / "images"
        self.transforms_path = exp_dir / "transforms.json"
        self.new_images_dir = self.exp_dir / f"images_{get_timestamp()}"
        self.cameras_in_rig = cameras_in_rig
        self.camera_offset = camera_offset
        self.fps = fps

        self.new_transforms_path = self.transforms_path.parent / f"transforms_{get_timestamp()}.json"
        self.target_camera_path_path = target_camera_path_path
        self.render_dir = render_dir
        self.colmap_applied_transforms = self.get_colmap_applied_transforms()
    
    def get_colmap_applied_transforms(self) -> Optional[np.ndarray]:
        transforms = load_json(self.transforms_path)
        colmap_applied_transform = transforms.get("applied_transform", None)
        return np.array(colmap_applied_transform) if colmap_applied_transform else None

    def copy_every_n_images(self):
        count = 0
        self.new_images_dir.mkdir(exist_ok=True)
        # Copy every n image from images_path to self.new_images_dir and rename it to be in the format %04d.png, the copy needs to be sorted by name
        for image_path in sorted(self.images_dir.glob("*.png"))[self.camera_offset :: self.cameras_in_rig]:
            new_image_path = self.new_images_dir / f"{count:04d}.png"
            cmd = ["cp", str(image_path), str(new_image_path)]
            subprocess.run(cmd, check=True)
            count += 1

    def copy_every_n_transforms(self, transforms_path: Optional[Path] = None):
        transforms = load_json(transforms_path) if transforms_path else load_json(self.transforms_path)
        frames = transforms["frames"]
        new_frames = [frame for i, frame in enumerate(frames) if (i + self.camera_offset) % self.cameras_in_rig == 0]
        new_transforms = copy.deepcopy(transforms)
        new_transforms["frames"] = new_frames

        write_json(self.new_transforms_path, new_transforms)

    def create_camera_path_from_transforms(
        self, source_dataparser_transforms_path: Optional[Path] = None) -> Path:
        transforms = load_json(self.new_transforms_path)
        frames = transforms["frames"]
        sorted_frames = sorted(frames, key=lambda x: x["file_path"])
        flattened_frames = [np.array(frame["transform_matrix"]).flatten().tolist() for frame in sorted_frames]

        # If the intrinsics are per-frame
        width = 0
        height = 0
        if 'w' not in transforms:
            widths = [frame['w'] for frame in sorted_frames[:10]]
            index = widths.index(max(widths))
            width = sorted_frames[index]['w']
            height = sorted_frames[index]['h']
        else:
            width = transforms['w']
            height = transforms['h']

        render_scale = 1
        while width * render_scale < 1600:
            render_scale += 1
        
        new_camera_path = {
            "keyframes": [],
            "camera_type": "perspective",
            "render_height": height * render_scale,
            "render_width": height * render_scale,
            "camera_path": [],  # "camera_to_world"-dict with 16 double values in a 1D list.
            "fps": self.fps,
            "seconds": len(frames) / self.fps,
            "smoothness_value": 0.4,  # TOOD: Should I get this somewhere else / change it?
            "is_cycle": "true",
        }

        for c2w in flattened_frames:
            new_camera_path["camera_path"].append(
                {
                    "camera_to_world": c2w,
                    "fov": 90,
                    "aspect": width / height,
                }
            )

        write_json(self.target_camera_path_path, new_camera_path)

        # Scale and translate the camera path to the trained NeRF's coordinate system.
        export_path = self.target_camera_path_path.parent / self.target_camera_path_path.name.replace(
            ".json", "_scaled.json"
        )

        if source_dataparser_transforms_path is None:
            with open(export_path, "w") as f:
                json.dump(new_camera_path, f, indent=4)
        else:
            transform_camera_path(
                camera_path_path=self.target_camera_path_path,
                dataparser_transform_path=source_dataparser_transforms_path,
                export_path=export_path,
                colmap_transform=self.colmap_applied_transforms,
            )

        return export_path

    def create_video_from_images(self, export_dir: Path) -> Path:
        export_dir.mkdir(exist_ok=True)
        output_path = export_dir / f"output-{get_timestamp()}.mp4"
        cmd = f"ffmpeg -framerate {self.fps} -i {self.new_images_dir}/%04d.png -c:v libx264 -r {self.fps} -pix_fmt yuv420p {output_path}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        print("✅ Created video from images")
        return output_path

    def create_side_by_side_video(self, video_paths: List[Path], export_path: Path):
        # Rescale all input-videos to 1080p
        rescaled_paths = []
        for video_path in video_paths:
            rescaled_video_path = video_path.parent / f"rescaled-{video_path.name}"
            cmd = f'ffmpeg -n -i {video_path} -vf "scale=w=800:h=600:force_original_aspect_ratio=1,pad=800:600:(ow-iw)/2:(oh-ih)/2" -c:v libx264 {rescaled_video_path}'
            print(cmd)
            subprocess.run(cmd, shell=True, check=True)
            rescaled_paths.append(rescaled_video_path)
        print("✅ Rescaled videos to 800x600")

        # Create side-by-side videoP
        cmd = f"ffmpeg -n {' '.join([f'-i {video_path}' for video_path in rescaled_paths])} -filter_complex hstack=inputs={len(rescaled_paths)} {export_path}"
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ Created side-by-side video at {export_path}")

    # def main(self):
    #     exp_path = Path("../data/images/side_by_side_test/0")
    #     model_path = exp_path / "side_by_side_test-0/nerfacto/2023-04-22_160821"

    #     transforms_path = exp_path / "transforms.json"
    #     new_transforms_path = exp_path / "new_transforms.json"
    #     images_path = exp_path / "images"
    #     new_image_dir = exp_path / "new_images"
    #     camera_path_path = exp_path / f"camera_path-{get_timestamp()}.json"
    #     dataparser_transforms_path = model_path / "dataparser_transforms.json"
    #     render_dir = exp_path / "renders"

    #     # Right camera in a rig with 2 cameras
    #     n = self.cameras_in_rig
    #     offset = self.camera_offset
    #     fps = self.fps

    #     copy_every_n_transforms(transforms_path, new_transforms_path, n=n, offset=offset)
    #     copy_every_n_images(images_path, new_image_dir, n=n, offset=offset)
    #     create_video_from_images(new_image_dir, export_dir=render_dir, fps=fps)
    #     create_camera_path_from_transforms(
    #         transforms_path=new_transforms_path,
    #         camera_path_path=camera_path_path,
    #         dataparser_transforms_path=dataparser_transforms_path,
    #         fps=fps,
    #     )


# if __name__ == "__main__":
#     exp_dir = Path(".")  # TODO: Experiment directory
#     model_dir = Path(".")  # TODO: Model directory
#     side_by_side_export_path = exp_dir / "renders" / "side_by_side.mp4"

#     camera_path = RenderSideBySide(
#         cameras_in_rig=2,
#         camera_offset=0,
#         exp_dir=exp_dir,
#         fps=24,
#         images_dir=exp_dir / "images",
#         render_dir=exp_dir / "renders",
#         target_camera_path_path=exp_dir / "camera_path.json",
#     )

#     camera_path.copy_every_n_images()
#     camera_path.copy_every_n_transforms()
#     input_images_render = camera_path.create_video_from_images(export_dir=exp_dir / "renders")
#     input_images_camera_path = camera_path.create_camera_path_from_transforms(model_dir / "dataparser_transforms.json")

#     model_render_path = model.render() # TODO: Render model

#     camera_path.create_side_by_side_video(video_paths=[input_images_render, model_render_path], export_path=side_by_side_export_path)
