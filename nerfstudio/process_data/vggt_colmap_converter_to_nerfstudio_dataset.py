import sys
sys.path.append('./vggt')

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import BaseConverterToNerfstudioDataset
from nerfstudio.process_data.colmap_utils import colmap_to_json
from nerfstudio.utils.rich_utils import CONSOLE

# Adjusted imports for VGGT utilities
from vggt_to_colmap import load_model, process_images, extrinsic_to_colmap_format, filter_and_prepare_points, write_colmap_cameras_txt, write_colmap_images_txt, write_colmap_points3D_txt, write_colmap_cameras_bin, write_colmap_images_bin, write_colmap_points3D_bin

@dataclass
class VGGTColmapConverterToNerfstudioDataset(BaseConverterToNerfstudioDataset):
    """Class to process VGGT data into a Nerfstudio-compatible dataset."""

    data: Path
    """Path to the input data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    conf_threshold: float = 50.0
    """Confidence threshold for filtering points."""
    mask_sky: bool = False
    """Whether to mask sky regions."""
    mask_black_bg: bool = False
    """Whether to mask black background regions."""
    mask_white_bg: bool = False
    """Whether to mask white background regions."""
    stride: int = 1
    """Stride for point sampling."""

    @property
    def image_dir(self) -> Path:
        return self.data

    def _run_vggt_to_colmap(self):
        """Run VGGT to generate COLMAP-compatible data."""
        model, device = load_model()
        predictions, image_names = process_images(self.image_dir, model, device)

        quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])
        points3D, image_points2D = filter_and_prepare_points(
            predictions,
            self.conf_threshold,
            mask_sky=self.mask_sky,
            mask_black_bg=self.mask_black_bg,
            mask_white_bg=self.mask_white_bg,
            stride=self.stride,
        )

        return quaternions, translations, points3D, image_points2D, image_names, predictions

    def _save_transforms(self, num_frames: int) -> List[str]:
        """Save transforms.json after processing VGGT data."""
        summary_log = []
        quaternions, translations, points3D, image_points2D, image_names, predictions = self._run_vggt_to_colmap()

        with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
            # Save COLMAP-compatible files before calling colmap_to_json
            write_colmap_cameras_txt(
                self.output_dir / "cameras.txt",
                predictions["intrinsic"],
                predictions["images"].shape[2],
                predictions["images"].shape[1],
            )
            write_colmap_images_txt(
                self.output_dir / "images.txt",
                quaternions,
                translations,
                image_points2D,
                image_names,
            )
            write_colmap_points3D_txt(
                self.output_dir / "points3D.txt",
                points3D,
            )

            # Save binary COLMAP-compatible files
            write_colmap_cameras_bin(
                self.output_dir / "cameras.bin",
                predictions["intrinsic"],
                predictions["images"].shape[2],
                predictions["images"].shape[1],
            )
            write_colmap_images_bin(
                self.output_dir / "images.bin",
                quaternions,
                translations,
                image_points2D,
                image_names,
            )
            write_colmap_points3D_bin(
                self.output_dir / "points3D.bin",
                points3D,
            )

            num_matched_frames = colmap_to_json(
                recon_dir=self.output_dir,
                output_dir=self.output_dir,
                camera_mask_path=None,
                image_id_to_depth_path=None,
                image_rename_map=None,
                keep_original_world_coordinate=False,
                use_single_camera_mode=True,
            )
            summary_log.append(f"VGGT-Colmap matched {num_matched_frames} images")

        return summary_log

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.image_dir.exists():
            raise RuntimeError(f"Image directory {self.image_dir} does not exist.")

    def main(self) -> None:
        """Main method to process VGGT data into a Nerfstudio-compatible dataset."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy downscaled images from the input directory to the output directory
        downscaled_image_dir = self.image_dir / "downscaled"
        if not downscaled_image_dir.exists():
            raise FileNotFoundError(f"Expected downscaled directory at {downscaled_image_dir}, but it does not exist.")

        output_image_dir = self.output_dir / "images"
        output_image_dir.mkdir(parents=True, exist_ok=True)

        for image_file in self.image_dir.iterdir():
            if image_file.is_file():
                shutil.copy(image_file, output_image_dir)

        print(f"Copied downscaled images to {output_image_dir}")

        summary_log = self._save_transforms(num_frames=0)

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")