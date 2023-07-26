# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from typing_extensions import Annotated

from nerfstudio.utils.metrics import LPIPSModule, PSNRModule, SSIMModule


@dataclass
class BaseMetrics:
    """Base class for metrics."""

    input_folder: Path = Path("input-folder")
    """Folder containing the renders."""
    output_folder: Path = Path("output-folder")
    """Folder to save the metrics."""
    device: str = "cuda:0"
    """Device to use for metrics."""

    def main(self) -> None:
        """Main function."""

        # image metrics
        psnr_module = PSNRModule().to(self.device)
        ssim_module = SSIMModule().to(self.device)
        lpips_module = LPIPSModule().to(self.device)

        rgb_gt_filenames = sorted(glob.glob(str(self.input_folder / "rgb_gt" / "*")))
        metrics = defaultdict()
        video = []

        experiment_name = os.path.basename(self.input_folder)
        print(f"Processing experiment: {experiment_name} ...")
        if len(rgb_gt_filenames) == 0:
            print("No rgb_gt images found, skipping experiment")
            sys.exit(0)

        for idx, rgb_gt_filename in enumerate(rgb_gt_filenames):
            rgb = media.read_image(rgb_gt_filename.replace("rgb_gt", "rgb"))
            rgb_gt = media.read_image(rgb_gt_filename)

            # move images to torch and to the correct device
            rgb = torch.from_numpy(rgb).float().to(self.device) / 255.0
            rgb_gt = torch.from_numpy(rgb_gt).float().to(self.device) / 255.0  # (H, W, 3)

            # compute the image metrics
            # reshape the images to (1, C, H, W)
            x = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            x_gt = rgb_gt.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

            psnr = float(psnr_module(x, x_gt)[0])
            ssim = float(ssim_module(x, x_gt)[0])
            lpips = float(lpips_module(x, x_gt)[0])

            metrics["psnr_list"].append(psnr)
            metrics["ssim_list"].append(ssim)
            metrics["lpips_list"].append(lpips)

            # save the images
            rgb_gt = (rgb_gt * 255.0).cpu().numpy().astype(np.uint8)
            rgb = (rgb * 255.0).cpu().numpy().astype(np.uint8)
            image = np.concatenate([rgb_gt, rgb], axis=1)
            image_filename = self.input_folder / "composited" / f"{idx:04d}.png"
            image_filename.parent.mkdir(parents=True, exist_ok=True)
            media.write_image(image_filename, image)
            video.append(image)

        # write out the video
        video_filename = self.input_folder / f"{experiment_name}.mp4"
        media.write_video(video_filename, video, fps=30)

        # convert metrics dict to a proper dictionary
        metrics = dict(metrics)
        metrics["psnr"] = np.mean(metrics["psnr_list"])
        metrics["ssim"] = np.mean(metrics["ssim_list"])
        metrics["lpips"] = np.mean(metrics["lpips_list"])

        for metric_name in sorted(metrics.keys()):
            if "_list" not in metric_name:
                print(f"{metric_name}: {metrics[metric_name]}")

        # write to a json file
        metrics_filename = self.output_folder / f"{experiment_name}.json"
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)


@dataclass
class NerfbusterMetrics(BaseMetrics):
    """Compute metrics for nerfbusters on the renders."""

    pseudo_gt_experiment_name: Path = Path("pseudo_gt_experiment_name")
    """Name of the experiment to use for pseudo ground truth."""
    max_depth: float = 2
    """Maximum depth to use for metrics."""
    min_views: int = 1
    """Minimum number of views to use for metrics."""

    def main(self) -> None:
        """Main function."""

        print("Using visibility masks from experiment: ", self.pseudo_gt_experiment_name)

        # image metrics
        psnr_module = PSNRModule().to(self.device)
        ssim_module = SSIMModule().to(self.device)
        lpips_module = LPIPSModule().to(self.device)

        rgb_gt_filenames = sorted(glob.glob(str(self.input_folder / "rgb_gt" / "*")))
        visibility_filenames = sorted(
            glob.glob(str(self.input_folder / self.pseudo_gt_experiment_name / "visibility_median_count" / "*"))
        )
        metrics = defaultdict()
        video = []  # images to make a video

        experiment_name = os.path.basename(self.input_folder)
        print(f"Processing experiment: {experiment_name} ...")
        if len(rgb_gt_filenames) == 0:
            print("No rgb_gt images found, skipping experiment")
            sys.exit(0)

        for idx, rgb_gt_filename in enumerate(rgb_gt_filenames):
            # read in the images
            depth = media.read_image(rgb_gt_filename.replace("rgb_gt", "depth"))
            depth_raw = np.load(rgb_gt_filename.replace("rgb_gt", "depth_raw").replace(".png", ".npy"))[..., 0]
            normals = media.read_image(rgb_gt_filename.replace("rgb_gt", "normals"))
            pseudo_gt_visibility = media.read_image(visibility_filenames[idx])
            psuedo_gt_depth_raw = np.load(
                visibility_filenames[idx].replace("visibility_median_count", "depth_raw").replace(".png", ".npy")
            )[..., 0]
            psuedo_gt_normals = media.read_image(
                visibility_filenames[idx].replace("visibility_median_count", "normals")
            )
            rgb = media.read_image(rgb_gt_filename.replace("rgb_gt", "rgb"))
            rgb_gt = media.read_image(rgb_gt_filename)

            # move images to torch and to the correct device
            depth = torch.from_numpy(depth).float().to(self.device) / 255.0  # 'depth' is a colormap
            depth_raw = torch.from_numpy(depth_raw).float().to(self.device)
            normals = torch.from_numpy(normals).float().to(self.device) / 255.0
            normals_raw = normals * 2.0 - 1.0
            pseudo_gt_visibility = torch.from_numpy(pseudo_gt_visibility).long().to(self.device)
            psuedo_gt_depth_raw = torch.from_numpy(psuedo_gt_depth_raw).float().to(self.device)
            psuedo_gt_normals = torch.from_numpy(psuedo_gt_normals).float().to(self.device) / 255.0
            psuedo_gt_normals_raw = psuedo_gt_normals * 2.0 - 1.0
            rgb = torch.from_numpy(rgb).float().to(self.device) / 255.0
            rgb_gt = torch.from_numpy(rgb_gt).float().to(self.device) / 255.0  # (H, W, 3)

            # create masks
            visibilty_mask = (pseudo_gt_visibility[..., 0] >= self.min_views).float()
            depth_mask = (depth_raw < self.max_depth).float()
            mask = visibilty_mask * depth_mask  # (H, W)
            mask = mask[..., None].repeat(1, 1, 3)

            # show masked versions
            # shape is (H, W, 3)
            rgb_gt_masked = rgb_gt * mask
            rgb_masked = rgb * mask
            depth_masked = depth * mask
            normals_masked = normals * mask

            # compute the image metrics
            # reshape the images to (1, C, H, W)
            x = rgb_masked.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            x_gt = rgb_gt_masked.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            m = mask.permute(2, 0, 1).unsqueeze(0)[:, 0:1]  # (1, 1, H, W)
            psnr = float(psnr_module(x, x_gt, m)[0])
            ssim = float(ssim_module(x, x_gt, m)[0])
            lpips = float(lpips_module(x, x_gt, m)[0])

            # depth
            depth_mask = mask[..., 0] == 1
            depth_mse = float(F.mse_loss(depth_raw[depth_mask], psuedo_gt_depth_raw[depth_mask]))

            # disparity
            disparity_raw = 1.0 / depth_raw
            psuedo_gt_disparity_raw = 1.0 / psuedo_gt_depth_raw
            disparity = float(torch.abs(disparity_raw[depth_mask] - psuedo_gt_disparity_raw[depth_mask]).mean())

            # make sure the normals raw are normalized
            normals_raw = normals_raw / torch.norm(normals_raw, dim=-1, keepdim=True)
            psuedo_gt_normals_raw = psuedo_gt_normals_raw / torch.norm(psuedo_gt_normals_raw, dim=-1, keepdim=True)
            eps = 1e-8
            costheta = (
                (normals_raw[mask == 1].view(-1, 3) * psuedo_gt_normals_raw[mask == 1].view(-1, 3))
                .sum(dim=-1)
                .clamp(-1 + eps, 1 - eps)
            )
            theta = torch.abs(torch.acos(costheta) * 180.0 / np.pi)
            normals_mse = float(theta.mean())
            normals_median = float(theta.median())

            # the angle thresholds
            metrics["normals_11.25"].append(float((theta < 11.25).float().mean()))
            metrics["normals_22.5"].append(float((theta < 22.5).float().mean()))
            metrics["normals_30"].append(float((theta < 30).float().mean()))

            # coverage
            coverage = float(mask[..., 0].sum() / visibilty_mask.sum())

            metrics["psnr_list"].append(psnr)
            metrics["ssim_list"].append(ssim)
            metrics["lpips_list"].append(lpips)
            metrics["depth_list"].append(depth_mse)
            metrics["disparity_list"].append(disparity)
            metrics["normals_list"].append(normals_mse)
            metrics["normals_median_list"].append(normals_median)
            metrics["coverage_list"].append(coverage)

            # save the images
            rgb_gt = (rgb_gt * 255.0).cpu().numpy().astype(np.uint8)
            pseudo_gt_visibility = (pseudo_gt_visibility).cpu().numpy().astype(np.uint8)
            rgb_gt_masked = (rgb_gt_masked * 255.0).cpu().numpy().astype(np.uint8)
            rgb_masked = (rgb_masked * 255.0).cpu().numpy().astype(np.uint8)
            depth_masked = (depth_masked * 255.0).cpu().numpy().astype(np.uint8)
            normals_masked = (normals_masked * 255.0).cpu().numpy().astype(np.uint8)
            image = np.concatenate(
                [rgb_gt, pseudo_gt_visibility, rgb_gt_masked, rgb_masked, depth_masked, normals_masked], axis=1
            )
            image_filename = self.input_folder / "composited" / f"{idx:04d}.png"
            image_filename.parent.mkdir(parents=True, exist_ok=True)
            media.write_image(image_filename, image)
            video.append(image)

        # write out the video
        video_filename = self.input_folder / f"{experiment_name}.mp4"
        media.write_video(video_filename, video, fps=30)

        # convert metrics dict to a proper dictionary
        metrics = dict(metrics)
        metrics["psnr"] = np.mean(metrics["psnr_list"])
        metrics["ssim"] = np.mean(metrics["ssim_list"])
        metrics["lpips"] = np.mean(metrics["lpips_list"])
        metrics["depth"] = np.mean(metrics["depth_list"])
        metrics["disparity"] = np.mean(metrics["disparity_list"])
        metrics["normals"] = np.mean(metrics["normals_list"])
        metrics["normals_median"] = np.mean(metrics["normals_median_list"])
        metrics["coverage"] = np.mean(metrics["coverage_list"])
        metrics["normals_11.25"] = np.mean(metrics["normals_11.25"])
        metrics["normals_22.5"] = np.mean(metrics["normals_22.5"])
        metrics["normals_30"] = np.mean(metrics["normals_30"])

        for metric_name in sorted(metrics.keys()):
            if "_list" not in metric_name:
                print(f"{metric_name}: {metrics[metric_name]}")

        # write to a json file
        metrics_filename = self.output_folder / f"{experiment_name}.json"
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[NerfbusterMetrics, tyro.conf.subcommand(name="nerfbusters-metrics")],
        Annotated[BaseMetrics, tyro.conf.subcommand(name="base-metrics")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
