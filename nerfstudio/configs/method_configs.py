# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro
from nerfacc import ContractionType

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.hyperspectral_datamanager import HyperspectralDataManagerConfig
from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.nerfacto_field import InputWavelengthStyle
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfstudio.models.nerfplayer_ngp import NerfplayerNGPModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures. This model will be continually updated.",
    "hs-nerfacto1": "Hyperspectral NeRF, 128 output channels, 1 density.",
    "hs-nerfacto2": "Hyperspectral NeRF, 128 output channels, 128 densities.",
    "rgb-alpha-nerfacto": "Multiple alpha channels (one for each color).",
    "depth-nerfacto": "Nerfacto with depth supervision.",
    "instant-ngp": "Implementation of Instant-NGP. Recommended real-time model for unbounded scenes.",
    "instant-ngp-bounded": "Implementation of Instant-NGP. Recommended for bounded real and synthetic scenes",
    "mipnerf": "High quality model for bounded scenes. (slow)",
    "semantic-nerfw": "Predicts semantic segmentations and filters out transient objects.",
    "vanilla-nerf": "Original NeRF model. (slow)",
    "tensorf": "tensorf",
    "dnerf": "Dynamic-NeRF model. (slow)",
    "phototourism": "Uses the Phototourism data.",
    "nerfplayer-nerfacto": "NeRFPlayer with nerfacto backbone.",
    "nerfplayer-ngp": "NeRFPlayer with InstantNGP backbone.",
}

method_configs["nerfacto"] = TrainerConfig(
    method_name="nerfacto",
    steps_per_eval_batch=50,
    steps_per_eval_image=500,
    steps_per_eval_all_images=5000,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="wandb",
)

method_configs["nerfacto-camera-pose-refinement"] = TrainerConfig(
    method_name="nerfacto-camera-pose-refinement",
    steps_per_eval_batch=9999999,
    steps_per_eval_image=9999999,
    steps_per_eval_all_images=9999999,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=60001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                train_split_percentage=1.0
            ),
            train_num_rays_per_batch=1 << 14,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=SchedulerConfig(max_steps=50000)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-8),
            # "scheduler": None,
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=4000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-8),
            # "scheduler": None,
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=2000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=7008),
    vis="viewer wandb",
)

method_configs["hs-nerfacto1"] = TrainerConfig(
    method_name="hs-nerfacto1",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    save_only_latest_checkpoint=False,
    max_num_iterations=30001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            # train_num_images_to_sample_from=32,  # This might be needed to not run out of GPU memory
            # train_num_times_to_repeat_images=250,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  num_output_color_channels=128),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    # vis="viewer",  # wandb
    vis="wandb",
)

method_configs["hs-nerfacto2"] = TrainerConfig(
    method_name="hs-nerfacto2",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    save_only_latest_checkpoint=False,
    max_num_iterations=30001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            # train_num_images_to_sample_from=32,  # This might be needed to not run out of GPU memory
            # train_num_times_to_repeat_images=250,
            # eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  num_output_color_channels=128,
                                  num_density_channels=128),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    # vis="viewer",  # wandb
    vis="wandb",
)

method_configs["hs-nerfacto3"] = TrainerConfig(
    method_name="hs-nerfacto3",
    steps_per_eval_batch=50,
    steps_per_eval_image=500,
    steps_per_eval_all_images=5000,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            # 2 GPUs:
            #  16 channels - 17GB
            #  22 channels - 21GB
            #  25 channels - 23GB <- better use 24 channels to be safe, in case batches overflow
            # 1 GPU:
            #  25 channels - 21GB
            dataparser=NerfstudioDataParserConfig(
                # num_images_to_use=36,
                num_hyperspectral_channels=128,
                # num_hyperspectral_channels=24,
            ),
            # train_num_rays_per_batch=4096 // 64,
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096 // 32,
            # IMPORTANT - to resume a run, use CLI arg --pipeline.datamanager.camera-optimizer.optimizer.lr "5e-6"
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            # train_num_images_to_sample_from=32,  # This might be needed to not run out of GPU memory
            # train_num_times_to_repeat_images=250,
            # eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 9,
                                  num_output_color_channels=128,
                                #   num_output_color_channels=24,
                                  num_density_channels=1,
                                  num_wavelength_samples_per_batch=16,
                                  wavelength_style=InputWavelengthStyle.AFTER_BASE,
                                  num_wavelength_encoding_freqs=8),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 14),
    # vis="viewer",  # wandb
    vis="wandb",
    # vis="wandb",
)

method_configs["hs-nerfacto3-rgb"] = TrainerConfig(
    method_name="hs-nerfacto3-rgb",
    steps_per_eval_batch=50,
    steps_per_eval_image=500,
    steps_per_eval_all_images=25000,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            # train_num_rays_per_batch=4096 // 64,
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            # IMPORTANT - to resume a run, use CLI arg --pipeline.datamanager.camera-optimizer.lr "5e-6"
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            # train_num_images_to_sample_from=32,  # This might be needed to not run out of GPU memory
            # train_num_times_to_repeat_images=250,
            # eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  num_output_color_channels=3,
                                  num_density_channels=1,
                                  proposal_wavelength_use=False,
                                  wavelength_style=InputWavelengthStyle.AFTER_BASE,
                                  num_wavelength_encoding_freqs=2,
                                  train_wavelengths_every_nth=1),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-8),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 14),
    # vis="viewer",  # wandb
    # vis="wandb",
    vis="wandb",
)

method_configs["rgb-alpha-nerfacto"] = TrainerConfig(  # Approach 2, for RGB : hs-nerfacto2-rgb
    method_name="rgb-alpha-nerfacto",
    steps_per_eval_batch=50,
    steps_per_eval_image=500,
    steps_per_eval_all_images=5000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            # train_num_images_to_sample_from=32,  # This might be needed to not run out of GPU memory
            # train_num_times_to_repeat_images=250,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  num_output_color_channels=3,
                                  num_density_channels=3),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    # vis="viewer",
    vis="wandb",
)


method_configs["camera_pose_refinement_2"] = TrainerConfig(  # Approach 2, for RGB : hs-nerfacto2-rgb
    method_name="camera_pose_refinement_2",
    steps_per_eval_batch=9999999,
    steps_per_eval_image=9999999,
    steps_per_eval_all_images=9999999,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=75001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                train_split_percentage=1.0
            ),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2),
                                                    scheduler=SchedulerConfig(max_steps=50000)),
            # train_num_images_to_sample_from=32,  # This might be needed to not run out of GPU memory
            # train_num_times_to_repeat_images=250,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  num_output_color_channels=3,
                                  num_density_channels=3),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
            # "scheduler": None,
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=40000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, weight_decay=1e-9),
            # "scheduler": None,
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=40000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=7008),
    # vis="viewer",
    vis="viewer wandb",
)


method_configs["iccv-1"] = TrainerConfig(
    method_name="iccv-1",
    steps_per_eval_batch=50,
    steps_per_eval_image=500,
    steps_per_eval_all_images=25000,
    steps_per_save=5000,
    save_only_latest_checkpoint=False,
    max_num_iterations=25002,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            train_num_images_to_sample_from=16,  # This might be needed to not run out of GPU memory
            train_num_times_to_repeat_images=250,
            eval_num_images_to_sample_from=1,
            # eval_num_times_to_repeat_images=-1
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=2048,
                                #   train_num_rays_per_chunk=2048,
                                  #train_num_rays_per_chunk=99999999, # <-- delete this
                                  num_output_color_channels=128),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, amsgrad=True),
            # "scheduler": None,
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=20000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, amsgrad=True),
            "scheduler": None,
            "scheduler": SchedulerConfig(lr_final=1e-4, max_steps=20000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    # vis="viewer",  # wandb
    vis="wandb",
)

method_configs["iccv-2"] = TrainerConfig(
    method_name="iccv-2",
    steps_per_eval_batch=50,
    steps_per_eval_image=500,
    steps_per_eval_all_images=25000,
    steps_per_save=5000,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            train_num_images_to_sample_from=16,  # This might be needed to not run out of GPU memory
            train_num_times_to_repeat_images=250,
            eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                  num_output_color_channels=128,
                                  num_density_channels=128),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    # vis="viewer",  # wandb
    vis="wandb",
)


method_configs["iccv-3"] = TrainerConfig(
    method_name="iccv-3",
    steps_per_eval_batch=50,
    steps_per_eval_image=5000,
    steps_per_eval_all_images=25000,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            # 2 GPUs:
            #  16 channels - 17GB
            #  22 channels - 21GB
            #  25 channels - 23GB <- better use 24 channels to be safe, in case batches overflow
            # 1 GPU:
            #  25 channels - 21GB
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096 // 32,
            # IMPORTANT - to resume a run, use CLI arg --pipeline.datamanager.camera-optimizer.optimizer.lr "5e-6"
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            train_num_images_to_sample_from=16,  # This might be needed to not run out of GPU memory
            train_num_times_to_repeat_images=250,
            eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 9,
                                  num_output_color_channels=128,
                                #   num_output_color_channels=24,
                                  num_density_channels=1,
                                  num_wavelength_samples_per_batch=8,
                                  wavelength_style=InputWavelengthStyle.BEFORE_BASE,
                                  num_wavelength_encoding_freqs=8,
                                  train_wavelengths_every_nth=1),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 14),
    # vis="viewer",  # wandb
    vis="wandb",
    # vis="wandb",
)

method_configs["iccv-4"] = TrainerConfig(
    method_name="iccv-4",
    steps_per_eval_batch=499,
    steps_per_eval_image=5000,
    steps_per_eval_all_images=25000,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            # 2 GPUs:
            #  16 channels - 17GB
            #  22 channels - 21GB
            #  25 channels - 23GB <- better use 24 channels to be safe, in case batches overflow
            # 1 GPU:
            #  25 channels - 21GB
            dataparser=NerfstudioDataParserConfig(num_hyperspectral_channels=128),
            # train_num_rays_per_batch=4096 // 64,
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096 // 32,
            # IMPORTANT - to resume a run, use CLI arg --pipeline.datamanager.camera-optimizer.optimizer.lr "5e-6"
            camera_optimizer=CameraOptimizerConfig(mode="SO3xR3",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            train_num_images_to_sample_from=10,  # This might be needed to not run out of GPU memory
            train_num_times_to_repeat_images=250,
            eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 8,
                                  num_output_color_channels=128,
                                  proposal_wavelength_use=False,
                                #   num_output_color_channels=24,
                                  num_density_channels=1,
                                  num_wavelength_samples_per_batch=12,
                                  wavelength_style=InputWavelengthStyle.AFTER_BASE,
                                  num_wavelength_encoding_freqs=4,
                                  train_wavelengths_every_nth=2,
                                  geo_feat_dim=15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15, weight_decay=1e-6),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 14),
    # vis="viewer",  # wandb
    vis="wandb",
    # vis="wandb",
)

method_configs["iccv-5"] = TrainerConfig(
    method_name="iccv-5",
    steps_per_eval_batch=50,
    steps_per_eval_image=5000,
    steps_per_eval_all_images=25000,
    # steps_per_eval_all_images=1000,
    steps_per_save=2500,
    save_only_latest_checkpoint=False,
    max_num_iterations=25001,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=HyperspectralDataManagerConfig(
            # 2 GPUs:
            #  16 channels - 17GB
            #  22 channels - 21GB
            #  25 channels - 23GB <- better use 24 channels to be safe, in case batches overflow
            # 1 GPU:
            #  25 channels - 21GB
            dataparser=NerfstudioDataParserConfig(
                # num_images_to_use=36,
                num_hyperspectral_channels=128,
                # num_hyperspectral_channels=24,
            ),
            # train_num_rays_per_batch=4096 // 64,
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096 // 32,
            # IMPORTANT - to resume a run, use CLI arg --pipeline.datamanager.camera-optimizer.optimizer.lr "5e-6"
            camera_optimizer=CameraOptimizerConfig(mode="off",
                                                   optimizer=AdamOptimizerConfig(
                                                       lr=6e-4, eps=1e-8, weight_decay=1e-2)),
            train_num_images_to_sample_from=10,  # This might be needed to not run out of GPU memory
            train_num_times_to_repeat_images=250,
            # eval_num_images_to_sample_from=1,
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 9,
                                  num_output_color_channels=128,
                                  proposal_wavelength_use=True,
                                #   num_output_color_channels=24,
                                  num_density_channels=1,
                                  num_wavelength_samples_per_batch=6,
                                  wavelength_style=InputWavelengthStyle.NONE,
                                  num_wavelength_encoding_freqs=8,
                                  train_wavelengths_every_nth=2),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 14),
    # vis="viewer",  # wandb
    vis="wandb",
    # vis="wandb",
)


method_configs["depth-nerfacto"] = TrainerConfig(
    method_name="depth-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DepthDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=DepthNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["instant-ngp"] = TrainerConfig(
    method_name="instant-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(eval_num_rays_per_chunk=8192),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)


method_configs["instant-ngp-bounded"] = TrainerConfig(
    method_name="instant-ngp-bounded",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=InstantNGPDataParserConfig(), train_num_rays_per_batch=8192),
        model=InstantNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            contraction_type=ContractionType.AABB,
            render_step_size=0.001,
            max_num_samples_per_ray=48,
            near_plane=0.01,
            background_color="black",
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)


method_configs["mipnerf"] = TrainerConfig(
    method_name="mipnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), train_num_rays_per_batch=1024),
        model=VanillaModelConfig(
            _target=MipNerfModel,
            loss_coefficients={"rgb_loss_coarse": 0.1, "rgb_loss_fine": 1.0},
            num_coarse_samples=128,
            num_importance_samples=128,
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        }
    },
)

method_configs["semantic-nerfw"] = TrainerConfig(
    method_name="semantic-nerfw",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=SemanticDataManagerConfig(
            dataparser=FriendsDataParserConfig(), train_num_rays_per_batch=4096, eval_num_rays_per_batch=8192
        ),
        model=SemanticNerfWModelConfig(eval_num_rays_per_chunk=1 << 16),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
    vis="viewer",
)

method_configs["vanilla-nerf"] = TrainerConfig(
    method_name="vanilla-nerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=VanillaModelConfig(_target=NeRFModel),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["tensorf"] = TrainerConfig(
    method_name="tensorf",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=BlenderDataParserConfig(),
        ),
        model=TensoRFModelConfig(),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.001),
            "scheduler": SchedulerConfig(lr_final=0.0001, max_steps=30000),
        },
        "encodings": {
            "optimizer": AdamOptimizerConfig(lr=0.02),
            "scheduler": SchedulerConfig(lr_final=0.002, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["dnerf"] = TrainerConfig(
    method_name="dnerf",
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(dataparser=DNeRFDataParserConfig()),
        model=VanillaModelConfig(
            _target=NeRFModel,
            enable_temporal_distortion=True,
            temporal_distortion_params={"kind": TemporalDistortionKind.DNERF},
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
        "temporal_distortion": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4, eps=1e-08),
            "scheduler": None,
        },
    },
)

method_configs["phototourism"] = TrainerConfig(
    method_name="phototourism",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=VariableResDataManagerConfig(  # NOTE: one of the only differences with nerfacto
            dataparser=PhototourismDataParserConfig(),  # NOTE: one of the only differences with nerfacto
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerfplayer-nerfacto"] = TrainerConfig(
    method_name="nerfplayer-nerfacto",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=VanillaPipelineConfig(
        datamanager=DepthDataManagerConfig(
            dataparser=DycheckDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=NerfplayerNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)

method_configs["nerfplayer-ngp"] = TrainerConfig(
    method_name="nerfplayer-ngp",
    steps_per_eval_batch=500,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=DynamicBatchPipelineConfig(
        datamanager=DepthDataManagerConfig(dataparser=DycheckDataParserConfig(), train_num_rays_per_batch=8192),
        model=NerfplayerNGPModelConfig(
            eval_num_rays_per_chunk=8192,
            contraction_type=ContractionType.AABB,
            render_step_size=0.001,
            max_num_samples_per_ray=48,
            near_plane=0.01,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        }
    },
    viewer=ViewerConfig(num_rays_per_chunk=64000),
    vis="viewer",
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
