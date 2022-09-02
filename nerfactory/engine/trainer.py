# Copyright 2022 The Plenoptix Team. All rights reserved.
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
Code to train model.
"""
from __future__ import annotations

import functools
import logging
import os
import typing
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfactory.configs import base as cfg
from nerfactory.dataloaders.structs import DatasetInputs
from nerfactory.optimizers.optimizers import Optimizers, setup_optimizers
from nerfactory.pipelines.base import Pipeline
from nerfactory.utils import profiler, writer
from nerfactory.utils.callbacks import Callback
from nerfactory.utils.decorators import check_main_thread
from nerfactory.utils.writer import EventName, TimeWriter
from nerfactory.viewer.server import viewer_utils

logging.getLogger("PIL").setLevel(logging.WARNING)


def train_loop(local_rank: int, world_size: int, config: cfg.Config) -> Any:
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank (int): current rank of process
        world_size (int): total number of gpus available
        config (Config): config file specifying training regimen

    Returns:
        Any: TODO(): determine the return type
    """
    trainer = Trainer(config, local_rank, world_size)
    trainer.setup()
    trainer.train()
    return 0


class Trainer:
    """Trainer class

    Args:
        config (Config): The configuration object.
        local_rank (int, optional): Local rank of the process. Defaults to 0.
        world_size (int, optional): World size of the process. Defaults to 1.
    """

    def __init__(self, config: cfg.Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.trainer.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            logging.warning("Mixed precision is disabled for CPU training.")
        # dataset variables
        self.dataset_inputs_train: DatasetInputs
        # model variables
        self.pipeline: Pipeline
        self.optimizers: Optimizers
        self.start_step = 0
        # visualizer variable
        banner_messages = None
        self.visualizer_state = viewer_utils.VisualizerState(config.viewer)
        if config.viewer.enable:
            banner_messages = [f"Viewer at: {self.visualizer_state.viewer_url}"]
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)
        # training callbacks
        self.callbacks: List[Callback]
        # logging variables
        writer.setup_event_writers(
            config.logging, max_iter=config.trainer.max_num_iterations, banner_messages=banner_messages
        )
        profiler.setup_profiler(config.logging)

    def setup(self, test_mode=False):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode (bool, optional): Whether to setup for testing. Defaults to False.
        """
        self.pipeline: Pipeline = self.config.pipeline.setup(device=self.device, test_mode=test_mode)
        self.optimizers = setup_optimizers(self.config.optimizers, self.pipeline.get_param_groups())

        self._load_checkpoint()

        if self.world_size > 1:
            self.pipeline = typing.cast(
                Pipeline, typing.cast(Pipeline, DDP(self.pipeline, device_ids=[self.local_rank]))
            )
            dist.barrier(device_ids=[self.local_rank])

        # TODO(ethan): do this for pipeline, not pipeline.model
        self.callbacks = self.pipeline.model.get_training_callbacks()

    def train(self) -> None:
        """Train the model."""
        self.visualizer_state.init_scene(
            image_dataset=self.pipeline.dataloader.train_image_dataset,
            dataset_inputs=self.pipeline.dataloader.train_datasetinputs,
        )
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.trainer.max_num_iterations
            for step in range(self.start_step, self.start_step + num_iterations):

                # Note: if visualizer used, the rendering of the visualizer will be included in the iteration train time
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as t:
                    loss_metric_dict = self.train_iteration(step)
                    with TimeWriter(writer, EventName.ITER_VIS_TIME, step=step) as t:
                        self.visualizer_state.update_scene(step, self.pipeline.model)

                train_num_rays_per_batch = self.config.pipeline.dataloader.train_num_rays_per_batch
                writer.put_scalar(name=EventName.RAYS_PER_SEC, scalar=train_num_rays_per_batch / t.duration, step=step)

                if step != 0 and step % self.config.logging.steps_per_log == 0:
                    writer.put_dict(name="Loss/train-loss_metrics_dict", scalar_dict=loss_metric_dict, step=step)
                if step != 0 and self.config.trainer.steps_per_save and step % self.config.trainer.steps_per_save == 0:
                    self._save_checkpoint(self.config.trainer.model_dir, step)
                if step % self.config.trainer.steps_per_test == 0:
                    self.pipeline.get_eval_loss_dict(step=step)
                self._write_out_storage(step)

        self._write_out_storage(num_iterations)

    def _write_out_storage(self, step: int) -> None:
        """Perform writes only during appropriate time steps

        Args:
            step (int): Current training step.
        """
        if (
            step % self.config.logging.steps_per_log == 0
            or (self.config.trainer.steps_per_save and step % self.config.trainer.steps_per_save == 0)
            or step % self.config.trainer.steps_per_test == 0
            or step == self.config.trainer.max_num_iterations
        ):
            writer.write_out_storage()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.trainer.load_dir
        load_step = self.config.trainer.load_step
        if load_dir is not None and load_step is not None:
            load_path = os.path.join(load_dir, f"step-{load_step:09d}.ckpt")
            assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self.start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            logging.info("done loading checkpoint from %s", load_path)
        else:
            logging.info("No checkpoints to load, training from scratch")

    @check_main_thread
    def _save_checkpoint(self, output_dir: Path, step: int) -> None:
        """Save the model and optimizers

        Args:
            output_dir: directory to save the checkpoint
            step: number of steps in training for given checkpoint
        """
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / f"step-{step:09d}.ckpt"
        if hasattr(self.pipeline, "module"):
            pipeline = self.pipeline.module.state_dict()  # type: ignore
        else:
            pipeline = self.pipeline.state_dict()
        torch.save(
            {
                "step": step,
                "pipeline": pipeline,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )

    @profiler.time_function
    def train_iteration(self, step: int) -> Dict[str, torch.Tensor]:
        """Run one iteration with a batch of inputs.

        Args:
            step: Current training step.

        Returns:
            Dictionary of model losses.
        """
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str = self.device.split(":")[0]
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        self.grad_scaler.update()

        self.optimizers.scheduler_step_all(step)
        for callback in self.callbacks:
            callback.after_step(step)

        # Merging loss and metrics dict into a single output.
        loss_dict["loss"] = loss
        loss_dict.update(metrics_dict)
        return loss_dict
