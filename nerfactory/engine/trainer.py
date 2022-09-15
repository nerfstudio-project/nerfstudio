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

import dataclasses
import functools
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfactory.configs import base as cfg
from nerfactory.optimizers.optimizers import Optimizers, setup_optimizers
from nerfactory.pipelines.base import Pipeline
from nerfactory.utils import profiler, writer
from nerfactory.utils.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfactory.utils.decorators import check_main_thread
from nerfactory.utils.writer import EventName, TimeWriter
from nerfactory.viewer.server import viewer_utils

logging.getLogger("PIL").setLevel(logging.WARNING)


def train_loop(local_rank: int, world_size: int, config: cfg.Config) -> Any:
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    trainer = Trainer(config, local_rank, world_size)
    trainer.setup()
    trainer.train()
    return 0


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process. Defaults to 0.
        world_size: World size of the process. Defaults to 1.
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
        # model variables
        self.pipeline: Pipeline
        self.optimizers: Optimizers
        self.start_step = 0
        # visualizer variable
        banner_messages = None
        self.visualizer_state = viewer_utils.VisualizerState(config.viewer, config_base_dir=self.config.base_dir)
        if config.viewer.enable:
            banner_messages = [f"Viewer at: {self.visualizer_state.viewer_url}"]
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)
        # training callbacks
        self.callbacks: List[TrainingCallback]
        # logging variables
        writer.setup_event_writers(
            config.logging, max_iter=config.trainer.max_num_iterations, banner_messages=banner_messages
        )
        profiler.setup_profiler(config.logging)

        writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)

    def setup(self, test_mode=False):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode: Whether to setup for testing. Defaults to False.
        """
        self.pipeline: Pipeline = self.config.pipeline.setup(
            device=self.device, test_mode=test_mode, world_size=self.world_size, local_rank=self.local_rank
        )
        self.optimizers = setup_optimizers(self.config.optimizers, self.pipeline.get_param_groups())

        self._load_checkpoint()

        # TODO(ethan): do this for pipeline, not pipeline.model
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(optimizers=self.optimizers, grad_scaler=self.grad_scaler, pipeline=self.pipeline)
        )

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_input_dataset is not None, "Missing DatsetInputs"

        self.visualizer_state.init_scene(
            dataset=self.pipeline.datamanager.train_input_dataset, start_train=self.config.viewer.start_train
        )
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.trainer.max_num_iterations
            for step in range(self.start_step, self.start_step + num_iterations):
                # if the visualizer used, the rendering of the visualizer will be included in the iteration train time
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                        )

                    loss_metric_dict = self.train_iteration(step)

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                    with TimeWriter(writer, EventName.ITER_VIS_TIME, step=step) as vis_t:
                        num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
                        try:
                            self.visualizer_state.update_scene(step, self.pipeline.model, num_rays_per_batch)
                        except RuntimeError:
                            time.sleep(0.03)  # sleep to allow buffer to reset
                            assert self.visualizer_state.vis is not None
                            self.visualizer_state.vis["renderingState/log_errors"].write(
                                "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                            )

                if step % self.config.logging.steps_per_log == 0:
                    writer.put_dict(name="Train Metrics and Loss", scalar_dict=loss_metric_dict, step=step)
                if step != 0 and self.config.trainer.steps_per_save and step % self.config.trainer.steps_per_save == 0:
                    self._save_checkpoint(self.config.trainer.model_dir, step)
                if step % self.config.trainer.steps_per_test == 0:
                    metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
                    writer.put_dict(name="Eval Metrics", scalar_dict=metrics_dict, step=step)
                    # write out the image dictionary returned too
                self._update_rays_per_sec(train_t, vis_t, step)
                self._write_out_storage(step)

        self._write_out_storage(num_iterations)

    def _update_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int):
        """Performs update on rays/sec calclation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _write_out_storage(self, step: int) -> None:
        """Perform writes only during appropriate time steps

        Args:
            step: Current training step.
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
        if load_dir is not None:
            load_step = self.config.trainer.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self.start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            logging.info("done loading checkpoint from %s", load_path)
        else:
            logging.info("No checkpoints to load, training from scratch")

        logging.info("saving checkpoints to: %s", self.config.trainer.model_dir)

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
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
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

        # Merging loss and metrics dict into a single output.
        loss_dict["loss"] = loss
        loss_dict.update(metrics_dict)
        return loss_dict
