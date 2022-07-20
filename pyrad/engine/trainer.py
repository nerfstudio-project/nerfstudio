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
import functools
import logging
import os
from typing import Dict

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtyping import TensorType

from pyrad.cameras.rays import RayBundle
from pyrad.data.dataloader import EvalDataloader, setup_dataset_eval, setup_dataset_train
from pyrad.graphs.base import setup_graph
from pyrad.optimizers.optimizers import setup_optimizers
from pyrad.utils import profiler, writer
from pyrad.utils.config import Config
from pyrad.utils.decorators import check_main_thread
from pyrad.utils.writer import EventName, TimeWriter
from pyrad.viewer.server import viewer_utils

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class

    Args:
        config (Config): The configuration object.
        local_rank (int, optional): Local rank of the process. Defaults to 0.
        world_size (int, optional): World size of the process. Defaults to 1.
    """

    def __init__(self, config: Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.trainer.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            logging.warning("Mixed precision is disabled for CPU training.")
        # dataset variables
        self.dataset_inputs_train = None
        self.dataloader_train = None
        self.dataloader_eval = None
        # model variables
        self.graph = None
        self.optimizers = None
        self.start_step = 0
        # logging variables
        writer.setup_event_writers(config.logging, max_iter=config.trainer.max_num_iterations)
        profiler.setup_profiler(config.logging)
        # visualizer variable
        self.visualizer_state = viewer_utils.VisualizerState(config.viewer)

        self.grad_scaler = GradScaler(enabled=self.mixed_precision)

    def setup(self, test_mode=False):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode (bool, optional): Whether to setup for testing. Defaults to False.
        """
        self.dataset_inputs_train, self.dataloader_train = setup_dataset_train(self.config.data, device=self.device)
        _, self.dataloader_eval = setup_dataset_eval(self.config.data, test_mode=test_mode, device=self.device)
        self.graph = setup_graph(self.config.graph, self.dataset_inputs_train, device=self.device)
        self.optimizers = setup_optimizers(self.config.optimizers, self.graph.get_param_groups())

        if self.config.trainer.resume_train.load_dir:
            self._load_checkpoint()

        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

        self.graph.register_callbacks()

    @classmethod
    def get_aggregated_loss(cls, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns the aggregated losses and the scalar for calling .backwards() on.
        # TODO: move this out to another file/class/etc.
        """
        # TODO(ethan): add loss weightings here from a config
        # e.g. weighted_losses = map(lambda k: some_weight_dict[k] * loss_dict[k], loss_dict.keys())
        weighted_losses = loss_dict.values()
        return functools.reduce(torch.add, weighted_losses)

    def train(self) -> None:
        """Train the model."""
        self.visualizer_state.init_scene(
            image_dataset=self.dataloader_train.image_sampler.dataset, dataset_inputs=self.dataset_inputs_train
        )
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.trainer.max_num_iterations
            iter_dataloader_train = iter(self.dataloader_train)
            for step in range(self.start_step, self.start_step + num_iterations):
                with TimeWriter(writer, EventName.ITER_LOAD_TIME, step=step):
                    ray_indices, batch = next(iter_dataloader_train)

                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as t:
                    loss_dict = self.train_iteration(ray_indices, batch, step)
                writer.put_scalar(name=EventName.RAYS_PER_SEC, scalar=ray_indices.shape[0] / t.duration, step=step)

                if step != 0 and step % self.config.logging.steps_per_log == 0:
                    writer.put_dict(name="Loss/train-loss_dict", scalar_dict=loss_dict, step=step)
                if step != 0 and self.config.trainer.steps_per_save and step % self.config.trainer.steps_per_save == 0:
                    self._save_checkpoint(self.config.trainer.model_dir, step)
                if step % self.config.trainer.steps_per_test == 0:
                    self.eval_with_dataloader(self.dataloader_eval, step=step)
                self._write_out_storage(step)
                self.visualizer_state.update_scene(step, self.graph)

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
        """Helper function to load graph and optimizer from prespecified checkpoint"""
        load_config = self.config.trainer.resume_train
        load_path = os.path.join(load_config.load_dir, f"step-{load_config.load_step:09d}.ckpt")
        assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        self.start_step = loaded_state["step"] + 1
        # load the checkpoints for graph and optimizer
        self.graph.load_graph(loaded_state)
        self.optimizers.load_optimizers(loaded_state)
        self.grad_scaler.load_state_dict(loaded_state["scaler"])
        logging.info("done loading checkpoint from %s", load_path)

    @check_main_thread
    def _save_checkpoint(self, output_dir: str, step: int) -> None:
        """Save the model and optimizers

        Args:
            output_dir: directory to save the checkpoint
            step: number of steps in training for given checkpoint
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ckpt_path = os.path.join(output_dir, f"step-{step:09d}.ckpt")
        torch.save(
            {
                "step": step,
                "model": self.graph.module.state_dict() if hasattr(self.graph, "module") else self.graph.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )

    @profiler.time_function
    def train_iteration(self, ray_indices: TensorType["num_rays", 3], batch: dict, step: int) -> Dict[str, float]:
        """Run one iteration with a batch of inputs.

        Args:
            ray_indices: Contains camera, row, and col indicies for target rays.
            batch: Batch of training data.
            step: Current training step.

        Returns:
            Dict[str, float]: Dictionary of model losses.
        """
        self.optimizers.zero_grad_all()
        with torch.autocast(device_type=ray_indices.device.type, enabled=self.mixed_precision):
            _, loss_dict = self.graph.forward(ray_indices, batch=batch)
            loss = loss_dict["aggregated_loss"]
        self.grad_scaler.scale(loss).backward()
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        self.grad_scaler.update()

        self.optimizers.scheduler_step_all(step)
        if self.graph.callbacks:
            for func_ in self.graph.callbacks:
                func_.after_step(step)
        return loss_dict

    @profiler.time_function
    def test_image(self, camera_ray_bundle: RayBundle, batch: dict, step: int = None) -> float:
        """Test a specific image.

        Args:
            camera_ray_bundle: Bundle of test rays.
            batch: Batch of data.
            step: Current training step.

        Returns:
            float: PSNR
        """
        self.graph.eval()
        image_idx = int(camera_ray_bundle.camera_indices[0, 0])
        outputs = self.graph.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        psnr = self.graph.log_test_image_outputs(image_idx, step, batch, outputs)
        self.graph.train()
        return psnr

    def eval_with_dataloader(self, dataloader: EvalDataloader, step: int = None) -> None:
        """Run evaluation with a given dataloader.

        Args:
            dataloader: Evaluation dataloader.
            step: Current training iteration.
        """
        for camera_ray_bundle, batch in dataloader:
            self.test_image(camera_ray_bundle, batch, step=step)
