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
import logging
import os
from typing import Callable, Dict, List

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torchtyping import TensorType

from pyrad.cameras.rays import RayBundle
from pyrad.data.dataloader import EvalDataloader, TrainDataloader
from pyrad.data.utils import DatasetInputs, get_dataset_inputs_from_dataset_config
from pyrad.optimizers.optimizers import Optimizers
from pyrad.utils import profiler, writer
from pyrad.utils.callbacks import update_occupancy
from pyrad.utils.decorators import check_main_thread
from pyrad.utils.misc import instantiate_from_dict_config
from pyrad.utils.writer import EventName, TimeWriter

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class

    Args:
        config (DictConfig): _description_
        local_rank (int, optional): _description_. Defaults to 0.
        world_size (int, optional): _description_. Defaults to 1.
        cpu (bool, optional): Whether or not to use the CPU.
    """

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        # dataset variables
        self.dataloader_train = None
        self.dataloader_eval = None
        # model variables
        self.graph = None
        self.optimizers = None
        self.start_step = 0
        # logging variables
        writer.setup_event_writers(config)
        profiler.setup_profiler(config.logging)
        self.device = "cpu" if self.world_size == 0 else f"cuda:{self.local_rank}"

    def setup(self, test_mode=False):
        """Setup the Trainer by calling other setup functions."""
        dataset_inputs_train = get_dataset_inputs_from_dataset_config(
            **self.config.data.dataset_inputs_train, split="train"
        )
        eval_split = "test" if test_mode else "val"
        dataset_inputs_eval = get_dataset_inputs_from_dataset_config(
            **self.config.data.dataset_inputs_eval, split=eval_split
        )
        self.setup_dataset_train(dataset_inputs_train)
        self.setup_dataset_eval(dataset_inputs_eval)
        self.setup_graph(dataset_inputs_train)

    @profiler.time_function
    def setup_dataset_train(self, dataset_inputs: DatasetInputs):
        """_summary_"""
        # ImageDataset
        image_dataset_train = instantiate_from_dict_config(
            self.config.data.image_dataset_train, **dataset_inputs.as_dict()
        )
        # ImageSampler
        image_sampler_train = instantiate_from_dict_config(
            self.config.data.dataloader_train.image_sampler, dataset=image_dataset_train, device=self.device
        )
        # PixelSampler
        pixel_sampler_train = instantiate_from_dict_config(self.config.data.dataloader_train.pixel_sampler)
        # Dataloader
        self.dataloader_train = TrainDataloader(image_sampler_train, pixel_sampler_train)

    @profiler.time_function
    def setup_dataset_eval(self, dataset_inputs: DatasetInputs):
        """Helper method to load test or val dataset based on test/train mode"""
        image_dataset_eval = instantiate_from_dict_config(
            self.config.data.image_dataset_eval, **dataset_inputs.as_dict()
        )
        self.dataloader_eval = instantiate_from_dict_config(
            self.config.data.dataloader_eval,
            image_dataset=image_dataset_eval,
            device=self.device,
            **dataset_inputs.as_dict(),
        )

    @profiler.time_function
    def setup_graph(self, dataset_inputs: DatasetInputs):
        """Setup the graph. The dataset inputs should be set with the training data.

        Args:
            dataset_inputs (DatasetInputs): The inputs which will be used to define the camera parameters.
        """
        self.graph = instantiate_from_dict_config(self.config.graph.network, **dataset_inputs.as_dict())
        self.graph.to(self.device)

        self.setup_optimizers()  # NOTE(ethan): can this be before DDP?

        if self.config.graph.resume_train.load_dir:
            self.load_checkpoint(self.config.graph.resume_train)

        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

    def setup_optimizers(self):
        """_summary_"""
        self.optimizers = Optimizers(self.config.graph.param_groups, self.graph.get_param_groups())

    def load_checkpoint(self, load_config: DictConfig) -> int:
        """Load the checkpoint from the given path

        Args:
            load_config (DictConfig): Configuration for loading a model.

        Returns:
            int: step iteration associated with the loaded checkpoint
        """
        load_path = os.path.join(load_config.load_dir, f"step-{load_config.load_step:09d}.ckpt")
        assert os.path.exists(load_path), f"Checkpoint {load_path} does not exist"
        loaded_state = torch.load(load_path, map_location="cpu")
        self.graph.load_state_dict({key.replace("module.", ""): value for key, value in loaded_state["model"].items()})
        for k, v in loaded_state["optimizers"].items():
            self.optimizers.optimizers[k].load_state_dict(v)
        self.start_step = loaded_state["step"] + 1
        logging.info("done loading checkpoint from %s", load_path)

    @check_main_thread
    def save_checkpoint(self, output_dir: str, step: int) -> None:
        """Save the model and optimizers

        Args:
            output_dir (str): directory to save the checkpoint
            step (int): number of steps in training for given checkpoint
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ckpt_path = os.path.join(output_dir, f"step-{step:09d}.ckpt")
        torch.save(
            {
                "step": step,
                "model": self.graph.module.state_dict() if hasattr(self.graph, "module") else self.graph.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
            },
            ckpt_path,
        )

    @classmethod
    def get_aggregated_loss(cls, loss_dict: Dict[str, torch.tensor]):
        """Returns the aggregated losses and the scalar for calling .backwards() on.
        # TODO: move this out to another file/class/etc.
        """
        loss_sum = 0.0
        for loss_name in loss_dict.keys():
            # TODO(ethan): add loss weightings here from a config
            loss_sum += loss_dict[loss_name]
        return loss_sum

    def train(self) -> None:
        """_summary_"""
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.graph.max_num_iterations
            iter_dataloader_train = iter(self.dataloader_train)
            for step in range(self.start_step, self.start_step + num_iterations):
                with TimeWriter(writer, EventName.ITER_LOAD_TIME, step=step):
                    ray_indices, batch = next(iter_dataloader_train)

                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as t:
                    loss_dict = self.train_iteration(ray_indices, batch, step, _callback=[update_occupancy])
                writer.put_scalar(name=EventName.RAYS_PER_SEC, scalar=ray_indices.shape[0] / t.duration, step=step)

                if step != 0 and step % self.config.logging.steps_per_log == 0:
                    writer.put_dict(name="Loss/train-loss_dict", scalar_dict=loss_dict, step=step)
                if step != 0 and self.config.graph.steps_per_save and step % self.config.graph.steps_per_save == 0:
                    self.save_checkpoint(self.config.graph.model_dir, step)
                if step % self.config.graph.steps_per_test == 0:
                    self.eval_with_dataloader(self.dataloader_eval, step=step)
                self._write_out_storage(step)

        self._write_out_storage(num_iterations)

    def _write_out_storage(self, step: int) -> None:
        """Perform writes only during appropriate time steps

        Args:
            step (int): Current training step.
        """
        if (
            step % self.config.logging.steps_per_log == 0
            or (self.config.graph.steps_per_save and step % self.config.graph.steps_per_save == 0)
            or step % self.config.graph.steps_per_test == 0
            or step == self.config.graph.max_num_iterations
        ):
            writer.write_out_storage()

    @profiler.time_function
    def train_iteration(
        self, ray_indices: TensorType["num_rays", 3], batch: dict, step: int, _callback: List[Callable] = None
    ) -> Dict[str, float]:
        """Run one iteration with a batch of inputs.

        Args:
            ray_indices (TensorType["num_rays", 3]): Contains camera, row, and col indicies for target rays.
            batch (dict): Batch of training data.
            step (int): Current training step.

        Returns:
            Dict[str, float]: Dictionary of model losses.
        """
        _, loss_dict = self.graph.forward(ray_indices, batch=batch)
        loss = loss_dict["aggregated_loss"]
        self.optimizers.zero_grad_all()
        loss.backward()
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(step)
        if _callback:
            for _func in _callback:
                _func(self.graph)
        return loss_dict

    @profiler.time_function
    def test_image(self, camera_ray_bundle: RayBundle, batch: dict, step: int = None) -> float:
        """Test a specific image.

        Args:
            camera_ray_bundle (RayBundle): Bundle of test rays.
            batch (dict): Batch of data.
            step (int): Current training step.

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
            dataloader (EvalDataLoader): Evaluation dataloader.
            step (int): Current training iteration.
        """
        for camera_ray_bundle, batch in dataloader:
            self.test_image(camera_ray_bundle, batch, step=step)
