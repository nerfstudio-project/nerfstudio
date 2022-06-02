"""
Code to train model.
"""
import copy
import logging
import os
from time import time
from typing import Callable, Dict, List

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from pyrad.nerf.dataset.utils import DatasetInputs, get_dataset_inputs_from_dataset_config
from pyrad.nerf.image_sampler import CacheImageSampler
from pyrad.nerf.optimizers import Optimizers
from pyrad.nerf.pixel_sampler import PixelSampler
from pyrad.utils import profiler, writer
from pyrad.utils.callbacks import update_occupancy
from pyrad.utils.decorators import check_main_thread
from pyrad.utils.misc import get_dict_to_torch, instantiate_from_dict_config

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class"""

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1):
        """_summary_

        Args:
            config (DictConfig): _description_
            local_rank (int, optional): _description_. Defaults to 0.
            world_size (int, optional): _description_. Defaults to 1.
            cpu (bool, optional): Whether or not to use the CPU.
        """
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        # dataset variables
        self.train_image_dataset = None
        self.train_image_sampler = None
        self.train_pixel_sampler = None
        self.val_image_dataset = None
        self.val_image_intrinsics = None
        self.val_image_camera_to_world = None
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
        dataset_inputs_train = get_dataset_inputs_from_dataset_config(**self.config.data.dataset, split="train")
        if test_mode:
            dataset_inputs_eval = get_dataset_inputs_from_dataset_config(**self.config.data.dataset, split="test")
        else:
            config_data_dataset_val = copy.deepcopy(self.config.data.dataset)
            config_data_dataset_val.downscale_factor = self.config.data.val_downscale_factor
            dataset_inputs_eval = get_dataset_inputs_from_dataset_config(**config_data_dataset_val, split="val")
        self.setup_dataset_train(dataset_inputs_train)
        self.setup_dataset_eval(dataset_inputs_eval)
        self.setup_graph(dataset_inputs_train)

    @profiler.time_function
    def setup_dataset_eval(self, dataset_inputs: DatasetInputs):
        """Helper method to load test or val dataset based on test/train mode"""
        self.val_image_dataset = instantiate_from_dict_config(
            self.config.data.image_dataset,
            image_filenames=dataset_inputs.image_filenames,
            downscale_factor=dataset_inputs.downscale_factor,
            semantics=dataset_inputs.semantics,
            alpha_color=dataset_inputs.alpha_color,
        )
        self.val_image_intrinsics = dataset_inputs.intrinsics
        self.val_image_camera_to_world = dataset_inputs.camera_to_world

    @profiler.time_function
    def setup_dataset_train(self, dataset_inputs: DatasetInputs):
        """_summary_"""
        self.train_image_dataset = instantiate_from_dict_config(
            self.config.data.image_dataset,
            image_filenames=dataset_inputs.image_filenames,
            downscale_factor=dataset_inputs.downscale_factor,
            semantics=dataset_inputs.semantics,
            alpha_color=dataset_inputs.alpha_color,
        )  # ImageDataset
        self.train_image_sampler = CacheImageSampler(
            self.train_image_dataset,
            num_samples_to_collate=len(self.train_image_dataset)
            if self.config.data.image_sampler.num_images_to_sample_from == 0
            else self.config.data.image_sampler.num_images_to_sample_from,
            num_times_to_repeat_images=self.config.data.image_sampler.num_times_to_repeat_images,
            device=self.device if self.config.data.image_sampler.move_to_graph_device else "cpu",
        )  # ImageSampler
        self.train_pixel_sampler = PixelSampler(
            num_rays_per_batch=self.config.data.pixel_sampler.num_rays_per_batch, keep_full_image=False
        )  # PixelSampler

    @profiler.time_function
    def setup_graph(self, dataset_inputs: DatasetInputs):
        """Setup the graph. The dataset inputs should be set with the training data.

        Args:
            dataset_inputs (DatasetInputs): The inputs which will be used to define the camera parameters.
        """
        self.graph = instantiate_from_dict_config(
            self.config.graph.network,
            intrinsics=dataset_inputs.intrinsics,
            camera_to_world=dataset_inputs.camera_to_world,
            scene_bounds=dataset_inputs.scene_bounds,
            stuff_classes=dataset_inputs.semantics.stuff_classes,
            stuff_colors=dataset_inputs.semantics.stuff_colors,
        )
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
            load_path (str): path from which to load the model

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
            model (Graph): Graph model to be saved
            optimizers (Optimizers): Optimizers to be saved
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
        train_start = time()
        num_iterations = self.config.graph.max_num_iterations
        iter_train_image_sampler = iter(self.train_image_sampler)
        for step in range(self.start_step, self.start_step + num_iterations):
            data_start = time()
            image_batch = next(iter_train_image_sampler)
            image_batch = get_dict_to_torch(image_batch, device=self.device)
            batch = self.train_pixel_sampler.sample(image_batch)
            writer.put_time(
                name=writer.EventName.ITER_LOAD_TIME,
                start_time=data_start,
                end_time=time(),
                step=step,
                avg_over_iters=True,
            )
            iter_start = time()
            loss_dict = self.train_iteration(batch, step, _callback=[update_occupancy])
            writer.put_time(
                name=writer.EventName.RAYS_PER_SEC,
                start_time=iter_start,
                end_time=time(),
                step=step,
                avg_over_iters=True,
                avg_over_batch=batch["indices"].shape[0],
            )
            writer.put_time(
                name=writer.EventName.ITER_TRAIN_TIME,
                start_time=iter_start,
                end_time=time(),
                step=step,
                avg_over_iters=True,
                update_eta=True,
            )

            if step != 0 and step % self.config.logging.steps_per_log == 0:
                writer.put_dict(name="Loss/train-loss_dict", scalar_dict=loss_dict, step=step)
            if step != 0 and self.config.graph.steps_per_save and step % self.config.graph.steps_per_save == 0:
                self.save_checkpoint(self.config.graph.model_dir, step)
            if step % self.config.graph.steps_per_test == 0:
                for image_idx in self.config.data.val_image_indices:
                    _ = self.test_image(image_idx=image_idx, step=step)
            self._write_out_storage(step)

        writer.put_time(
            name=writer.EventName.TOTAL_TRAIN_TIME, start_time=train_start, end_time=time(), step=num_iterations
        )
        self._write_out_storage(num_iterations)

    def _write_out_storage(self, step):
        """Perform writes only during appropriate time steps"""
        if (
            step % self.config.logging.steps_per_log == 0
            or (self.config.graph.steps_per_save and step % self.config.graph.steps_per_save == 0)
            or step % self.config.graph.steps_per_test == 0
            or step == self.config.graph.max_num_iterations
        ):
            writer.write_out_storage()

    @profiler.time_function
    def train_iteration(self, batch: dict, step: int, _callback: List[Callable] = None):
        """Run one iteration with a batch of inputs."""
        ray_indices = batch["indices"]
        _, loss_dict = self.graph.forward(ray_indices, batch=batch, step=step)
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
    def test_image(self, image_idx, step):
        """Test a specific image."""
        self.graph.eval()
        intrinsics = self.val_image_intrinsics[image_idx].to(self.device)
        camera_to_world = self.val_image_camera_to_world[image_idx].to(self.device)
        chunk_size = self.config.data.val_num_rays_per_chunk
        training_camera_index = image_idx  # TODO(ethan): change this because training and test should not be the same
        outputs = self.graph.get_outputs_for_camera(
            intrinsics, camera_to_world, chunk_size=chunk_size, training_camera_index=training_camera_index
        )
        batch = self.val_image_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device)
        psnr = self.graph.log_test_image_outputs(image_idx, step, batch, outputs)
        self.graph.train()
        return psnr
