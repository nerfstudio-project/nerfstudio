"""
Code to train model.
"""
import logging
import os
from time import time
from typing import Dict

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import mattport.utils.writer
from mattport.nerf.dataset.collate import CollateIterDataset, collate_batch_size_one
from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import DatasetInputs, get_dataset_inputs_dict
from mattport.nerf.metrics import get_psnr
from mattport.nerf.optimizers import Optimizers
from mattport.utils import profiler
from mattport.utils.decorators import check_main_thread
from mattport.utils.stats_tracker import Stats, StatsTracker

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class"""

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1, cpu=False):
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
        self.train_dataset = None
        self.train_dataloader = None
        self.val_image_dataset = None
        # model variables
        self.graph = None
        self.optimizers = None
        self.start_step = 0
        # logging variables
        self.is_main_thread = world_size != 0 and local_rank % world_size == 0
        writer = getattr(mattport.utils.writer, self.config.logging.writer.type)
        self.writer = writer(self.is_main_thread, save_dir=self.config.logging.writer.save_dir)
        self.stats = StatsTracker(config, self.is_main_thread)
        if not profiler.PROFILER and self.config.logging.enable_profiler:
            profiler.PROFILER = profiler.Profiler(config, self.is_main_thread)
        self.device = f"cuda:{self.local_rank}" if not cpu else "cpu"

    @profiler.time_function
    def setup(self):
        """Setup the Trainer by calling other setup functions."""
        dataset_inputs_dict = get_dataset_inputs_dict(**self.config.data.dataset)
        self.setup_datasets(dataset_inputs_dict)
        self.setup_graph(dataset_inputs_dict["train"])

    def collate_fn(self, batch_list):
        """TODO(ethan): I need to replace this.
        I'm only using this for multiprocess pickle issues for now.
        """
        return collate_batch(batch_list, self.config.data.dataloader.num_rays_per_batch, keep_full_image=False)

    @profiler.time_function
    def setup_datasets(self, dataset_inputs_dict: Dict[str, DatasetInputs]):
        """_summary_"""
        self.train_image_dataset = ImageDataset(
            image_filenames=dataset_inputs_dict["train"].image_filenames,
            downscale_factor=dataset_inputs_dict["train"].downscale_factor,
            alpha_color=dataset_inputs_dict["train"].alpha_color,
        )

        self.train_dataset = CollateIterDataset(
            self.train_image_dataset,
            collate_fn=self.collate_fn,
            num_samples_to_collate=self.config.data.dataloader.num_images_to_sample_from,
            num_times_to_repeat=self.config.data.dataloader.num_times_to_repeat_images,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.config.data.dataloader.num_workers,
            collate_fn=collate_batch_size_one,
            pin_memory=True,
        )
        self.val_image_dataset = ImageDataset(
            image_filenames=dataset_inputs_dict["val"].image_filenames,
            downscale_factor=dataset_inputs_dict["val"].downscale_factor,
            alpha_color=dataset_inputs_dict["val"].alpha_color,
        )

    @profiler.time_function
    def setup_graph(self, dataset_inputs: DatasetInputs):
        """Setup the graph. The dataset inputs should be set with the training data.

        Args:
            dataset_inputs (DatasetInputs): The inputs which will be used to define the camera parameters.
        """
        self.graph = instantiate(
            self.config.graph.network,
            intrinsics=dataset_inputs.intrinsics,
            camera_to_world=dataset_inputs.camera_to_world,
        ).to(self.device)
        self.setup_optimizers()

        if self.config.graph.resume_train.load_dir:
            self.load_checkpoint(self.config.graph.resume_train)

        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

    def setup_optimizers(self):
        """_summary_"""
        # TODO(ethan): handle different world sizes
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
    def get_aggregated_loss(cls, losses: Dict[str, torch.tensor]):
        """Returns the aggregated losses and the scalar for calling .backwards() on.
        # TODO: move this out to another file/class/etc.
        """
        loss_sum = 0.0
        loss_dict = {}
        for loss_name in losses.keys():
            # TODO(ethan): add loss weightings here from a config
            loss_sum += losses[loss_name]
            loss_dict[loss_name] = float(losses[loss_name])
        return loss_sum, loss_dict

    def train(self) -> None:
        """_summary_"""
        train_start = time()
        num_iterations = self.config.graph.max_num_iterations
        iter_dataset = iter(self.train_dataloader)
        for i, step in enumerate(range(self.start_step, self.start_step + num_iterations)):
            data_start = time()
            batch = next(iter_dataset)
            self.stats.update_time(Stats.ITER_LOAD_TIME, data_start, time(), step=step)

            iter_start = time()
            loss_dict = self.train_iteration(batch, step)
            self.stats.update_time(
                Stats.RAYS_PER_SEC, iter_start, time(), step=step, batch_size=batch["indices"].shape[0]
            )
            self.stats.update_time(Stats.ITER_TRAIN_TIME, iter_start, time(), step=step)

            if step != 0 and step % self.config.logging.steps_per_log == 0:
                self.writer.write_scalar_dict(loss_dict, step, group="Loss", prefix="train-")
                # TODO: add the learning rates to tensorboard/logging
            if step != 0 and self.config.graph.steps_per_save and step % self.config.graph.steps_per_save == 0:
                self.save_checkpoint(self.config.graph.model_dir, step)
            if step % self.config.graph.steps_per_test == 0:
                for image_idx in self.config.data.validation_image_indices:
                    self.test_image(image_idx=image_idx, step=step)
            self.stats.print_stats(i / num_iterations)

        self.stats.update_time(Stats.TOTAL_TRAIN_TIME, train_start, time(), step=-1)
        self.stats.print_stats(-1)

    @profiler.time_function
    def train_iteration(self, batch: dict, step: int):
        """Run one iteration with a batch of inputs."""
        # move batch to correct device
        ray_indices = batch["indices"].to(self.device)
        graph_outputs = self.graph(ray_indices)
        batch["pixels"] = batch["pixels"].to(self.device)
        losses = (
            self.graph.module.get_losses(batch, graph_outputs)
            if hasattr(self.graph, "module")
            else self.graph.get_losses(batch, graph_outputs)
        )
        loss_sum, loss_dict = self.get_aggregated_loss(losses)
        self.optimizers.zero_grad_all()
        loss_sum.backward()
        self.optimizers.scheduler_step_all(step)  # NOTE(ethan): I think the scheduler needs to know what step we are on
        self.optimizers.optimizer_step_all()
        return loss_dict

    @profiler.time_function
    def test_image(self, image_idx, step):
        """Test a specific image."""
        image = self.val_image_dataset[image_idx]["image"]  # ground truth
        image_height, image_width, _ = image.shape
        pixel_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
        pixel_coords = torch.stack(pixel_coords, dim=-1).long()
        all_ray_indices = torch.cat([torch.ones_like(pixel_coords[..., :1]) * image_idx, pixel_coords], dim=-1).view(
            -1, 3
        )
        with torch.no_grad():
            num_rays = all_ray_indices.shape[0]
            chunk_size = 1024
            rgb_coarse = []
            rgb_fine = []
            for i in range(0, num_rays, chunk_size):
                ray_indices = all_ray_indices[i : i + chunk_size].to(self.device)
                graph_outputs = self.graph(ray_indices)
                rgb_coarse.append(graph_outputs["rgb_coarse"])
                rgb_fine.append(graph_outputs["rgb_fine"])
            rgb_coarse = torch.cat(rgb_coarse).view(image_height, image_width, 3).detach().cpu()
            rgb_fine = torch.cat(rgb_fine).view(image_height, image_width, 3).detach().cpu()

        combined_image = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        self.writer.write_image(f"image_idx_{image_idx}-rgb_coarse_fine", combined_image, step, group="val")

        coarse_psnr = get_psnr(image, rgb_coarse)
        self.writer.write_scalar(f"image_idx_{image_idx}-coarse_psnr", float(coarse_psnr), step, group="val")

        fine_psnr = get_psnr(image, rgb_fine)
        self.stats.update_value(Stats.CURR_TEST_PSNR, float(fine_psnr), step)
        self.writer.write_scalar(f"image_idx_{image_idx}-fine_psnr", float(fine_psnr), step, group="val")
