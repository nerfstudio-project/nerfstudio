"""
Code to train model.
"""
import datetime
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

from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.collate import CollateIterDataset, collate_batch_size_one
from mattport.nerf.dataset.utils import get_dataset_inputs
from mattport.nerf.metrics import get_psnr
from mattport.nerf.optimizers import Optimizers
from mattport.utils.decorators import check_main_thread
from mattport.utils.writer import LocalWriter, TensorboardWriter

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class"""

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        # dataset variables
        self.train_image_dataset = None
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        # model variables
        self.graph = None
        self.optimizers = None
        self.start_step = 0
        # logging variables
        self.is_main_thread = local_rank % world_size == 0
        self.local_writer = LocalWriter(local_rank, world_size, save_dir=os.path.join(os.getcwd(), "writer"))
        self.tensorboard_writer = TensorboardWriter(
            local_rank, world_size, save_dir=os.path.join(os.getcwd(), "writer")
        )
        self.time_dict = {}

    def setup_dataset(self):
        """_summary_"""
        dataset_inputs = get_dataset_inputs(**self.config.dataset)
        self.train_image_dataset = ImageDataset(
            image_filenames=dataset_inputs.image_filenames, downscale_factor=dataset_inputs.downscale_factor
        )
        self.train_dataset = CollateIterDataset(
            self.train_image_dataset,
            collate_fn=lambda batch_list: collate_batch(
                batch_list, self.config.dataloader.num_rays_per_batch, keep_full_image=False
            ),
            num_samples_to_collate=self.config.dataloader.num_images_to_sample_from,
            num_times_to_repeat=self.config.dataloader.num_times_to_repeat_images,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=collate_batch_size_one,
            pin_memory=True,
        )
        # TODO(ethan): implement the test dataset

    def setup_graph(self):
        """_summary_"""
        dataset_inputs = get_dataset_inputs(**self.config.dataset)
        self.graph = instantiate(
            self.config.graph, intrinsics=dataset_inputs.intrinsics, camera_to_world=dataset_inputs.camera_to_world
        ).to(f"cuda:{self.local_rank}")
        self.setup_optimizers()

        if self.config.resume_train.load_dir:
            self.load_checkpoint(self.config.resume_train)

        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

    def setup_optimizers(self):
        """_summary_"""
        # TODO(ethan): handle different world sizes
        self.optimizers = Optimizers(self.config.param_groups, self.graph.get_param_groups())

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

    def update_time(self, name, start_time, end_time, num_iter=1, batch_size=None):
        """update the time dictionary with running averages/cumulative durations

        Args:
            name (_type_): name of function time we are logging
            start_time (_type_): start time for the call
            end_time (_type_): end time when the call finished executing
            num_iter (int, optional): number of total iteration steps. Defaults to 1.
            batch_size (_type_, optional): total number of rays in a batch;
                if None, reports duration instead of batch per second. Defaults to None.
        """
        val = end_time - start_time
        if batch_size:
            val = batch_size / val

        self.time_dict[name] = (self.time_dict.get(name, 0) * num_iter + val) / (num_iter + 1)
        if name == "single iter (time)":
            remain_iter = self.config.max_num_iterations - num_iter
            self.time_dict["ETA (time)"] = remain_iter * self.time_dict[name]

    def print_times(self):
        """helper to print out the timing dictionary."""
        mssg = ""
        for k, v in self.time_dict.items():
            if "(time)" in k:
                v = str(datetime.timedelta(seconds=v))
            mssg += f"{k}: {v} | "
        print(mssg, end="\r")

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
        num_iterations = self.config.max_num_iterations
        iter_dataset = iter(self.train_dataloader)
        for step in range(self.start_step, self.start_step + num_iterations):
            data_start = time()
            batch = next(iter_dataset)
            self.update_time("load data (time)", data_start, time(), num_iter=step)
            iter_start = time()
            loss_dict = self.train_iteration(batch, step)
            self.update_time(
                "avg. rays/s (1/s)", iter_start, time(), num_iter=step, batch_size=batch["indices"].shape[0]
            )
            self.update_time("single iter (time)", iter_start, time(), num_iter=step)
            if step != 0 and step % self.config.steps_per_log == 0:
                self.tensorboard_writer.write_scalar_dict(loss_dict, step, group="Loss", prefix="train-")
                logging.info("{%d}/{%d} with losses {%s}", step, num_iterations, loss_dict)
                # TODO: add the learning rates to tensorboard/logging
            if step != 0 and self.config.steps_per_save and step % self.config.steps_per_save == 0:
                self.save_checkpoint(self.config.model_dir, step)
                logging.info("Saved ckpt at {%d}", step)
            if step != 0 and step % self.config.steps_per_test == 0:
                self.test_image(image_idx=0, step=step)
                self.test_image(image_idx=10, step=step)
                self.test_image(image_idx=20, step=step)
            self.print_times()
        self.update_time("total train", train_start, time())

    def train_iteration(self, batch: dict, step: int):
        """Run one iteration with a batch of inputs."""
        # move batch to correct device
        ray_indices = batch["indices"].to(f"cuda:{self.local_rank}")
        graph_outputs = self.graph(ray_indices)
        batch["pixels"] = batch["pixels"].to(f"cuda:{self.local_rank}")
        losses = self.graph.get_losses(batch, graph_outputs)  # or self.graph.module
        loss_sum, loss_dict = self.get_aggregated_loss(losses)
        self.optimizers.zero_grad_all()
        loss_sum.backward()
        self.optimizers.scheduler_step_all(step)  # NOTE(ethan): I think the scheduler needs to know what step we are on
        self.optimizers.optimizer_step_all()
        return loss_dict

    def test_image(self, image_idx, step):
        """Test a specific image."""
        logging.info("Running test iteration.")
        image = self.train_image_dataset[image_idx]["image"]  # ground truth
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
                ray_indices = all_ray_indices[i : i + chunk_size].to(f"cuda:{self.local_rank}")
                graph_outputs = self.graph(ray_indices)
                rgb_coarse.append(graph_outputs["rgb_coarse"])
                rgb_fine.append(graph_outputs["rgb_fine"])
            rgb_coarse = torch.cat(rgb_coarse).view(image_height, image_width, 3).detach().cpu()
            rgb_fine = torch.cat(rgb_fine).view(image_height, image_width, 3).detach().cpu()

        combined_image = torch.cat([image, rgb_coarse, rgb_fine], dim=1)
        self.tensorboard_writer.write_image(
            f"image_idx_{image_idx}-rgb_coarse_fine", combined_image, step, group="test"
        )

        fine_psnr = get_psnr(image, rgb_fine)
        self.tensorboard_writer.write_scalar(f"image_idx_{image_idx}-fine_psnr", fine_psnr, step, group="test")
