"""
Code to train model.
"""
import logging
import os

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import get_dataset_inputs
from mattport.nerf.optimizers import Optimizers
from mattport.utils.decorators import check_main_thread
from mattport.utils.writer import LocalWriter

logging.getLogger("PIL").setLevel(logging.WARNING)


class Trainer:
    """Training class"""

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        # dataset variables
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
        self.local_writer = LocalWriter(local_rank, world_size, save_dir=None)

    def setup_dataset(self):
        """_summary_"""
        dataset_inputs = get_dataset_inputs(**self.config.dataset)
        self.train_dataset = ImageDataset(
            image_filenames=dataset_inputs.image_filenames, downscale_factor=dataset_inputs.downscale_factor
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.num_images_to_sample_from,
            collate_fn=lambda batch: collate_batch(
                batch, self.config.dataloader.num_rays_per_batch, keep_full_image=True
            ),
            num_workers=self.config.dataloader.num_workers,
            shuffle=True,
        )
        # TODO(ethan): implement the test data

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

    def train(self) -> None:
        """_summary_"""
        num_iterations = 1000
        for step in range(self.start_step, self.start_step + num_iterations):
            batch = next(iter(self.train_dataloader))
            losses = self.train_iteration(step, batch)
            self.local_writer.write_text(f"{step}/{num_iterations}: losses", losses)
            if step % self.config.save_step == 0:
                self.save_checkpoint(self.config.model_dir, step)
                self.local_writer.write_text(f"ckpt at {step}", f"{losses}\n")

    def train_iteration(self, step: int, batch: dict):
        """Run one iteration with a batch of inputs."""
        # move batch to correct device
        ray_indices = batch.indices.to(f"cuda:{self.local_rank}")
        graph_outputs = self.graph(ray_indices)
        batch.pixels = batch.pixels.to(f"cuda:{self.local_rank}")
        losses = self.graph.module.get_losses(batch, graph_outputs)
        return losses

    def test(self, image_idx):
        """Test a specific image."""
        with torch.no_grad():
            ray_indices = None
            