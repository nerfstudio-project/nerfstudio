"""
Code to train model.
"""

import logging
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import get_dataset_inputs


class Trainer:
    """Training class"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, config: DictConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.graph = None
        self.optimizer = None

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
        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

    def setup_optimizer(self):
        """_summary_"""
        self.optimizer = optim.Adam(self.graph.parameters(), lr=self.config.optimizer.lr)

    def load_checkpoint(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def save_checkpoint(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def train(self) -> None:
        """_summary_"""
        # is_main_thread = self.local_rank % self.world_size == 0
        # raise NotImplementedError
        num_epochs = 10
        for _ in range(num_epochs):
            self.train_epoch()

    def train_epoch(self):
        """_summary_"""
        num_iters = 100
        for _ in tqdm(range(num_iters)):
            batch = next(iter(self.train_dataloader))
            self.train_iteration(batch)

    def train_iteration(self, batch):
        """_summary_"""
        # TODO(): save checkpoints, and do logging only on rank 0 device only
        if self.local_rank == 0:
            pass

        # move batch to correct device
        ray_indices = batch.indices.to(f"cuda:{self.local_rank}")
        graph_outputs = self.graph(ray_indices)
        batch.pixels = batch.pixels.to(f"cuda:{self.local_rank}")
        losses = self.graph.get_losses(batch, graph_outputs)
        logging.info(losses)
        