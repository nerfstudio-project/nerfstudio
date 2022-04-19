"""
Code to train model.
"""
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from mattport.nerf.dataset.image_dataset import ImageDataset, collate_batch
from mattport.nerf.dataset.utils import get_dataset_inputs
from mattport.nerf.graph.base import Graph
from mattport.nerf.optimizer import Optimizer


class Trainer:
    """Training class"""

    def __init__(self, local_rank: int, world_size: int, config: DictConfig):
        self.local_rank = local_rank
        self.world_size = world_size
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
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
        # TODO(ethan): implement test data

    def setup_graph(self):
        """_summary_"""
        self.graph = Graph(self.config.network).to(f"cuda:{self.local_rank}")
        if self.world_size > 1:
            self.graph = DDP(self.graph, device_ids=[self.local_rank])
            dist.barrier(device_ids=[self.local_rank])

    def setup_optimizer(self):
        """_summary_"""
        self.optimizer = Optimizer(params=self.graph.parameters(), **self.config.optimizer)

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
        raise NotImplementedError

    def train_epoch(self):
        """_summary_"""
        for _ in range(len(self.train_dataset)):
            batch = next(iter(self.train_dataloader))
            self.train_iteration(batch)

    def train_iteration(self, batch):
        """_summary_"""
        # TODO(): save checkpoints, and do logging only on rank 0 device only
        # if gpu == 0:
        output_dict = self.graph(batch.indices)
        
        loss = self.graph.get_loss(batch, )

        loss.backward()
        # update model
