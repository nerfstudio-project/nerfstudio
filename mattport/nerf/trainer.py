"""
Code to train model.
"""
from omegaconf import DictConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mattport.nerf.dataset.build import build_dataset
from mattport.nerf.graph import Graph
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
        self.train_dataset = build_dataset(self.config.dataset)
        # self.test_dataset = build_dataset(self.config.dataset)

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
        raise NotImplementedError

    def train_iteration(self):
        """_summary_"""
        # TODO(): save checkpoints, and do logging only on rank 0 device only
        # if gpu == 0:
        raise NotImplementedError
