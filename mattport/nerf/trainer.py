"""
Code to train model.
"""
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from mattport.nerf.dataset.build import build_dataset
from mattport.nerf.optimizer import Optimizer


class Trainer:
    """Training class"""

    def __init__(self, config: dict):
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
        raise NotImplementedError

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

    def train(self, gpu: int) -> None:
        """_summary_"""
        self.setup_ddp(gpu, self.config.world_size)

        # TODO(): implement training pipeline. below I add skeleton code for compatibility with DDP
        # self.setup_graph()
        # self.graph = DDP(self.graph, device_ids=[gpu])
        # self.setup_dataset()
        # train_sampler = DistributedSampler(self.dataset, rank=gpu, num_replicas=self.config.world_size, shuffle=True)
        # TODO(): create dataloader that takes as input the train_sampler
        # self.setup_optimizer()

        # TODO(): calls to train_epoch and train_iteration loop

        dist.destroy_process_group()
        raise NotImplementedError

    def train_epoch(self, gpu: int):
        """_summary_"""
        raise NotImplementedError

    def train_iteration(self, gpu: int):
        """_summary_"""
        # TODO(): save checkpoints, and do logging only on rank 0 device only
        # if gpu == 0:
        raise NotImplementedError

    def setup_ddp(self, gpu: int, world_size: int) -> None:
        """Initialization of processes for distributed training.

        Args:
            gpu (int): process rank number
            world_size (int): total number of gpus available for training
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=world_size)
        torch.cuda.set_device(gpu)

        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

    def run_multiprocess_train(self) -> None:
        """function to spawn multiple processes for training."""
        if self.config.world_size < 1:
            self.config.world_size = torch.cuda.device_count()
        mp.spawn(self.train, nprocs=self.config.world_size)
