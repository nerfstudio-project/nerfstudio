"""
Code to train model.
"""
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig

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

    def train(self) -> None:
        """_summary_"""
        raise NotImplementedError

    def train_epoch(self):
        """_summary_"""
        raise NotImplementedError

    def train_iteration(self):
        """_summary_"""
        # TODO(): save checkpoints, and do logging only on rank 0 device only
        # if gpu == 0:
        raise NotImplementedError
