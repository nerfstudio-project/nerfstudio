from abc import abstractmethod
import logging
import torch

from mattport.nerf.optimizer import Optimizer
from mattport.nerf.graph import Graph
from mattport.nerf.dataset.build import build_dataset


class Trainer(object):
    def __init__(self, config: dict):
        self.config = config

    def setup_dataset(self):
        """_summary_
        """
        self.train_dataset = build_dataset(self.config.dataset)
        self.test_dataset = build_dataset(self.config.dataset)

    def setup_graph(self):
        """_summary_
        """
        self.graph = Graph(
            encodings_config=self.config.encodings,
            renderer_config=self.config.renderer,
            loss_config=self.config.loss
        )

    def setup_optimizer(self):
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

    def train(self):
        """_summary_"""
        pass

    def train_epoch(self):
        """_summary_"""
        pass

    def train_iteration(self):
        """_summary_"""
        pass
