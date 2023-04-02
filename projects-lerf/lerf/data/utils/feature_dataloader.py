import os
import typing
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import torch

class FeatureDataloader(ABC):
    def __init__(
            self,
            cfg: dict,
            device: torch.device,
            image_list: torch.Tensor, # (N, 3, H, W)
            cache_path: Path,
    ):
        self.cfg = cfg
        self.device = device
        self.cache_path = cache_path
        self.data = None # only expect data to be cached, nothing else
        self.try_load(image_list) # don't save image_list, avoid duplicates

    @abstractmethod
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        pass

    @abstractmethod
    def create(self, image_list: torch.Tensor):
        pass

    def load(self):
        self.data = torch.from_numpy(np.load(self.cache_path))

    def save(self):
        np.save(self.cache_path, self.data)

    def try_load(self, img_list: torch.Tensor):
        try:
            self.load()
        except FileNotFoundError:
            self.create(img_list)
            self.save()