import dataclasses
import os
import typing
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path

import torch


@dataclasses.dataclass
class FeatureDataloader(ABC):
    image_list: torch.Tensor  # (N, 3, H, W)
    device: torch.device
    cfg: dict
    cache_dir: Path
    data_dict: typing.Dict[int, torch.Tensor] = None  # N-length dict of (multi-channel, H_d, W_d)

    @abstractmethod
    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        pass

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def load(self, cache_path):
        pass

    @abstractmethod
    def save(self, cache_path):
        pass

    def try_load(self, cache_path: Path):
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            self.load(cache_path)
        except FileNotFoundError:
            self.create()
            self.save(cache_path)
