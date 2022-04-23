"""
Generic Writer class
"""


import os
from abc import abstractmethod
from typing import Dict

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchtyping import TensorType

from mattport.utils.decorators import check_main_thread

to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)


class Writer:
    """Writer class"""

    def __init__(self, local_rank: int, world_size: int, save_dir: str):
        self.save_dir = save_dir
        self.is_main_thread = local_rank % world_size == 0

    @abstractmethod
    def write_image(
        self, name: str, x: TensorType["H", "W", 3], step: int, group: str = None, prefix: str = None
    ) -> None:
        """_summary_

        Args:
            name (str): data identifier
            x (TensorType["H", "W", 3]): rendered image to write
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        raise NotImplementedError

    @check_main_thread
    def write_text(self, name: str, x: str, step: int, group: str = None, prefix: str = None) -> None:
        """Required method to write a string to summary

        Args:
            name (str): data identifier
            x (str): string to write
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(self, name: str, scalar: float, step: int, group: str = None, prefix: str = None) -> None:
        """Required method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        raise NotImplementedError

    @check_main_thread
    def write_scalar_dict(
        self, scalar_dict: Dict[str, float], step: int, group: str = None, prefix: str = None
    ) -> None:
        """Function that writes out all scalars from a given dictionary to the logger

        Args:
            scalar_dict (dict): dictionary containing all scalar values with key names and quantities
            step (int): the time step to log
            group (str): the group e.g., "Loss", "Accuracy", "Time"
            prefix (str): the prefix e.g., "train-", "test-"
        """
        if self.is_main_thread:
            for name, scalar in scalar_dict.items():
                self.write_scalar(name, scalar, step, group=group, prefix=prefix)


class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, local_rank: int, world_size: int, save_dir: str):
        super().__init__(local_rank, world_size, save_dir)
        if self.is_main_thread:
            self.tb_writer = SummaryWriter(log_dir=self.save_dir)

    @check_main_thread
    def write_image(
        self, name: str, x: TensorType["H", "W", 3], step: int, group: str = None, prefix: str = None
    ) -> None:
        """_summary_

        Args:
            name (str): data identifier
            x (TensorType["H", "W", 3]): rendered image to write
        """
        x = to8b(x)
        self.tb_writer.add_image(name, x, step, dataformats="HWC")

    @check_main_thread
    def write_scalar(self, name: str, scalar: float, step: int, group: str = None, prefix: str = None) -> None:
        """Tensorboard method to write a single scalar value to the logger

        Args:
            name (str): data identifier
            x (float): x value to write
            y (float): y value to write
            group (str)): a prefix to group tensorboard scalars
        """
        group_s = f"{group}/" if group else ""
        prefix_s = f"{prefix}" if prefix else ""
        self.tb_writer.add_scalar(f"{group_s}{prefix_s}{name}", scalar, step)


class LocalWriter(Writer):
    """Local Writer Class"""

    @check_main_thread
    def write_image(
        self, name: str, x: TensorType["H", "W", 3], step: int, group: str = None, prefix: str = None
    ) -> None:
        x = to8b(x)
        image_path = os.path.join(self.save_dir, f"{name}.jpg")
        imageio.imwrite(image_path, np.uint8(x.cpu().numpy() * 255.0))

    @check_main_thread
    def write_text(self, name: str, x: str, step: int, group: str = None, prefix: str = None) -> None:
        """Logs text locally to the terminal in place
        Args:
            name (str): data identifier
            x (str): string to write
        """
        print(f"{name}: {x}, step: {step}", end="\r")

    @check_main_thread
    def write_scalar(self, name: str, scalar: float, step: int, group: str = None, prefix: str = None) -> None:
        self.write_text(name, f"scalar: {scalar:04f}", step)
