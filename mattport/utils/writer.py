"""
Generic Writer class
"""


from abc import abstractmethod

from torch.utils.tensorboard import SummaryWriter


class Writer:
    """Writer class"""

    def __init__(self, local_rank: int, world_size: int, save_dir: str):
        self.save_dir = save_dir
        self.writer = None
        self.is_main_thread = local_rank % world_size == 0

    @abstractmethod
    def write_text(self, name: str, x: str) -> None:
        """Required method to write a string to summary

        Args:
            name (str): data identifier
            x (str): string to write
        """
        raise NotImplementedError

    @abstractmethod
    def write_scalar(self, name: str, x: float, y: float) -> None:
        """Required method to write a single scalar value to the logger

        Args:
            name (str): name of chart to which you log (x,y)
            x (float): x value to write
            y (float): y value to write
        """
        raise NotImplementedError

    def write_scalar_dict(self, scalar_dict: dict) -> None:
        """Function that writes out all scalars from a given dictionary to the logger

        Args:
            scalar_dict (dict): dictionary containing all scalar values
        """
        if self.is_main_thread:
            for name, (x, y) in scalar_dict.items():
                self.writer.write_scalar(name, x, y)


class TensorboardWriter(Writer):
    """Tensorboard Writer Class"""

    def __init__(self, local_rank: int, world_size: int, save_dir: str):
        super().__init__(local_rank, world_size, save_dir)
        if self.is_main_thread:
            self.writer = SummaryWriter(log_dir=self.save_dir)

    def write_text(self, name: str, x: str) -> None:
        """Tensorboard method to write a string to summary

        Args:
            name (str): data identifier
            x (str): string to write
        """
        if self.is_main_thread:
            self.writer.add_text(name, x)

    @abstractmethod
    def write_scalar(self, name: str, x: float, y: float) -> None:
        """Tensorboard method to write a single scalar value to the logger

        Args:
            name (str): name of chart to which you log (x,y)
            x (float): x value to write
            y (float): y value to write
        """
        if self.is_main_thread:
            self.writer.add_scalar(name, x, y)
