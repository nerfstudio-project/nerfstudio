"""
The dataset baseclass.
"""

from abc import abstractmethod
import torch

from mattport.nerf.cameras import Camera


class Dataset(torch.utils.data.Dataset):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, data_directory: str, dataset_type: str, scene: str):
        """_summary_

        Args:
            data_directory (str): _description_
            type (str): _description_
            scene (str): _description_
        """
        super().__init__()
        self.data_directory = data_directory
        self.dataset_type = dataset_type
        self.scene = scene

    def __len__(self):
        pass

    @abstractmethod
    def get_image(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def get_camera(self, idx) -> Camera:
        """_summary_

        Args:
            idx (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            Camera: _description_
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
