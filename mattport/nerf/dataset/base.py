from abc import abstractmethod
import torch


class Dataset(torch.utils.data.Dataset):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(
        self,
        data_directory: str,
    ) -> None:
        super().__init__()
        pass

    def __len__(self):
        pass

    @abstractmethod
    def get_image(self, idx):
        pass

    @abstractmethod
    def get_pose(self, idx):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError
