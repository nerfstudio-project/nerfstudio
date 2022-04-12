"""
The dataset baseclass.
"""

from abc import abstractmethod
import torch
import imageio

from mattport.structures.cameras import Camera


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

        # data to populate
        self.image_filenames = []
        self.cameras = []

    def __len__(self):
        return len(self.image_filenames)

    def get_image(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        image_filename = self.image_filenames[idx]
        image = imageio.imread(image_filename)
        return image

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
        return self.cameras[idx]

    def __getitem__(self, idx):
        """Returns the rays for the camera.

        Args:
            idx (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        # raise NotImplementedError

        # TODO(ethan): we should have a dataset which always takes rays from the same camera
        # or, it could take rays from multiple cameras

        camera = self.cameras[idx]
        rays = camera.generate_all_rays()  # (H, W, 6)
        camera_indices = torch.ones_like(rays[:, :, 0:1]) * idx
        rgbs = self.get_image(idx) / 255.0

        return {
            "rays": rays,
            "rgbs": rgbs,
            "camera_indices": camera_indices
            # question: do we want to have more information here?
        }


# question... what do I want the dataset to return?
