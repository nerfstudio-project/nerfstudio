"""
For loading the blender dataset format.
"""

from typing import Dict, Optional, Tuple, Union

from omegaconf import ListConfig

from mattport.nerf.dataset.blender import load_blender_data
from mattport.nerf.dataset.friends import load_friends_data
from mattport.nerf.dataset.structs import DatasetInputs
from mattport.structures.colors import get_color
from mattport.utils.io import get_absolute_path


def get_dataset_inputs_dict(
    data_directory: str,
    dataset_type: str,
    downscale_factor: int = 1,
    alpha_color: Optional[Union[str, list, ListConfig]] = None,
    splits: Tuple[str] = ("train", "val", "test"),
) -> Dict[str, DatasetInputs]:
    """Returns the dataset inputs, which will be used with an ImageDataset and RayGenerator.
    # TODO: implement the `test` split, which will have depths and normals, etc.

    Args:
        data_directory (str): Location of data
        dataset_type (str): Name of dataset type
        downscale_factor (int, optional): How much to downscale images. Defaults to 1.0.
        alpha_color (str, list, optional): Sets transparent regions to specified color, otherwise black.

    Returns:
        Dict[str, DatasetInputs]: The inputs needed for generating rays.
    """
    dataset_inputs_dict = {}

    if alpha_color is not None:
        alpha_color = get_color(alpha_color)

    if dataset_type == "blender":
        for split in splits:
            dataset_inputs = load_blender_data(
                get_absolute_path(data_directory),
                downscale_factor=downscale_factor,
                split=split,
                alpha_color=alpha_color,
            )
            dataset_inputs_dict[split] = dataset_inputs
    elif dataset_type == "friends":
        for split in splits:
            dataset_inputs = load_friends_data(
                get_absolute_path(data_directory), downscale_factor=downscale_factor, split=split
            )
            dataset_inputs_dict[split] = dataset_inputs
    else:
        raise NotImplementedError(f"{dataset_type} is not a valid dataset type")

    return dataset_inputs_dict
