"""
For loading the blender dataset format.
"""

from typing import Optional, Union

from omegaconf import ListConfig

from pyrad.nerf.dataset.format.blender import load_blender_data
from pyrad.nerf.dataset.format.friends import load_friends_data
from pyrad.nerf.dataset.format.instant_ngp import load_instant_ngp_data
from pyrad.nerf.dataset.structs import DatasetInputs
from pyrad.structures.colors import get_color
from pyrad.utils.io import get_absolute_path


def get_dataset_inputs(
    data_directory: str,
    dataset_format: str,
    split: str,
    downscale_factor: int = 1,
    alpha_color: Optional[Union[str, list, ListConfig]] = None,
    load_dataset_inputs_from_cache: bool = False,
) -> DatasetInputs:
    """Returns the dataset inputs, which will be used with an ImageDataset and RayGenerator.
    # TODO: implement the `test` split, which will have depths and normals, etc.

    Args:
        data_directory (str): Location of data
        dataset_format (str): Name of dataset type
        downscale_factor (int, optional): How much to downscale images. Defaults to 1.0.
        alpha_color (str, list, optional): Sets transparent regions to specified color, otherwise black.
        load_from_cache (bool)
    Returns:
        DatasetInputs: The inputs needed for generating rays.
    """

    if load_dataset_inputs_from_cache:
        print("Loading from cache.")
        return None

    if alpha_color is not None:
        alpha_color = get_color(alpha_color)

    if dataset_format == "blender":
        dataset_inputs = load_blender_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split=split,
            alpha_color=alpha_color,
        )
    elif dataset_format == "friends":
        # TODO(ethan): change this hack, and do proper splits for the friends dataset
        # currently we assume that there is only the training set of images
        dataset_inputs = load_friends_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split="train",
            include_point_cloud=False,
        )
    elif dataset_format == "instant_ngp":
        dataset_inputs = load_instant_ngp_data(
            get_absolute_path(data_directory),
            downscale_factor=downscale_factor,
            split="train",
        )
    else:
        raise NotImplementedError(f"{dataset_format} is not a valid dataset type")

    return dataset_inputs
