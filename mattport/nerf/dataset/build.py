"""
Builds the dataset
"""
import importlib
import logging

from mattport.nerf.dataset.base import Dataset


def build_dataset(config: dict) -> Dataset:
    """Returns the dataset according to the specified config.
    # TODO(ethan): change the type from dict to be a dataclass

    Args:
        config (dict): _description_
    """

    # TODO(ethan): better handle this with configs
    module = importlib.import_module("mattport.nerf.dataset.blender_dataset")
    dataset = module.BlenderDataset(
        data_directory=config.data_directory, dataset_type=config.dataset_type, scene=config.scene
    )
    logging.info(dataset)
    return dataset
