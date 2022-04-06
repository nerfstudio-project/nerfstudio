from argon2 import DEFAULT_RANDOM_SALT_LENGTH
from mattport.nerf.dataset.base import Dataset
import importlib
import logging

def build_dataset(config: dict) -> Dataset:
    """Returns the dataset according to the specified config.
    # TODO(ethan): change the type from dict to be a dataclass

    Args:
        config (dict): _description_
    """

    module = importlib.import_module("mattport.nerf.dataset.base")
    dataset = module.Dataset(**config)
    logging.info(dataset)
    return dataset