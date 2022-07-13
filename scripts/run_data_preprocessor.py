"""
Script to process different dataset types into our format.
"""

# TODO(ethan): we should write a script here to convert from other datasets to our formats
# e.g., Blender, LLFF, ShapeNet, DTU
from hydra import compose, initialize
import json

from pyrad.data.utils import get_dataset_inputs
from pyrad.utils.io import get_absolute_path, make_dir, write_to_pkl
from pyrad.utils.misc import get_hash_str_from_dict


def save_dataset_inputs_to_cache(
    dataset_name="friends_TBBT-big_living_room", splits=["train", "val", "test"], downscale_factors=[1, 2, 4]
):
    """Save the datasets input to cache.
    TODO(ethan): use a serializable representation that's cleaner instead of pickle."""
    with initialize(version_base="1.2", config_path="../configs"):
        config = compose(config_name="default_setup.yaml", overrides=[f"data/dataset={dataset_name}"])
    for downscale_factor in downscale_factors:
        for split in splits:
            config.data.dataset.downscale_factor = downscale_factor
            dataset_config_hash = get_hash_str_from_dict(dict(config.data.dataset))
            dataset_config_hash_filename = make_dir(
                get_absolute_path(f"cache/dataset_inputs/{dataset_config_hash}-{split}.pkl")
            )
            dataset_inputs = get_dataset_inputs(**config.data.dataset, split=split)
            write_to_pkl(dataset_config_hash_filename, dataset_inputs)


if __name__ == "__main__":
    save_dataset_inputs_to_cache()
