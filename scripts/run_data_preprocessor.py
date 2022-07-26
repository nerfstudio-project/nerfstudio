"""
Script to process different dataset types into our format.
"""

# TODO(ethan): we should write a script here to convert from other datasets to our formats
# e.g., Blender, LLFF, ShapeNet, DTU
from typing import List

from hydra import compose, initialize

from nerfactory.data.utils import get_dataset_inputs
from nerfactory.utils.io import get_absolute_path, make_dir, write_to_pkl
from nerfactory.utils.misc import get_hash_str_from_dict


def save_dataset_inputs_to_cache(
    dataset_name: str = "friends_TBBT-big_living_room", splits: List = None, downscale_factors: List = None
):
    """Save the datasets input to cache.
    TODO(ethan): use a serializable representation that's cleaner instead of pickle.

    Args:
        dataset_name (str): name of the dataset
        splits (List[str]): list of splits to save. Defaults to ["train", "val", "test"]
        downscale_factors (List[int]): list of downscale factors to save. Defaults to [1, 2, 4]
    """
    if splits is None:
        splits = ["train", "val", "test"]
    if downscale_factors is None:
        downscale_factors = [1, 2, 4]
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
