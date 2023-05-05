# pylint: disable=all
from copy import copy
from typing import Any

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataparserOutputs,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.depth_dataset import DepthDataset


def test_data_manager_type_inference():
    # Mock for a faster test
    class DummyDataParser:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, __name: str) -> Any:
            return None

        def get_dataparser_outputs(self, *args, **kwargs):
            return DataparserOutputs(
                [],
                Cameras(
                    torch.ones((0, 4, 4)),
                    torch.ones((0, 1)),
                    torch.ones((0, 1)),
                    torch.ones((0, 1)),
                    torch.ones((0, 1)),
                    10,
                    10,
                ),
                metadata={"depth_filenames": [], "depth_unit_scale_factor": 1.0},
            )

    config = VanillaDataManagerConfig()
    setattr(config, "dataparser", InstantiateConfig(_target=DummyDataParser))
    setattr(config.dataparser, "data", None)

    assert VanillaDataManager(config).dataset_type is InputDataset
    assert VanillaDataManager[DepthDataset](config).dataset_type is DepthDataset

    class tmp(VanillaDataManager):
        pass

    assert tmp(config).dataset_type is InputDataset

    class tmp2(VanillaDataManager[DepthDataset]):
        pass

    assert tmp2(config).dataset_type is DepthDataset
