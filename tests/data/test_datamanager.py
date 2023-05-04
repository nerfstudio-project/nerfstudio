# pylint: disable=all
from copy import copy

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    VanillaDataManager,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datasets.depth_dataset import DepthDataset


def test_data_manager_type_inference():
    # Mock for a faster test
    _VanillaDataManager = copy(VanillaDataManager)
    _VanillaDataManager.__init__ = lambda self: DataManager.__init__(self)

    assert _VanillaDataManager().get_dataset_type() is InputDataset
    assert _VanillaDataManager[DepthDataset]().get_dataset_type() is DepthDataset

    class tmp(_VanillaDataManager):
        pass

    assert tmp().get_dataset_type() is InputDataset

    class tmp2(_VanillaDataManager[DepthDataset]):
        pass

    assert tmp2().get_dataset_type() is DepthDataset
