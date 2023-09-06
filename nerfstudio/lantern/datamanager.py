"""
Lantern HDR DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager, VanillaDataManagerConfig)
from nerfstudio.lantern.dataset import HDRInputDataset


@dataclass
class HDRVanillaDataManagerConfig(VanillaDataManagerConfig):
    """A HDR data manager, based on VanillaDataManager"""

    _target: Type = field(default_factory=lambda: VanillaDataManager[HDRInputDataset])
    """Target class to instantiate."""
    