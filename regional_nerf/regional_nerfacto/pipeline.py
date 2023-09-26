"""
Nerfstudio RNerf Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from regional_nerfacto.datamanager import RNerfDataManagerConfig
from regional_nerfacto.model import RNerfModel, RNerfModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)


@dataclass
class RNerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: RNerfPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = RNerfDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = RNerfModelConfig()
    """specifies the model config"""


class RNerfPipeline(VanillaPipeline):
    """RNerf Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: RNerfPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.set_enu_transform(self.datamanager.enu2nerf, self.datamanager.nerf2enu, self.datamanager.enu2nerf_points, self.datamanager.nerf2enu_points, self.datamanager.osm_image, self.datamanager.osm_scale)
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                RNerfModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])