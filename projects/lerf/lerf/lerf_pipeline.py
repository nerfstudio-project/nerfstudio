import typing
from dataclasses import dataclass, field
from typing import Literal, Type

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.configs import base_config as cfg
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)

from lerf.data.lerf_datamanager import (
    LERFDataManager,
    LERFDataManagerConfig,
)
from lerf.lerf import LERFModel, LERFModelConfig
from lerf.encoders.image_encoder import BaseImageEncoderConfig, BaseImageEncoder


@dataclass
class LERFPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: LERFPipeline)
    """target class to instantiate"""
    datamanager: LERFDataManagerConfig = LERFDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = LERFModelConfig()
    """specifies the model config"""
    network: BaseImageEncoderConfig = BaseImageEncoderConfig()
    """specifies the vision-language network config"""


class LERFPipeline(VanillaPipeline):
    def __init__(
        self,
        config: LERFPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.image_encoder: BaseImageEncoder = config.network.setup()

        self.datamanager: LERFDataManager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            image_encoder=self.image_encoder,
        )
        self.datamanager.to(device)

        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            image_encoder=self.image_encoder,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(LERFModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])
