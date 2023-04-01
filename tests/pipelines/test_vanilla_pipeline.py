"""
Test pipeline
"""
from pathlib import Path

# pylint: disable=too-few-public-methods
# pylint: disable=no-self-use
# pylint: disable=missing-class-docstring
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import torch
from torch import nn

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import DataparserOutputs, InputDataset
from nerfstudio.pipelines.base_pipeline import (
    Model,
    ModelConfig,
    VanillaDataManagerConfig,
    VanillaPipeline,
    VanillaPipelineConfig,
)


class MockedDataManager:
    """Mocked data manager"""

    def __init__(self, *args, **kwargs):
        num_images = 0
        self.train_dataset = InputDataset(
            DataparserOutputs(
                image_filenames=[Path("filename.png")] * num_images,
                cameras=Cameras(
                    camera_to_worlds=torch.ones([num_images, 3, 4], dtype=torch.float32),
                    fx=1.0,
                    fy=1.0,
                    cx=1.0,
                    cy=1.0,
                    width=2,
                    height=2,
                ),
            )
        )

    def to(self, *args, **kwargs):
        "Mocked to"
        return self


def test_load_state_dict():
    """Test pipeline load_state_dict calls model's load_state_dict"""
    was_called = False

    class MockedModel(Model):  #
        """Mocked model"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_parameter("param", nn.Parameter(torch.ones((3,))))
            self.register_module(
                "module", Model(*args, **kwargs)
            )  # make sure that non-DDP checkpoint won't be processed as DDP one

        def load_state_dict(self, state_dict, strict=True):
            """Mocked load_state_dict"""
            nonlocal was_called
            was_called = True
            assert strict
            assert "param" in state_dict
            assert "module.device_indicator_param" in state_dict

    config = VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=MockedDataManager,
        ),
        model=ModelConfig(_target=MockedModel),
    )
    pipeline = VanillaPipeline(config, "cpu")
    state_dict = pipeline.state_dict()
    state_dict["_model.param"].mul_(2)  # pylint: disable=unsubscriptable-object
    pipeline.load_pipeline(state_dict, 0)
    assert was_called
    assert pipeline.model.param[0].item() == 2

    # preparation for another test
    state_dict["_model.param"].mul_(2)  # pylint: disable=unsubscriptable-object
    was_called = False
    # pretends to be a DDP checkpoint
    ddp_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_model."):
            # replace the prefix "_model" with "_model.module"
            ddp_state_dict["_model.module." + key[len("_model.") :]] = value
        else:
            ddp_state_dict[key] = value
    # load DDP checkpoint
    pipeline.load_pipeline(ddp_state_dict, 0)
    assert was_called
    assert pipeline.model.param[0].item() == 4
