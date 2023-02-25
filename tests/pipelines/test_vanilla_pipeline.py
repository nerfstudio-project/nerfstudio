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

        def load_state_dict(self, *args, strict=True):
            """Mocked load_state_dict"""
            nonlocal was_called
            was_called = True
            assert strict

    config = VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=MockedDataManager,
        ),
        model=ModelConfig(_target=MockedModel),
    )
    pipeline = VanillaPipeline(config, "cpu")
    state_dict = pipeline.state_dict()
    pipeline.load_pipeline(state_dict, 0)
    assert was_called
