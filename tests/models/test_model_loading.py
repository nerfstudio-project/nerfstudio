"""
Test model
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
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig


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


def test_load_model():
    """Test that a NeRF Model can be exported and read correctly"""

    config = VanillaModelConfig(_target=NeRFModel)
    model = NeRFModel(config, scene_box=SceneBox(), num_train_data=1)
    model_state_dict = model.state_dict()
    # Model weights are set to a known fixed value to compare the result of the model read
    model_state_dict[  # pylint: disable=unsubscriptable-object,unsupported-assignment-operation
        "lpips.net.lin0.model.1.weight"
    ][:] = torch.Tensor([[[[1]] for _ in range(64)]])

    # A new model is created with known weights of different value than the pipeline
    model2 = NeRFModel(config, scene_box=SceneBox(), num_train_data=1)
    model2_state_dict = model2.state_dict()
    content = [[[2]] for _ in range(64)]
    model2_state_dict[  # pylint: disable=unsubscriptable-object,unsupported-assignment-operation
        "lpips.net.lin0.model.1.weight"
    ][:] = torch.Tensor([content])

    assert not torch.equal(
        model_state_dict["lpips.net.lin0.model.1.weight"],  # pylint: disable=unsubscriptable-object
        model2_state_dict["lpips.net.lin0.model.1.weight"],  # pylint: disable=unsubscriptable-object
    )

    # After loading the model, the model and pipeline weights should be the same
    model.load_model_state(model2_state_dict, 0)
    model_state_dict = model.state_dict()

    assert torch.equal(
        model_state_dict["lpips.net.lin0.model.1.weight"],  # pylint: disable=unsubscriptable-object
        model2_state_dict["lpips.net.lin0.model.1.weight"],  # pylint: disable=unsubscriptable-object
    )


def test_dpp_detection():
    """Test that DPP models are correctly detected"""
    config = VanillaModelConfig(_target=NeRFModel)
    model = NeRFModel(config, scene_box=SceneBox(), num_train_data=1)
    state_dict = model.state_dict()
    assert Model.is_dpp(state_dict) is False

    ddp_state_dict = {}
    for key, value in state_dict.items():
        # replace the prefix "_model" with "_model.module"
        ddp_state_dict["module." + key[len("_model.") :]] = value
    assert Model.is_dpp(ddp_state_dict)
