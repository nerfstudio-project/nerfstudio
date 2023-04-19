# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Message type definitions. For synchronization with the TypeScript definitions, see
`_typescript_interface_gen.py.`"""

from __future__ import annotations

import dataclasses
from typing import Any, Tuple

import viser.infra
from typing_extensions import Literal, override


class NerfstudioMessage(viser.infra.Message):
    """Base message type for controlling our viewer."""

    @override
    def redundancy_key(self) -> str:
        return type(self).__name__


@dataclasses.dataclass
class BackgroundImageMessage(NerfstudioMessage):
    """Message for rendering a background image."""

    media_type: Literal["image/jpeg", "image/png"]
    base64_data: str


@dataclasses.dataclass
class GuiAddMessage(NerfstudioMessage):
    """Sent server->client to add a new GUI input."""

    name: str
    folder_labels: Tuple[str]
    leva_conf: Any

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.name}"


@dataclasses.dataclass
class GuiRemoveMessage(NerfstudioMessage):
    """Sent server->client to add a new GUI input."""

    name: str

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.name}"


@dataclasses.dataclass
class GuiUpdateMessage(NerfstudioMessage):
    """Sent client->server when a GUI input is changed."""

    name: str
    value: Any

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.name}"


@dataclasses.dataclass
class GuiSetHiddenMessage(NerfstudioMessage):
    """Sent client->server when a GUI input is changed."""

    name: str
    hidden: bool

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.name}"


@dataclasses.dataclass
class GuiSetValueMessage(NerfstudioMessage):
    """Sent server->client to set the value of a particular input."""

    name: str
    value: Any

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.name}"


@dataclasses.dataclass
class GuiSetLevaConfMessage(NerfstudioMessage):
    """Sent server->client to override some part of an input's Leva config."""

    name: str
    leva_conf: Any

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.name}"


@dataclasses.dataclass
class FilePathInfoMessage(NerfstudioMessage):
    """Experiment file path info"""

    config_base_dir: str
    """ Base directory for config files """
    data_base_dir: str
    """ Base directory for data files """
    export_path_name: str
    """ Name of the export folder """


@dataclasses.dataclass
class CameraMessage(NerfstudioMessage):
    """Render camera data."""

    aspect: float
    """ Aspect ratio of the camera """
    render_aspect: float
    """ Aspect ratio of the render window """
    fov: float
    """ Field of view of the camera """
    matrix: Tuple[
        float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float
    ]
    """ Camera matrix """
    camera_type: Literal["perspective", "fisheye", "equirectangular"]
    """ Camera type """
    is_moving: bool
    """ True if the camera is moving, False otherwise """
    timestamp: int
    """JSON computed by the camera class"""


@dataclasses.dataclass
class SceneBoxMessage(NerfstudioMessage):
    """Scene Box data."""

    min: Tuple[float, float, float]
    """ Minimum coordinates of the scene box """
    max: Tuple[float, float, float]
    """ Maximum coordinates of the scene box """


@dataclasses.dataclass
class DatasetImageMessage(NerfstudioMessage):
    """Message for rendering a dataset image frustum."""

    idx: str
    """Index of the image in the threejs scene"""
    json: Any
    """JSON computed by the camera class"""

    @override
    def redundancy_key(self) -> str:
        return f"{type(self).__name__}_{self.idx}"


@dataclasses.dataclass
class TrainingStateMessage(NerfstudioMessage):
    """Wheather the scene is in training mode or not."""

    training_state: Literal["training", "paused", "completed"]
    """True if the model is currently trianing, False otherwise"""


@dataclasses.dataclass
class CameraPathPayloadMessage(NerfstudioMessage):
    """Camera path"""

    camera_path_filename: str
    """ Camera path filename """
    camera_path: Any
    """ Camera path data """


@dataclasses.dataclass
class CameraPathOptionsRequest(NerfstudioMessage):
    """Request list of existing camera paths"""


@dataclasses.dataclass
class CameraPathsMessage(NerfstudioMessage):
    """Dictionary of camera paths"""

    payload: Any
    """ Dictionary of camera paths """


@dataclasses.dataclass
class CropParamsMessage(NerfstudioMessage):
    """Crop parameters"""

    crop_enabled: bool
    """ Crop parameters """
    crop_bg_color: Tuple[int, int, int]
    """ Crop background color, range 0-255 """
    crop_center: Tuple[float, float, float]
    """ Center of the crop box """
    crop_scale: Tuple[float, float, float]
    """ Scale of the crop box """


@dataclasses.dataclass
class StatusMessage(NerfstudioMessage):
    """Status message."""

    eval_res: str
    """ Resolution of the viewer display in plain text """
    step: int
    """ Current step """


@dataclasses.dataclass
class SaveCheckpointMessage(NerfstudioMessage):
    """Save checkpoint message."""


@dataclasses.dataclass
class UseTimeConditioningMessage(NerfstudioMessage):
    """Use time conditioning message."""


@dataclasses.dataclass
class TimeConditionMessage(NerfstudioMessage):
    """Time conditioning message."""

    time: float
    """ Time conditioning value """
