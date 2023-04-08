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
import functools
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Type, cast

import msgpack
import numpy as onp
from typing_extensions import Literal

if TYPE_CHECKING:
    from ._server import ClientId
else:
    ClientId = Any


def _prepare_for_serialization(value: Any) -> Any:
    """Prepare any special types for serialization. Currently just maps numpy arrays to
    their underlying data buffers."""

    if isinstance(value, onp.ndarray):
        return value.data if value.data.c_contiguous else value.copy().data
    return value


class Message:
    """Base message type for controlling our viewer."""

    type: ClassVar[str]
    excluded_self_client: Optional[ClientId] = None
    """Don't send this message to a particular client. Example of when this is useful:
    for synchronizing GUI stuff, we want to """

    def serialize(self) -> bytes:
        """Convert a Python Message object into bytes."""
        mapping = {k: _prepare_for_serialization(v) for k, v in vars(self).items()}
        out = msgpack.packb({"type": self.type, **mapping})
        assert isinstance(out, bytes)
        return out

    @staticmethod
    def deserialize(message: bytes) -> Message:
        """Convert bytes into a Python Message object."""
        mapping = msgpack.unpackb(message)

        # msgpack deserializes to lists by default, but all of our annotations use
        # tuples.
        mapping = {k: tuple(v) if isinstance(v, list) else v for k, v in mapping.items()}
        message_type = Message._subclass_from_type_string()[cast(str, mapping.pop("type"))]
        return message_type(**mapping)

    @staticmethod
    @functools.lru_cache
    def _subclass_from_type_string() -> Dict[str, Type[Message]]:
        subclasses = Message.get_subclasses()
        return {s.type: s for s in subclasses}

    @staticmethod
    def get_subclasses() -> List[Type[Message]]:
        """Recursively get message subclasses."""

        def _get_subclasses(typ: Type[Message]) -> List[Type[Message]]:
            out = []
            for sub in typ.__subclasses__():
                out.append(sub)
                out.extend(_get_subclasses(sub))
            return out

        return _get_subclasses(Message)


@dataclasses.dataclass
class BackgroundImageMessage(Message):
    """Message for rendering a background image."""

    type: ClassVar[str] = "background_image"
    media_type: Literal["image/jpeg", "image/png"]
    base64_data: str


@dataclasses.dataclass
class GuiAddMessage(Message):
    """Sent server->client to add a new GUI input."""

    type: ClassVar[str] = "add_gui"
    name: str
    folder_labels: Tuple[str]
    leva_conf: Any


@dataclasses.dataclass
class GuiRemoveMessage(Message):
    """Sent server->client to add a new GUI input."""

    type: ClassVar[str] = "remove_gui"
    name: str


@dataclasses.dataclass
class GuiUpdateMessage(Message):
    """Sent client->server when a GUI input is changed."""

    type: ClassVar[str] = "gui_update"
    name: str
    value: Any


@dataclasses.dataclass
class GuiSetHiddenMessage(Message):
    """Sent client->server when a GUI input is changed."""

    type: ClassVar[str] = "gui_set_hidden"
    name: str
    hidden: bool


@dataclasses.dataclass
class GuiSetValueMessage(Message):
    """Sent server->client to set the value of a particular input."""

    type: ClassVar[str] = "gui_set"
    name: str
    value: Any


@dataclasses.dataclass
class GuiSetLevaConfMessage(Message):
    """Sent server->client to override some part of an input's Leva config."""

    type: ClassVar[str] = "gui_set_leva_conf"
    name: str
    leva_conf: Any


@dataclasses.dataclass
class ResetSceneMessage(Message):
    """Reset scene."""

    type: ClassVar[str] = "reset_scene"


@dataclasses.dataclass
class FilePathInfoMessage(Message):
    """Experiment file path info"""

    type: ClassVar[str] = "path_info"
    config_base_dir: str
    """ Base directory for config files """
    data_base_dir: str
    """ Base directory for data files """
    export_path_name: str
    """ Name of the export folder """


@dataclasses.dataclass
class CameraMessage(Message):
    """Render camera data."""

    type: ClassVar[str] = "camera"
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
class SceneBoxMessage(Message):
    """Scene Box data."""

    type: ClassVar[str] = "scene_box"
    min: Tuple[float, float, float]
    """ Minimum coordinates of the scene box """
    max: Tuple[float, float, float]
    """ Maximum coordinates of the scene box """


@dataclasses.dataclass
class DatasetImageMessage(Message):
    """Message for rendering a dataset image frustum."""

    type: ClassVar[str] = "dataset_image"
    idx: str
    """Index of the image in the threejs scene"""
    json: Any
    """JSON computed by the camera class"""


@dataclasses.dataclass
class IsTrainingMessage(Message):
    """Wheather the scene is in training mode or not."""

    type: ClassVar[str] = "is_training"
    is_training: bool
    """True if the model is currently trianing, False otherwise"""


@dataclasses.dataclass
class CameraPathPayloadMessage(Message):
    """Camera path"""

    type: ClassVar[str] = "camera_path_payload"
    camera_path_filename: str
    """ Camera path filename """
    camera_path: Any
    """ Camera path data """


@dataclasses.dataclass
class CameraPathOptionsRequest(Message):
    """Request list of existing camera paths"""

    type: ClassVar[str] = "camera_path_options"


@dataclasses.dataclass
class CameraPathsMessage(Message):
    """Dictionary of camera paths"""

    type: ClassVar[str] = "camera_paths"
    payload: Any
    """ Dictionary of camera paths """
