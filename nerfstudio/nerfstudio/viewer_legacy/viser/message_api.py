# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""This module contains the MessageApi class, which is the interface for sending messages to the Viewer"""

from __future__ import annotations

import abc
import base64
import contextlib
import io
import time
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    cast,
    overload,
)

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
from typing_extensions import LiteralString, ParamSpec, assert_never

from nerfstudio.data.scene_box import SceneBox

from . import messages
from .gui import GuiHandle, GuiSelectHandle, _GuiHandleState

if TYPE_CHECKING:
    from viser.infra import ClientId


P = ParamSpec("P")


def _colors_to_uint8(colors: onp.ndarray) -> onpt.NDArray[onp.uint8]:
    """Convert intensity values to uint8. We assume the range [0,1] for floats, and
    [0,255] for integers."""
    if colors.dtype != onp.uint8:
        if onp.issubdtype(colors.dtype, onp.floating):
            colors = onp.clip(colors * 255.0, 0, 255).astype(onp.uint8)
        if onp.issubdtype(colors.dtype, onp.integer):
            colors = onp.clip(colors, 0, 255).astype(onp.uint8)
    return colors


def _encode_image_base64(
    image: onp.ndarray,
    file_format: Literal["png", "jpeg"],
    quality: Optional[int] = None,
) -> Tuple[Literal["image/png", "image/jpeg"], str]:
    """Encode an image as a base64 string.

    Args:
        image: The image to encode.
        file_format: The format to encode the image as.
        quality: The quality to encode the image as. Only used for JPEG.

    Returns:
        A tuple of the media type and the base64-encoded image.
    """

    media_type: Literal["image/png", "image/jpeg"]
    image = _colors_to_uint8(image)
    with io.BytesIO() as data_buffer:
        if file_format == "png":
            media_type = "image/png"
            iio.imwrite(data_buffer, image, extension=".png")
        elif file_format == "jpeg":
            media_type = "image/jpeg"
            iio.imwrite(
                data_buffer,
                image[..., :3],  # Strip alpha.
                extension=".jpeg",
                quality=75 if quality is None else quality,
            )
        else:
            assert_never(file_format)

        base64_data = base64.b64encode(data_buffer.getvalue()).decode("ascii")

    return media_type, base64_data


TVector = TypeVar("TVector", bound=tuple)


def _cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if isinstance(vector, tuple):
        assert len(vector) == length
        return cast(TVector, vector)
    assert cast(onp.ndarray, vector).shape == (length,)
    return cast(TVector, tuple(map(float, vector)))


T = TypeVar("T")
IntOrFloat = TypeVar("IntOrFloat", int, float)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)


class MessageApi(abc.ABC):
    """Interface for all commands we can use to send messages over a websocket connection.

    Should be implemented by both our global server object (for broadcasting) and by
    individual clients."""

    def __init__(self) -> None:
        self._handle_state_from_gui_name: Dict[str, _GuiHandleState[Any]] = {}
        self._gui_folder_labels: List[str] = []

    @abc.abstractmethod
    def _queue(self, message: messages.NerfstudioMessage) -> None:
        """Abstract method for sending messages."""
        ...

    @contextlib.contextmanager
    def gui_folder(self, label: str) -> Generator[None, None, None]:
        """Context for placing all GUI elements into a particular folder.

        We currently only support one folder level.

        Args:
            label: The label for the folder.
        """
        self._gui_folder_labels.append(label)
        yield
        assert self._gui_folder_labels.pop() == label

    def add_gui_button(self, name: str) -> GuiHandle[bool]:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`.

        Currently, all button names need to be unique.

        Args:
            name: The name of the gui element.
        """
        return self._add_gui_impl(
            name,
            initial_value=False,
            leva_conf={"type": "BUTTON", "settings": {}},
            is_button=True,
        )

    def add_gui_checkbox(self, name: str, initial_value: bool, hint: Optional[str] = None) -> GuiHandle[bool]:
        """Add a checkbox to the GUI.

        Args:
            name: The name of the checkbox.
            initial_value: The initial value of the checkbox.
            hint: A hint for the checkbox.
        """
        assert isinstance(initial_value, bool)
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={"value": initial_value, "label": name},
            hint=hint,
        )

    def add_gui_text(self, name: str, initial_value: str, hint: Optional[str] = None) -> GuiHandle[str]:
        """Add a text input to the GUI.

        Args:
            name: The name of the text input.
            initial_value: The initial value of the text input.
            hint: A hint for the text input.
        """
        assert isinstance(initial_value, str)
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={"value": initial_value, "label": name},
            hint=hint,
        )

    def add_gui_number(self, name: str, initial_value: IntOrFloat, hint: Optional[str] = None) -> GuiHandle[IntOrFloat]:
        """Add a number input to the GUI.

        Args:
            name: The name of the number.
            initial_value: The initial value of the number.
            hint: A hint for the number.
        """
        assert isinstance(initial_value, (int, float))
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={"value": initial_value, "label": name},
            hint=hint,
        )

    def add_gui_vector2(
        self,
        name: str,
        initial_value: Tuple[float, float] | onp.ndarray,
        step: Optional[float] = None,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI.

        Args:
            name: The name of the vector.
            initial_value: The initial value of the vector.
            step: The step size for the vector.
            hint: A hint for the vector.
        """
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            _cast_vector(initial_value, length=2),
            leva_conf={
                "value": initial_value,
                "label": name,
                "step": step,
            },
            hint=hint,
        )

    def add_gui_vector3(
        self,
        name: str,
        initial_value: Tuple[float, float, float] | onp.ndarray,
        step: Optional[float] = None,
        lock: bool = False,
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI.

        Args:
            name: The name of the vector.
            initial_value: The initial value of the vector.
            step: The step size for the vector.
            lock: Whether the vector is locked.
        """
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            _cast_vector(initial_value, length=3),
            leva_conf={
                "label": name,
                "value": initial_value,
                "step": step,
                "lock": lock,
            },
            hint=hint,
        )

    # Resolve type of value to a Literal whenever possible.
    @overload
    def add_gui_select(
        self,
        name: str,
        options: List[TLiteralString],
        initial_value: Optional[TLiteralString] = None,
        hint: Optional[str] = None,
    ) -> GuiSelectHandle[TLiteralString]: ...

    @overload
    def add_gui_select(
        self,
        name: str,
        options: List[str],
        initial_value: Optional[str] = None,
        hint: Optional[str] = None,
    ) -> GuiSelectHandle[str]: ...

    def add_gui_select(
        self,
        name: str,
        options: List[TLiteralString] | List[str],
        initial_value: Optional[TLiteralString | str] = None,
        hint: Optional[str] = None,
    ) -> GuiSelectHandle[TLiteralString] | GuiSelectHandle[str]:
        """Add a dropdown to the GUI.

        Args:
            name: The name of the dropdown.
            options: The options to choose from.
            initial_value: The initial value of the dropdown.
            hint: A hint for the dropdown.
        """
        assert len(options) > 0
        if initial_value is None:
            initial_value = options[0]
        return GuiSelectHandle(
            self._add_gui_impl(
                "/".join(self._gui_folder_labels + [name]),
                initial_value,
                leva_conf={
                    "value": initial_value,
                    "label": name,
                    "options": options,
                },
                hint=hint,
            )._impl
        )

    # Resolve type of value to a Literal whenever possible.
    @overload
    def add_gui_button_group(
        self,
        name: str,
        options: List[TLiteralString],
        initial_value: Optional[TLiteralString] = None,
    ) -> GuiHandle[TLiteralString]: ...

    @overload
    def add_gui_button_group(
        self,
        name: str,
        options: List[str],
        initial_value: Optional[str] = None,
    ) -> GuiHandle[str]: ...

    def add_gui_button_group(
        self,
        name: str,
        options: List[TLiteralString] | List[str],
        initial_value: Optional[TLiteralString | str] = None,
    ) -> GuiHandle[TLiteralString] | GuiHandle[str]:
        """Add a button group to the GUI.

        Args:
            name: The name of the button group.
            options: The options to choose from.
            initial_value: The initial value of the button group.
        """
        assert len(options) > 0
        if initial_value is None:
            initial_value = options[0]
        return self._add_gui_impl(
            name,
            initial_value,
            leva_conf={"type": "BUTTON_GROUP", "label": name, "options": options},
            is_button=True,
        )

    def add_gui_slider(
        self,
        name: str,
        low: IntOrFloat,
        high: IntOrFloat,
        step: Optional[IntOrFloat],
        initial_value: IntOrFloat,
        hint: Optional[str] = None,
    ) -> GuiHandle[IntOrFloat]:
        """Add a slider to the GUI.

        Args:
            name: The name of the slider.
            low: The minimum value of the slider.
            high: The maximum value of the slider.
            step: The step size of the slider.
            initial_value: The initial value of the slider.
            hint: A hint for the slider.
        """
        assert high >= low
        if step is not None:
            assert step <= (high - low)
        assert high >= initial_value >= low

        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={
                "value": initial_value,
                "label": name,
                "min": low,
                "max": high,
                "step": step,
            },
            hint=hint,
        )

    def add_gui_rgb(
        self,
        name: str,
        initial_value: Tuple[int, int, int],
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[int, int, int]]:
        """Add an RGB picker to the GUI.

        Args:
            name: The name of the color picker.
            initial_value: The initial value of the color picker.
            hint: A hint for color picker.
        """
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={
                "value": {
                    "r": initial_value[0],
                    "g": initial_value[1],
                    "b": initial_value[2],
                },
                "label": name,
            },
            encoder=lambda rgb: dict(zip("rgb", rgb)),
            decoder=lambda rgb_dict: (rgb_dict["r"], rgb_dict["g"], rgb_dict["b"]),
            hint=hint,
        )

    def add_gui_rgba(
        self,
        name: str,
        initial_value: Tuple[int, int, int, int],
        hint: Optional[str] = None,
    ) -> GuiHandle[Tuple[int, int, int, int]]:
        """Add an RGBA picker to the GUI.

        Args:
            name: The name of the color picker.
            initial_value: The initial value of the color picker.
            hint: A hint for color picker.
        """
        return self._add_gui_impl(
            "/".join(self._gui_folder_labels + [name]),
            initial_value,
            leva_conf={
                "value": {
                    "r": initial_value[0],
                    "g": initial_value[1],
                    "b": initial_value[2],
                    "a": initial_value[3],
                },
                "label": name,
            },
            encoder=lambda rgba: dict(zip("rgba", rgba)),
            decoder=lambda rgba_dict: (
                rgba_dict["r"],
                rgba_dict["g"],
                rgba_dict["b"],
                rgba_dict["a"],
            ),
            hint=hint,
        )

    def set_background_image(
        self,
        image: onp.ndarray,
        file_format: Literal["png", "jpeg"] = "jpeg",
        quality: Optional[int] = None,
    ) -> None:
        """Set the background image of the scene.

        Args:
            image: The image to set as the background. Must be a 3D numpy array of shape (H, W, 3).
            file_format: The file format to use for the image.
            quality: The quality of the image, if using jpeg. Must be an integer between 0 and 100.
        """
        media_type, base64_data = _encode_image_base64(image, file_format, quality=quality)
        self._queue(messages.BackgroundImageMessage(media_type=media_type, base64_data=base64_data))

    def send_file_path_info(self, config_base_dir: Path, data_base_dir: Path, export_path_name: str) -> None:
        """Send file path info to the scene.

        Args:
            config_base_dir: The base directory for config files.
            data_base_dir: The base directory for data files.
            export_path_name: The name for the export folder.
        """
        self._queue(
            messages.FilePathInfoMessage(
                config_base_dir=str(config_base_dir),
                data_base_dir=str(data_base_dir),
                export_path_name=export_path_name,
            )
        )

    def update_scene_box(self, scene_box: SceneBox) -> None:
        """Update the scene box.

        Args:
            scene_box: The scene box.
        """
        self._queue(
            messages.SceneBoxMessage(
                min=tuple(scene_box.aabb[0].tolist()),  # type: ignore
                max=tuple(scene_box.aabb[1].tolist()),  # type: ignore
            )
        )

    def add_dataset_image(self, idx: str, json: Dict) -> None:
        """Add a dataset image to the scene.

        Args:
            idx: The index of the image.
            json: The json dict from the camera frustum and image.
        """
        self._queue(messages.DatasetImageMessage(idx=idx, json=json))

    def set_training_state(self, training_state: Literal["training", "paused", "completed"]) -> None:
        """Set the training mode.

        Args:
            training_state: The training mode.
        """
        self._queue(messages.TrainingStateMessage(training_state=training_state))

    def set_camera(
        self,
        position: Optional[Tuple[float, float, float]] = None,
        look_at: Optional[Tuple[float, float, float]] = None,
        fov: Optional[int] = None,
        instant: bool = False,
    ) -> None:
        """Update the camera object in the viewer. If any of the arguments are None, the corresponding value will not
        be set in the viewer. For example, setting position only will maintain the same look-at point while moving
        the origin of the camera

        Args:
            position: The position in world coordinates of the camera
            look_at: The position in world coordinates of the new look at point
            fov: The new field of view
            instant: Whether to move the camera instantly or animate
        """
        self._queue(messages.SetCameraMessage(look_at=look_at, position=position, fov=fov, instant=instant))

    def send_camera_paths(self, camera_paths: Dict[str, Any]) -> None:
        """Send camera paths to the scene.

        Args:
            camera_paths: A dictionary of camera paths.
        """
        self._queue(messages.CameraPathsMessage(payload=camera_paths))

    def send_crop_params(
        self,
        crop_enabled: bool,
        crop_bg_color: Tuple[int, int, int],
        crop_center: Tuple[float, float, float],
        crop_scale: Tuple[float, float, float],
    ) -> None:
        """Send crop parameters to the scene.

        Args:
            crop_enabled: Whether crop is enabled.
            crop_bg_color: The background color of the crop.
            crop_center: The center of the crop.
            crop_scale: The scale of the crop.
        """
        self._queue(
            messages.CropParamsMessage(
                crop_enabled=crop_enabled, crop_bg_color=crop_bg_color, crop_center=crop_center, crop_scale=crop_scale
            )
        )

    def send_status_message(self, eval_res: str, step: int):
        """Send status message

        Args:
            eval_res: The resolution of the render in plain text.
            step: The current step.
        """
        self._queue(messages.StatusMessage(eval_res=eval_res, step=step))

    def send_output_options_message(self, options: List[str]):
        """Send output options message

        Args:
            options: The list of output options
        """
        self._queue(messages.OutputOptionsMessage(options=options))

    def _add_gui_impl(
        self,
        name: str,
        initial_value: T,
        leva_conf: dict,
        is_button: bool = False,
        encoder: Callable[[T], Any] = lambda x: x,
        decoder: Callable[[Any], T] = lambda x: x,
        hint: Optional[str] = None,
    ) -> GuiHandle[T]:
        """Private helper for adding a simple GUI element."""

        if hint is not None:
            assert not is_button
            leva_conf["hint"] = hint

        handle_state = _GuiHandleState(
            name,
            typ=type(initial_value),
            api=self,
            value=initial_value,
            last_updated=time.time(),
            folder_labels=self._gui_folder_labels,
            update_cb=[],
            leva_conf=leva_conf,
            is_button=is_button,
            encoder=encoder,
            decoder=decoder,
        )
        self._handle_state_from_gui_name[name] = handle_state
        handle_state.cleanup_cb = lambda: self._handle_state_from_gui_name.pop(name)

        # For broadcasted GUI handles, we should synchronize all clients.
        # This will be a no-op for client handles.
        if not is_button:

            def sync_other_clients(client_id: ClientId, value: Any) -> None:
                message = messages.GuiSetValueMessage(name=name, value=handle_state.encoder(value))
                message.excluded_self_client = client_id
                self._queue(message)

            handle_state.sync_cb = sync_other_clients

        self._queue(
            messages.GuiAddMessage(
                name=name,
                folder_labels=tuple(self._gui_folder_labels),
                leva_conf=leva_conf,
            )
        )
        return GuiHandle(handle_state)

    def use_time_conditioning(self) -> None:
        """Use time conditioning."""
        self._queue(messages.UseTimeConditioningMessage())

    def _handle_gui_updates(
        self: MessageApi,
        client_id: ClientId,
        message: messages.GuiUpdateMessage,
    ) -> None:
        handle_state = self._handle_state_from_gui_name.get(message.name, None)
        if handle_state is None:
            return

        value = handle_state.typ(handle_state.decoder(message.value))

        # Only call update when value has actually changed.
        if not handle_state.is_button and value == handle_state.value:
            return

        # Update state.
        handle_state.value = value
        handle_state.last_updated = time.time()

        # Trigger callbacks.
        for cb in handle_state.update_cb:
            cb(GuiHandle(handle_state))
        if handle_state.sync_cb is not None:
            handle_state.sync_cb(client_id, value)
