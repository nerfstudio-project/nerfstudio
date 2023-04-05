from __future__ import annotations

import abc
import base64
import contextlib
import io
import time
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
)

import imageio.v3 as iio
import numpy as onp
import numpy.typing as onpt
from typing_extensions import LiteralString, ParamSpec, assert_never

from . import _messages
from ._gui import GuiHandle, _GuiHandleState
from ._scene_handle import TransformControlsHandle, _TransformControlsState

if TYPE_CHECKING:
    from ._server import ClientId


P = ParamSpec("P")


# TODO(by): the function signatures below are super redundant.
#
# We can strip out a ton of it and replace with a ParamSpec-based decorator factory, but
# we'd be reliant on a pyright bug fix that only just happened:
#     https://github.com/microsoft/pyright/issues/4813
#
# We should probably wait at least until the next pylance version drops.
#
# def _wrap_message(
#     message_cls: Callable[P, _messages.Message]
# ) -> Callable[[Callable], Callable[P, None]]:
#     """Wrap a message type."""
#
#     def inner(self: MessageApi, *args: P.args, **kwargs: P.kwargs) -> None:
#         message = message_cls(*args, **kwargs)
#         self._queue(message)
#
#     return lambda _: inner  # type: ignore


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
    format: Literal["png", "jpeg"],
    quality: Optional[int] = None,
) -> Tuple[Literal["image/png", "image/jpeg"], str]:
    media_type: Literal["image/png", "image/jpeg"]
    image = _colors_to_uint8(image)
    with io.BytesIO() as data_buffer:
        if format == "png":
            media_type = "image/png"
            iio.imwrite(data_buffer, image, format="PNG")
        elif format == "jpeg":
            media_type = "image/jpeg"
            iio.imwrite(
                data_buffer,
                image[..., :3],  # Strip alpha.
                format="JPEG",
                quality=75 if quality is None else quality,
            )
        else:
            assert_never(format)

        base64_data = base64.b64encode(data_buffer.getvalue()).decode("ascii")

    return media_type, base64_data


TVector = TypeVar("TVector", bound=tuple)


def _cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if isinstance(vector, tuple):
        assert len(vector) == length
        return cast(TVector, vector)
    else:
        assert cast(onp.ndarray, vector).shape == (length,)
        return cast(TVector, tuple(map(float, vector)))


IntOrFloat = TypeVar("IntOrFloat", int, float)
TLiteralString = TypeVar("TLiteralString", bound=LiteralString)


class MessageApi(abc.ABC):
    """Interface for all commands we can use to send messages over a websocket connection.

    Should be implemented by both our global server object (for broadcasting) and by
    invidividual clients."""

    def __init__(self) -> None:
        self._handle_state_from_gui_name: Dict[str, _GuiHandleState[Any]] = {}
        self._handle_state_from_transform_controls_name: Dict[
            str, _TransformControlsState
        ] = {}
        self._incoming_handlers: List[Callable[[ClientId, _messages.Message], None]] = [
            lambda client_id, msg: _handle_gui_updates(self, client_id, msg),
            lambda client_id, msg: _handle_transform_controls_updates(
                self, client_id, msg
            ),
        ]
        self._gui_folder_label = "User"

    @contextlib.contextmanager
    def gui_folder(self, label: str) -> Generator[None, None, None]:
        """Context for placing all GUI elements into a particular folder.

        We currently only support one folder level."""
        old_folder_label = self._gui_folder_label
        self._gui_folder_label = label
        yield
        self._gui_folder_label = old_folder_label

    def add_gui_button(self, name: str, disabled: bool = False) -> GuiHandle[bool]:
        """Add a button to the GUI. The value of this input is set to `True` every time
        it is clicked; to detect clicks, we can manually set it back to `False`.

        Currently, all button names need to be unique."""
        return _add_gui_impl(
            self,
            name,
            initial_value=False,
            leva_conf={"type": "BUTTON", "settings": {"disabled": disabled}},
            is_button=True,
        )

    def add_gui_checkbox(
        self, name: str, initial_value: bool, disabled: bool = False
    ) -> GuiHandle[bool]:
        """Add a checkbox to the GUI."""
        assert isinstance(initial_value, bool)
        return _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            initial_value,
            leva_conf={"value": initial_value, "label": name, "disabled": disabled},
        )

    def add_gui_text(
        self, name: str, initial_value: str, disabled: bool = False
    ) -> GuiHandle[str]:
        """Add a text input to the GUI."""
        assert isinstance(initial_value, str)
        return _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            initial_value,
            leva_conf={"value": initial_value, "label": name, "disabled": disabled},
        )

    def add_gui_number(
        self, name: str, initial_value: IntOrFloat, disabled: bool = False
    ) -> GuiHandle[IntOrFloat]:
        """Add a number input to the GUI."""
        assert isinstance(initial_value, (int, float))
        return _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            initial_value,
            leva_conf={"value": initial_value, "label": name, "disabled": disabled},
        )

    def add_gui_vector2(
        self,
        name: str,
        initial_value: Tuple[float, float] | onp.ndarray,
        step: Optional[float] = None,
        disabled: bool = False,
    ) -> GuiHandle[Tuple[float, float]]:
        """Add a length-2 vector input to the GUI."""
        return _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            _cast_vector(initial_value, length=2),
            leva_conf={
                "value": initial_value,
                "label": name,
                "step": step,
                "disabled": disabled,
            },
        )

    def add_gui_vector3(
        self,
        name: str,
        initial_value: Tuple[float, float, float] | onp.ndarray,
        step: Optional[float] = None,
        lock: bool = False,
        disabled: bool = False,
    ) -> GuiHandle[Tuple[float, float, float]]:
        """Add a length-3 vector input to the GUI."""
        return _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            _cast_vector(initial_value, length=3),
            leva_conf={
                "label": name,
                "value": initial_value,
                "step": step,
                "lock": lock,
                "disabled": disabled,
            },
        )

    def add_gui_select(
        self,
        name: str,
        options: List[TLiteralString],
        initial_value: Optional[TLiteralString] = None,
        disabled: bool = False,
    ) -> GuiHandle[TLiteralString]:
        """Add a dropdown to the GUI."""
        assert len(options) > 0
        if initial_value is None:
            initial_value = options[0]
        out: GuiHandle[TLiteralString] = _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            initial_value,
            leva_conf={
                "value": initial_value,
                "label": name,
                "options": options,
                "disabled": disabled,
            },
        )
        return out

    def add_gui_slider(
        self,
        name: str,
        min: IntOrFloat,
        max: IntOrFloat,
        step: Optional[IntOrFloat],
        initial_value: IntOrFloat,
        disabled: bool = False,
    ) -> GuiHandle[IntOrFloat]:
        """Add a dropdown to the GUI."""
        assert max >= min
        if step is not None:
            assert step <= (max - min)
        assert max >= initial_value >= min

        return _add_gui_impl(
            self,
            self._gui_folder_label + "/" + name,
            initial_value,
            leva_conf={
                "value": initial_value,
                "label": name,
                "min": min,
                "max": max,
                "step": step,
                "disabled": disabled,
            },
        )

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        scale: float = 0.3,
        color: Tuple[int, int, int]
        | Tuple[float, float, float]
        | onp.ndarray = (80, 120, 255),
    ) -> None:
        color = tuple(
            value if isinstance(value, int) else int(value * 255)  # type: ignore
            for value in color
        )
        self._queue(
            _messages.CameraFrustumMessage(
                name=name,
                fov=fov,
                aspect=aspect,
                scale=scale,
                # (255, 255, 255) => 0xffffff, etc
                color=int(color[0] * (256**2) + color[1] * 256 + color[2]),
            )
        )

    def add_frame(
        self,
        name: str,
        wxyz: Tuple[float, float, float, float] | onp.ndarray,
        position: Tuple[float, float, float] | onp.ndarray,
        show_axes: bool = True,
        axes_length: float = 0.5,
        axes_radius: float = 0.025,
    ) -> None:
        self._queue(
            _messages.FrameMessage(
                name=name,
                wxyz=_cast_vector(wxyz, length=4),
                position=_cast_vector(position, length=3),
                show_axes=show_axes,
                axes_length=axes_length,
                axes_radius=axes_radius,
            )
        )

    def add_point_cloud(
        self,
        name: str,
        position: onp.ndarray,
        color: onp.ndarray,
        point_size: float = 0.1,
    ) -> None:
        self._queue(
            _messages.PointCloudMessage(
                name=name,
                position=position.astype(onp.float32),
                color=_colors_to_uint8(color),
                point_size=point_size,
            )
        )

    def add_mesh(
        self,
        name: str,
        vertices: onp.ndarray,
        faces: onp.ndarray,
        color: Tuple[int, int, int]
        | Tuple[float, float, float]
        | onp.ndarray = (90, 200, 255),
        wireframe: bool = False,
    ) -> None:
        color = tuple(
            value if isinstance(value, int) else int(value * 255)  # type: ignore
            for value in color
        )
        self._queue(
            _messages.MeshMessage(
                name,
                vertices.astype(onp.float32),
                faces.astype(onp.uint32),
                # (255, 255, 255) => 0xffffff, etc
                color=int(color[0] * (256**2) + color[1] * 256 + color[2]),
                wireframe=wireframe,
            )
        )

    def set_background_image(
        self,
        image: onp.ndarray,
        format: Literal["png", "jpeg"] = "jpeg",
        quality: Optional[int] = None,
    ) -> None:
        media_type, base64_data = _encode_image_base64(image, format, quality=quality)
        self._queue(
            _messages.BackgroundImageMessage(
                media_type=media_type, base64_data=base64_data
            )
        )

    def add_image(
        self,
        name: str,
        image: onp.ndarray,
        render_width: float,
        render_height: float,
        format: Literal["png", "jpeg"] = "jpeg",
        quality: Optional[int] = None,
    ) -> None:
        media_type, base64_data = _encode_image_base64(image, format, quality=quality)
        self._queue(
            _messages.ImageMessage(
                name=name,
                media_type=media_type,
                base64_data=base64_data,
                render_width=render_width,
                render_height=render_height,
            )
        )

    def add_transform_controls(
        self,
        name: str,
        scale: float = 1.0,
        line_width: float = 2.5,
        fixed: bool = False,
        auto_transform: bool = True,
        active_axes: Tuple[bool, bool, bool] = (True, True, True),
        disable_axes: bool = False,
        disable_sliders: bool = False,
        disable_rotations: bool = False,
        translation_limits: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ] = ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        rotation_limits: Tuple[
            Tuple[float, float], Tuple[float, float], Tuple[float, float]
        ] = ((-1000.0, 1000.0), (-1000.0, 1000.0), (-1000.0, 1000.0)),
        depth_test: bool = True,
        opacity: float = 1.0,
    ) -> TransformControlsHandle:
        # That decorator factory would be really helpful here...
        self._queue(
            _messages.TransformControlsMessage(
                name=name,
                scale=scale,
                line_width=line_width,
                fixed=fixed,
                auto_transform=auto_transform,
                active_axes=active_axes,
                disable_axes=disable_axes,
                disable_sliders=disable_sliders,
                disable_rotations=disable_rotations,
                translation_limits=translation_limits,
                rotation_limits=rotation_limits,
                depth_test=depth_test,
                opacity=opacity,
            )
        )

        def sync_cb(client_id: ClientId, state: _TransformControlsState) -> None:
            message = _messages.TransformControlsSetMessage(
                name=name, wxyz=state.wxyz, position=state.position
            )
            message.excluded_self_client = client_id
            self._queue(message)

        state = _TransformControlsState(
            name=name,
            api=self,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            last_updated=time.time(),
            update_cb=[],
            sync_cb=sync_cb,
        )
        self._handle_state_from_transform_controls_name[name] = state
        return TransformControlsHandle(state)

    def remove_scene_node(self, name: str) -> None:
        self._queue(_messages.RemoveSceneNodeMessage(name=name))

        if name in self._handle_state_from_transform_controls_name:
            self._handle_state_from_transform_controls_name.pop(name)

    def set_scene_node_visibility(self, name: str, visible: bool) -> None:
        self._queue(_messages.SetSceneNodeVisibilityMessage(name=name, visible=visible))

    def reset_scene(self):
        self._queue(_messages.ResetSceneMessage())

    def _handle_incoming_message(
        self, client_id: ClientId, message: _messages.Message
    ) -> None:
        for cb in self._incoming_handlers:
            cb(client_id, message)

    @abc.abstractmethod
    def _queue(self, message: _messages.Message) -> None:
        """Abstract method for sending messages."""
        ...


def _handle_gui_updates(
    self: MessageApi,
    client_id: ClientId,
    message: _messages.Message,
) -> None:
    if not isinstance(message, _messages.GuiUpdateMessage):
        return

    handle_state = self._handle_state_from_gui_name.get(message.name, None)
    if handle_state is None:
        return

    # Only call update when value has actually changed.
    if not handle_state.is_button and message.value == handle_state.value:
        return

    value = handle_state.typ(message.value)

    # Update state.
    handle_state.value = value
    handle_state.last_updated = time.time()

    # Trigger callbacks.
    for cb in handle_state.update_cb:
        cb(GuiHandle(handle_state))
    if handle_state.sync_cb is not None:
        handle_state.sync_cb(client_id, message.value)


def _handle_transform_controls_updates(
    self: MessageApi,
    client_id: ClientId,
    message: _messages.Message,
) -> None:
    if not isinstance(message, _messages.TransformControlsUpdateMessage):
        return

    handle_state = self._handle_state_from_transform_controls_name.get(
        message.name, None
    )
    if handle_state is None:
        return

    # Update state.
    handle_state.wxyz = message.wxyz
    handle_state.position = message.position
    handle_state.last_updated = time.time()

    # Trigger callbacks.
    for cb in handle_state.update_cb:
        cb(TransformControlsHandle(handle_state))
    if handle_state.sync_cb is not None:
        handle_state.sync_cb(client_id, handle_state)


T = TypeVar("T")


def _add_gui_impl(
    api: MessageApi,
    name: str,
    initial_value: T,
    leva_conf: dict,
    is_button: bool = False,
) -> GuiHandle[T]:
    """Private helper for adding a simple GUI element."""

    handle_state = _GuiHandleState(
        name,
        typ=type(initial_value),
        api=api,
        value=initial_value,
        last_updated=time.time(),
        folder_label=api._gui_folder_label,
        update_cb=[],
        leva_conf=leva_conf,
        is_button=is_button,
    )
    api._handle_state_from_gui_name[name] = handle_state
    handle_state.cleanup_cb = lambda: api._handle_state_from_gui_name.pop(name)

    # For broadcasted GUI handles, we should synchronize all clients.
    from ._server import ViserServer

    if not is_button and isinstance(api, ViserServer):

        def sync_other_clients(client_id: ClientId, value: Any) -> None:
            message = _messages.GuiSetValueMessage(name=name, value=value)
            message.excluded_self_client = client_id
            api._queue(message)

        handle_state.sync_cb = sync_other_clients

    api._queue(
        _messages.GuiAddMessage(
            name=name,
            folder=api._gui_folder_label,
            leva_conf=leva_conf,
        )
    )
    return GuiHandle(handle_state)
