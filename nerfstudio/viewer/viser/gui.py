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


""" Manages GUI communication.

Should be almost identical to: https://github.com/brentyi/viser/blob/main/viser/_gui.py
"""
from __future__ import annotations

import dataclasses
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import numpy as onp
from viser.infra import ClientId

from .messages import (
    GuiRemoveMessage,
    GuiSetHiddenMessage,
    GuiSetLevaConfMessage,
    GuiSetValueMessage,
)

if TYPE_CHECKING:
    from .message_api import MessageApi


T = TypeVar("T")


@dataclasses.dataclass
class _GuiHandleState(Generic[T]):
    """Internal API for GUI elements."""

    name: str
    typ: Type[T]
    api: MessageApi
    value: T
    last_updated: float

    folder_labels: List[str]
    """Name of the folders this GUI input was placed into."""

    update_cb: List[Callable[[GuiHandle[T]], None]]
    """Registered functions to call when this input is updated."""

    leva_conf: Dict[str, Any]
    """Input config for Leva."""

    is_button: bool
    """Indicates a button element, which requires special handling."""

    sync_cb: Optional[Callable[[ClientId, T], None]] = None
    """Callback for synchronizing inputs across clients."""

    cleanup_cb: Optional[Callable[[], Any]] = None
    """Function to call when GUI element is removed."""

    # Encoder: run on outgoing message values.
    # Decoder: run on incoming message values.
    #
    # This helps us handle cases where types used by Leva don't match what we want to
    # expose as a Python API.
    #
    # noqa because ruff --fix currently breaks these lines.
    encoder: Callable[[T], Any] = lambda x: x  # noqa
    decoder: Callable[[Any], T] = lambda x: x  # noqa


@dataclasses.dataclass(frozen=True)
class GuiHandle(Generic[T]):
    """Handle for a particular GUI input in our visualizer.

    Lets us get values, set values, and detect updates."""

    # Let's shove private implementation details in here...
    _impl: _GuiHandleState[T]

    def on_update(self, func: Callable[[GuiHandle[T]], None]) -> Callable[[GuiHandle[T]], None]:
        """Attach a function to call when a GUI input is updated. Happens in a thread.

        Callbacks are passed the originating GUI handle, which can be useful in loops.
        """
        self._impl.update_cb.append(func)
        return func

    def get_value(self) -> T:
        """Get the value of the GUI input."""
        return self._impl.value

    def get_update_timestamp(self) -> float:
        """Get the last time that this input was updated."""
        return self._impl.last_updated

    def set_value(self, value: Union[T, onp.ndarray]) -> GuiHandle[T]:
        """Set the value of the GUI input."""
        if isinstance(value, onp.ndarray):
            assert len(value.shape) <= 1, f"{value.shape} should be at most 1D!"
            value = tuple(map(float, value))  # type: ignore

        # Send to client, except for buttons.
        if not self._impl.is_button:
            self._impl.api._queue(GuiSetValueMessage(self._impl.name, self._impl.encoder(value)))  # type: ignore

        # Set internal state. We automatically convert numpy arrays to the expected
        # internal type. (eg 1D arrays to tuples)
        self._impl.value = type(self._impl.value)(value)  # type: ignore
        self._impl.last_updated = time.time()

        # Call update callbacks.
        for cb in self._impl.update_cb:
            cb(self)

        return self

    def set_disabled(self, disabled: bool) -> GuiHandle[T]:
        """Allow/disallow user interaction with the input."""
        if self._impl.is_button:
            self._impl.leva_conf["settings"]["disabled"] = disabled
            self._impl.api._queue(
                GuiSetLevaConfMessage(self._impl.name, self._impl.leva_conf),
            )
        else:
            self._impl.leva_conf["disabled"] = disabled
            self._impl.api._queue(
                GuiSetLevaConfMessage(self._impl.name, self._impl.leva_conf),
            )

        return self

    def set_hidden(self, hidden: bool) -> GuiHandle[T]:
        """Temporarily hide this GUI element from the visualizer."""
        self._impl.api._queue(GuiSetHiddenMessage(self._impl.name, hidden=hidden))
        return self

    def remove(self) -> None:
        """Permanently remove this GUI element from the visualizer."""
        self._impl.api._queue(GuiRemoveMessage(self._impl.name))
        assert self._impl.cleanup_cb is not None
        self._impl.cleanup_cb()


TString = TypeVar("TString", bound=str)


@dataclasses.dataclass(frozen=True)
class GuiSelectHandle(GuiHandle[TString], Generic[TString]):
    """Handle for a particular drop-down input in our visualizer.

    Lets us get values, set values, and detect updates."""

    def set_options(self, options: List[TString]) -> None:
        """Assign a new set of options for the dropdown menu.

        For projects that care about typing: the static type of `options` should be
        consistent with the `StringType` associated with a handle. Literal types will be
        inferred where possible when handles are instantiated; for the most flexibility,
        we can declare handles as `GuiHandle[str]`.
        """

        # Make sure initial value is in options.
        self._impl.leva_conf["options"] = options
        if self._impl.leva_conf["value"] not in options:
            self._impl.leva_conf["value"] = options[0]

        # Update options.
        self._impl.api._queue(
            GuiSetLevaConfMessage(self._impl.name, self._impl.leva_conf),
        )

        # Make sure current value is in options.
        if self.get_value() not in options:
            self.set_value(options[0])
