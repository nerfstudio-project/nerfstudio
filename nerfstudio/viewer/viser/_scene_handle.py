from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, TypeVar, cast

import numpy as onp

from . import _messages

if TYPE_CHECKING:
    from ._message_api import ClientId, MessageApi

TVector = TypeVar("TVector", bound=tuple)


def _cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
    if isinstance(vector, tuple):
        assert len(vector) == length
        return cast(TVector, vector)
    else:
        assert cast(onp.ndarray, vector).shape == (length,)
        return cast(TVector, tuple(map(float, vector)))


# TODO(by): we can add helpers for stuff like removing scene nodes, click events,
# etc here...

# @dataclasses.dataclass
# class _SceneHandleState:
#     name: str
#     api: MessageApi
#
#
# @dataclasses.dataclass(frozen=True)
# class SceneHandle:
#     _impl: _SceneHandleState
#


@dataclasses.dataclass
class _TransformControlsState:
    name: str
    api: MessageApi
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    last_updated: float

    update_cb: List[Callable[[TransformControlsHandle], None]]
    sync_cb: Optional[Callable[[ClientId, _TransformControlsState], None]] = None


@dataclasses.dataclass(frozen=True)
class TransformControlsState:
    wxyz: Tuple[float, float, float, float]
    position: Tuple[float, float, float]
    last_updated: float


@dataclasses.dataclass(frozen=True)
class TransformControlsHandle:
    _impl: _TransformControlsState

    def get_state(self) -> TransformControlsState:
        return TransformControlsState(
            self._impl.wxyz, self._impl.position, self._impl.last_updated
        )

    def on_update(
        self, func: Callable[[TransformControlsHandle], None]
    ) -> Callable[[TransformControlsHandle], None]:
        self._impl.update_cb.append(func)
        return func

    def set_state(
        self,
        wxyz: Tuple[float, float, float, float] | onp.ndarray,
        position: Tuple[float, float, float] | onp.ndarray,
    ) -> None:
        self._impl.api._queue(
            _messages.TransformControlsSetMessage(
                self._impl.name, _cast_vector(wxyz, 4), _cast_vector(position, 3)
            )
        )
