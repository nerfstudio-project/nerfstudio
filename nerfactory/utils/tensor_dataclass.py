# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""Tensor dataclass"""

import dataclasses
from copy import deepcopy
from typing import Callable, Dict, NoReturn, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

TensorDataclassT = TypeVar("TensorDataclassT", bound="TensorDataclass")


class TensorDataclass:
    """@dataclass of tensors with the same size batch. Allows indexing and standard tensor ops.
    Fields that are not Tensors will not be batched unless they are also a TensorDataclass.
    Any fields that are dictionaries will have their Tensors or TensorDataclasses batched, and
    dictionaries will have their tensors or TensorDataclasses considered in the initial broadcast.

    Example:

    .. code-block:: python

        @dataclass
        class TestTensorDataclass(TensorDataclass):
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor = None

        # Create a new tensor dataclass with batch size of [2,3,4]
        test = TestTensorDataclass(a=torch.ones((2, 3, 4, 2)), b=torch.ones((4, 3)))

        test.shape  # [2, 3, 4]
        test.a.shape  # [2, 3, 4, 2]
        test.b.shape  # [2, 3, 4, 3]

        test.reshape((6,4)).shape  # [6, 4]
        test.flatten().shape  # [24,]

        test[..., 0].shape  # [2, 3]
        test[:, 0, :].shape  # [2, 4]
    """

    _shape: tuple

    def __post_init__(self) -> None:
        if not dataclasses.is_dataclass(self):
            raise TypeError("TensorDataclass must be a dataclass")

        batch_shapes = self._get_dict_batch_shapes(dataclasses.asdict(self))
        if len(batch_shapes) == 0:
            raise ValueError("TensorDataclass must have at least one tensor")
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        broadcasted_fields = self._broadcast_dict_fields(dataclasses.asdict(self), batch_shape)
        for f, v in broadcasted_fields.items():
            self.__setattr__(f, v)

        self.__setattr__("_shape", batch_shape)

    def _get_dict_batch_shapes(self, dict_: Dict) -> list:
        """Returns batch shapes of all tensors in a dictionary"""
        batch_shapes = []
        for v in dict_.values():
            if isinstance(v, torch.Tensor):
                batch_shapes.append(v.shape[:-1])
            elif isinstance(v, TensorDataclass):
                batch_shapes.append(v.shape)
            elif isinstance(v, Dict):
                batch_shapes.extend(self._get_dict_batch_shapes(v))
        return batch_shapes

    def _broadcast_dict_fields(self, dict_: Dict, batch_shape) -> Dict:
        """Broadcasts all tensors in a dictionary according to batch_shape"""
        new_dict = {}
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.broadcast_to((*batch_shape, v.shape[-1]))
            elif isinstance(v, TensorDataclass):
                new_dict[k] = v.broadcast_to(batch_shape)
            elif isinstance(v, Dict):
                new_dict[k] = self._broadcast_dict_fields(v, batch_shape)
        return new_dict

    def __getitem__(self: TensorDataclassT, indices) -> TensorDataclassT:
        if isinstance(indices, torch.Tensor):
            return self._apply_fn_to_fields(lambda x: x[indices])
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        tensor_fn = lambda x: x[indices + (slice(None),)]
        dataclass_fn = lambda x: x[indices]
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn)

    def __setitem__(self, indices, value) -> NoReturn:
        raise RuntimeError("Index assignment is not supported for TensorDataclass")

    def __len__(self) -> int:
        return self.shape[0]

    def __bool__(self) -> bool:
        if len(self) == 0:
            raise ValueError(
                f"The truth value of {self.__class__.__name__} when `len(x) == 0` "
                "is ambiguous. Use `len(x)` or `x is not None`."
            )
        return True

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the batch shape of the tensor dataclass."""
        return self._shape

    @property
    def size(self) -> int:
        """Returns the number of elements in the tensor dataclass batch dimension."""
        if len(self._shape) == 0:
            return 1
        return int(np.prod(self._shape))

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor dataclass."""
        return len(self._shape)

    def reshape(self: TensorDataclassT, shape: Tuple[int, ...]) -> TensorDataclassT:
        """Returns a new TensorDataclass with the same data but with a new shape.

        This should deepcopy as well.

        Args:
            shape (Tuple[int, ...]): The new shape of the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        if isinstance(shape, int):
            shape = (shape,)
        tensor_fn = lambda x: x.reshape((*shape, x.shape[-1]))
        dataclass_fn = lambda x: x.reshape(shape)
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn)

    def flatten(self: TensorDataclassT) -> TensorDataclassT:
        """Returns a new TensorDataclass with flattened batch dimensions

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self.reshape((-1,))

    def broadcast_to(self: TensorDataclassT, shape: Union[torch.Size, Tuple[int, ...]]) -> TensorDataclassT:
        """Returns a new TensorDataclass broadcast to new shape.

        Changes to the original tensor dataclass should effect the returned tensor dataclass,
        meaning it is NOT a deepcopy, and they are still linked.

        Args:
            shape (Tuple[int, ...]): The new shape of the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self._apply_fn_to_fields(lambda x: x.broadcast_to((*shape, x.shape[-1])))

    def to(self: TensorDataclassT, device) -> TensorDataclassT:
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but on the specified device.
        """
        return self._apply_fn_to_fields(lambda x: x.to(device))

    def _apply_fn_to_fields(
        self: TensorDataclassT, fn: Callable, dataclass_fn: Optional[Callable] = None
    ) -> TensorDataclassT:
        """Applies a function to all fields of the tensor dataclass.

        Args:
            fn (Callable): The function to apply to tensor fields.
            dataclass_fn (Optional[Callable]): The function to apply to TensorDataclass fields.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """

        new_fields = self._apply_fn_to_dict(dataclasses.asdict(self), fn, dataclass_fn)

        return dataclasses.replace(self, **new_fields)

    def _apply_fn_to_dict(self, dict_: Dict, fn: Callable, dataclass_fn: Optional[Callable] = None) -> Dict:
        """A helper function for _apply_fn_to_fields, applying a function to all fields of dict_"""

        field_names = dict_.keys()
        new_dict = {}
        for f in field_names:
            v = dict_[f]
            if v is not None:
                if isinstance(v, TensorDataclass) and dataclass_fn is not None:
                    new_dict[f] = dataclass_fn(v)
                elif isinstance(v, (torch.Tensor, TensorDataclass)):
                    new_dict[f] = fn(v)
                elif isinstance(v, Dict):
                    new_dict[f] = self._apply_fn_to_dict(v, fn, dataclass_fn)
                else:
                    new_dict[f] = deepcopy(v)

        return new_dict
