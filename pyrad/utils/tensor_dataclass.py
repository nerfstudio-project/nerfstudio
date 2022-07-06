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
from typing import Tuple, Union

import numpy as np
import torch


class TensorDataclass:
    """Data class of tensors with the same size batch. Allows indexing and standard tensor ops.
    Fields that are not Tensors will not be batched unless they are also a TensorDataclass.

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

        field_names = [f.name for f in dataclasses.fields(self)]
        batch_shapes = []
        for f in field_names:
            v = self.__getattribute__(f)
            if v is not None:
                if isinstance(v, torch.Tensor):
                    batch_shapes.append(v.shape[:-1])
                elif isinstance(v, TensorDataclass):
                    batch_shapes.append(v.shape)
        if len(batch_shapes) == 0:
            raise ValueError("TensorDataclass must have at least one tensor")
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        for f in field_names:
            v = self.__getattribute__(f)
            if v is not None:
                if isinstance(v, torch.Tensor):
                    self.__setattr__(f, v.broadcast_to((*batch_shape, v.shape[-1])))
                elif isinstance(v, TensorDataclass):
                    self.__setattr__(f, v.broadcast_to(batch_shape))

        self.__setattr__("_shape", batch_shape)

    def __getitem__(self, indices) -> "TensorDataclass":
        if isinstance(indices, torch.Tensor):
            return self._apply_fn_to_fields(lambda x: x[indices])
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        tensor_fn = lambda x: x[indices + (slice(None),)]
        dataclass_fn = lambda x: x[indices]
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn)

    def __setitem__(self, indices, value) -> "TensorDataclass":
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
    def shape(self) -> tuple:
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

    def reshape(self, shape: Tuple[int, ...]) -> "TensorDataclass":
        """Returns a new TensorDataclass with the same data but with a new shape.

        Args:
            shape (Tuple[int]): The new shape of the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        if isinstance(shape, int):
            shape = (shape,)
        tensor_fn = lambda x: x.reshape((*shape, x.shape[-1]))
        dataclass_fn = lambda x: x.reshape(shape)
        return self._apply_fn_to_fields(tensor_fn, dataclass_fn)

    def flatten(self) -> "TensorDataclass":
        """Returns a new TensorDataclass with flattened batch dimensions

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self.reshape((-1,))

    def broadcast_to(self, shape: Union[torch.Size, Tuple[int]]) -> "TensorDataclass":
        """Returns a new TensorDataclass broadcast to new shape.

        Args:
            shape (Tuple[int]): The new shape of the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        return self._apply_fn_to_fields(lambda x: x.broadcast_to((*shape, x.shape[-1])))

    def to(self, device) -> "TensorDataclass":
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but on the specified device.
        """
        return self._apply_fn_to_fields(lambda x: x.to(device))

    def _apply_fn_to_fields(self, fn: callable, dataclass_fn: callable = None) -> "TensorDataclass":
        """Applies a function to all fields of the tensor dataclass.

        Args:
            fn (callable): The function to apply to tensor fields.
            dataclass_fn (callable): The function to apply to TensorDataclass fields. Else use fn.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """

        field_names = [f.name for f in dataclasses.fields(self)]
        new_fields = {}
        for f in field_names:
            v = self.__getattribute__(f)
            if v is not None:
                if isinstance(v, TensorDataclass) and dataclass_fn is not None:
                    new_fields[f] = dataclass_fn(v)
                elif isinstance(v, (torch.Tensor, TensorDataclass)):
                    new_fields[f] = fn(v)

        return dataclasses.replace(self, **new_fields)
