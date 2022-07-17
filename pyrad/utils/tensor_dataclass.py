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

from typing import List, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

import pyrad.cuda as pyrad_cuda

_IDENTIFER = "__tensordataclass__"


def is_tensordataclass(x):
    """Checks whether a data is a TensorDataclass."""
    return hasattr(x.__class__, _IDENTIFER)


# pylint: disable=unused-variable,eval-used,too-many-statements
def tensordataclass(cls):
    """Decorator that warps a cls into TensorDataclass. In TensorDataclass, all the tensors have the
    same size batch. Allows indexing and standard tensor ops. Fields that are not Tensors will not
    be batched unless they are also a TensorDataclass.

    Example:
    .. code-block:: python
        # pylint: disable=no-member,not-an-iterable
        @tensordataclass
        class TestTensorDataclass():
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

    field_keys = list(cls.__annotations__.keys())
    # _FIELD_TYPES = list(cls.__annotations__.values())

    def __init__(self, *args, **kwargs) -> None:
        # Quietly pop out and save these arguments from kwargs.
        self._shape: Tuple[int] = kwargs.pop("_shape", None)
        self._packed_info: TensorType["num_chunks", 3] = kwargs.pop("_packed_info", None)

        # check the remaining arguments
        n_params = len(args) + len(kwargs.values())
        n_params_max = len(field_keys)
        assert n_params <= n_params_max, (
            f"__init__() takes from 1 to {n_params_max + 1} positional arguments" f" but {n_params + 1} were given"
        )

        # collect attributes
        init_params = {}
        for i, value in enumerate(args):
            key = field_keys[i]
            init_params.update({key: value})
        for key, value in kwargs.items():
            assert key not in init_params, f"__init__() got multiple values for argument '{key}'"
            assert key in field_keys, f"__init__() got an unexpected keyword argument '{key}'"
            init_params.update({key: value})
        for key in field_keys:
            if key in init_params:
                continue
            assert hasattr(self, key), f"__init__() missing 1 required positional argument: '{key}'"
            value = getattr(self, key)
            init_params.update({key: value})

        # set attributes
        for i, (key, value) in enumerate(init_params.items()):
            # TODO(ruilongli): figure out a way to check type for TensorType
            # assert value is None or isinstance(value, _FIELD_TYPES[i]), (
            #     f"__init__() got a wrong type {type(value)} v.s. {_FIELD_TYPES[i]}" f" for argument `{key}`"
            # )
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self) -> None:
        if self.is_packed():
            # Do nothing for a packed tensor
            return

        batch_shapes = []
        for f in field_keys:
            v = self.__getattribute__(f)
            if v is not None:
                if isinstance(v, torch.Tensor):
                    batch_shapes.append(v.shape[:-1])
                elif is_tensordataclass(v):
                    batch_shapes.append(v.shape)
        if len(batch_shapes) == 0:
            raise ValueError("TensorDataclass must have at least one tensor")
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        for f in field_keys:
            v = self.__getattribute__(f)
            if v is not None:
                if isinstance(v, torch.Tensor):
                    self.__setattr__(f, v.broadcast_to((*batch_shape, v.shape[-1])))
                elif is_tensordataclass(v):
                    self.__setattr__(f, v.broadcast_to(batch_shape))

        self._shape = batch_shape

    def __getitem__(self, indices) -> cls:
        if self.is_packed():
            raise IndexError("Packed TensorDataClass can not be indexed!")
        if isinstance(indices, torch.Tensor):
            return self.apply_fn_to_fields(lambda x: x[indices])
        if isinstance(indices, (int, slice)):
            indices = (indices,)
        tensor_fn = lambda x: x[indices + (slice(None),)]
        dataclass_fn = lambda x: x[indices]
        return self.apply_fn_to_fields(tensor_fn, dataclass_fn)

    def __setitem__(self, indices, value) -> cls:
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
        shape = self.__getattribute__("_shape")
        if shape is None:
            raise RuntimeError("Packed TensorDataClass does not have a shape defined!")
        return shape

    @property
    def size(self) -> int:
        """Returns the number of elements in the tensor dataclass batch dimension."""
        if len(self.shape) == 0:
            return 1
        return int(np.prod(self.shape))

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the tensor dataclass."""
        return len(self.shape)

    def reshape(self, shape: Tuple[int, ...]) -> cls:
        """Returns a new TensorDataclass with the same data but with a new shape.

        Args:
            shape (Tuple[int]): The new shape of the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        if self.is_packed():
            raise RuntimeError("Packed TensorDataclass can not be reshaped!")
        if isinstance(shape, int):
            shape = (shape,)
        tensor_fn = lambda x: x.reshape((*shape, x.shape[-1]))
        dataclass_fn = lambda x: x.reshape(shape)
        return self.apply_fn_to_fields(tensor_fn, dataclass_fn)

    def flatten(self) -> cls:
        """Returns a new TensorDataclass with flattened batch dimensions

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        if self.is_packed():
            raise RuntimeError("Packed TensorDataclass can not be flatten!")
        return self.reshape((-1,))

    def broadcast_to(self, shape: Union[torch.Size, Tuple[int]]) -> cls:
        """Returns a new TensorDataclass broadcast to new shape.

        Args:
            shape (Tuple[int]): The new shape of the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but with a new shape.
        """
        if self.is_packed():
            raise RuntimeError("Packed TensorDataclass can not be broadcasted!")
        return self.apply_fn_to_fields(lambda x: x.broadcast_to((*shape, x.shape[-1])))

    def to(self, device) -> cls:
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but on the specified device.
        """
        return self.apply_fn_to_fields(lambda x: x.to(device))

    def apply_fn_to_fields(
        self,
        fn: callable,
        dataclass_fn: callable = None,
        exclude_fields: List[str] = None,
        **kwargs,
    ) -> cls:
        """Applies a function to all fields of the tensor dataclass.

        Args:
            fn (callable): The function to apply to tensor fields.
            dataclass_fn (callable): The function to apply to TensorDataclass fields. Else use fn.
            exclude_fields (List[str]): The fields to be excluded from calling fn and dataclass_fn.
            **kwargs: Additional arguments to initialize the new TensorDataclass.

        Returns:
            cls: A new class with the same data but with a new shape.
        """
        if exclude_fields is None:
            exclude_fields = []

        field_names = [f for f in field_keys if f not in exclude_fields]
        new_fields = {}
        for f in field_names:
            v = self.__getattribute__(f)
            if v is not None:
                if is_tensordataclass(v) and dataclass_fn is not None:
                    new_fields[f] = dataclass_fn(v)
                elif isinstance(v, torch.Tensor) or is_tensordataclass(v):
                    new_fields[f] = fn(v)
        new_fields.update(kwargs)

        return cls(**new_fields)

    def is_packed(self) -> bool:
        """Returns whether the data are packed."""
        return self.__getattribute__("_packed_info") is not None

    def pack(self, valid_mask: torch.Tensor = None) -> cls:
        """Returns a new TensorDataclass with packed data.

        Args:
            valid_mask: a boolen mask used for packing the data. If none, it will try to use the
                attribute with the name of `valid_mask`.

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but everything is packed.
        """
        if self.is_packed():
            raise RuntimeError("The TensorDataclass is already packed!")

        if valid_mask is None:
            if hasattr(self, "valid_mask"):
                valid_mask = self.valid_mask
            else:
                raise RuntimeError("Please provide a `valid_mask` to pack the TensorDataclass!")
        valid_mask = valid_mask.broadcast_to(self.shape)

        def unique(x, dim=-1):
            unique, inverse, counts = torch.unique(x, return_inverse=True, return_counts=True, dim=dim)
            perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([dim]), perm.flip([dim])
            indices = inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)
            return unique, indices, counts

        # TODO(ruilongli): code is too argly here.
        offsets = torch.cumprod(
            torch.tensor(self.shape[:0:-1], dtype=torch.long, device=valid_mask.device), dim=0
        ).flip([0])
        indices = valid_mask.nonzero()[:, :-1]
        indices_sum = (indices * offsets).sum(dim=-1)
        _, u_indices, counts = unique(indices_sum)
        indices_unique = indices[u_indices]
        counts_cum = torch.cumsum(counts, dim=-1)
        start_indices = torch.nn.functional.pad(counts_cum[:-1], (1, 0), value=0)
        packed_info = torch.cat([indices_unique, start_indices[:, None], counts[:, None]], dim=-1)

        def _callback(x):
            return x[valid_mask]

        def _callback_dataclass(x):
            # TODO(ruilongli): no need to calculate packed_info recursively
            return x.pack(valid_mask)

        tensor_dataclass = self.apply_fn_to_fields(
            _callback, _callback_dataclass, _packed_info=packed_info, _shape=self.shape
        )
        # print(self.__class__)
        # print(valid_mask.nonzero())
        # print(packed_info)
        # print("-------------")
        return tensor_dataclass

    def unpack(self, shape: Tuple[int] = None, padding_value: float = 0) -> cls:
        """Unpack a tensordataclass into batched format."""
        if not self.is_packed():
            raise RuntimeError("This datatensorclass is already unpacked!")
        if shape is None:
            shape = self.__getattribute__("_shape")
        packed_info = self.__getattribute__("_packed_info")

        def _callback(x):
            output_size = torch.Size(list(shape) + [x.shape[-1]])
            out = pyrad_cuda.unpack(x, packed_info.type(torch.int), output_size)
            return out

        def _callback_dataclass(x):
            return x.unpack(shape, padding_value=padding_value)

        tensor_dataclass = self.apply_fn_to_fields(_callback, _callback_dataclass, _shape=shape, _packed_info=None)
        return tensor_dataclass

    setattr(cls, _IDENTIFER, {})
    setattr(cls, "__init__", __init__)
    # all those attributes below can be overrided.
    for attr_name in [
        "__post_init__",
        "__getitem__",
        "__setitem__",
        "__len__",
        "__bool__",
        "shape",
        "size",
        "ndim",
        "reshape",
        "flatten",
        "broadcast_to",
        "to",
        "apply_fn_to_fields",
        "is_packed",
        "pack",
        "unpack",
    ]:
        if hasattr(cls, attr_name):
            continue
        setattr(cls, attr_name, eval(attr_name))
    return cls
