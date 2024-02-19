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

"""Tensor dataclass"""

import dataclasses
from copy import deepcopy
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, TypeVar, Union

import numpy as np
import torch

TensorDataclassT = TypeVar("TensorDataclassT", bound="TensorDataclass")


class TensorDataclass:
    """@dataclass of tensors with the same size batch. Allows indexing and standard tensor ops.
    Fields that are not Tensors will not be batched unless they are also a TensorDataclass.
    Any fields that are dictionaries will have their Tensors or TensorDataclasses batched, and
    dictionaries will have their tensors or TensorDataclasses considered in the initial broadcast.
    Tensor fields must have at least 1 dimension, meaning that you must convert a field like torch.Tensor(1)
    to torch.Tensor([1])

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

    # A mapping from field-name (str): n (int)
    # Any field OR any key in a dictionary field with this name (field-name) and a corresponding
    # torch.Tensor will be assumed to have n dimensions after the batch dims. These n final dimensions
    # will remain the same shape when doing reshapes, broadcasting, etc on the tensordataclass
    _field_custom_dimensions: Dict[str, int] = {}

    def __post_init__(self) -> None:
        """Finishes setting up the TensorDataclass

        This will 1) find the broadcasted shape and 2) broadcast all fields to this shape 3)
        set _shape to be the broadcasted shape.
        """
        for k, v in self._field_custom_dimensions.items():
            assert (
                isinstance(v, int) and v > 1
            ), f"Custom dimensions must be an integer greater than 1, since 1 is the default, received {k}: {v}"

        # Shim to prevent pyright from narrowing `self` to DataclassInstance.
        self_dc = self
        if not dataclasses.is_dataclass(self_dc):
            raise TypeError("TensorDataclass must be a dataclass")

        batch_shapes = self._get_dict_batch_shapes({f.name: getattr(self, f.name) for f in dataclasses.fields(self_dc)})
        if len(batch_shapes) == 0:
            raise ValueError("TensorDataclass must have at least one tensor")
        batch_shape = torch.broadcast_shapes(*batch_shapes)

        broadcasted_fields = self._broadcast_dict_fields(
            {f.name: getattr(self, f.name) for f in dataclasses.fields(self_dc)}, batch_shape
        )
        for f, v in broadcasted_fields.items():
            object.__setattr__(self, f, v)

        object.__setattr__(self, "_shape", batch_shape)

    def _get_dict_batch_shapes(self, dict_: Dict) -> List:
        """Returns batch shapes of all tensors in a dictionary

        Args:
            dict_: The dictionary to get the batch shapes of.

        Returns:
            The batch shapes of all tensors in the dictionary.
        """
        batch_shapes = []
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                if isinstance(self._field_custom_dimensions, dict) and k in self._field_custom_dimensions:
                    batch_shapes.append(v.shape[: -self._field_custom_dimensions[k]])
                else:
                    batch_shapes.append(v.shape[:-1])
            elif isinstance(v, TensorDataclass):
                batch_shapes.append(v.shape)
            elif isinstance(v, Dict):
                batch_shapes.extend(self._get_dict_batch_shapes(v))
        return batch_shapes

    def _broadcast_dict_fields(self, dict_: Dict, batch_shape) -> Dict:
        """Broadcasts all tensors in a dictionary according to batch_shape

        Args:
            dict_: The dictionary to broadcast.

        Returns:
            The broadcasted dictionary.
        """
        new_dict = {}
        for k, v in dict_.items():
            if isinstance(v, torch.Tensor):
                # Apply field-specific custom dimensions.
                if isinstance(self._field_custom_dimensions, dict) and k in self._field_custom_dimensions:
                    new_dict[k] = v.broadcast_to(
                        (
                            *batch_shape,
                            *v.shape[-self._field_custom_dimensions[k] :],
                        )
                    )
                else:
                    new_dict[k] = v.broadcast_to((*batch_shape, v.shape[-1]))
            elif isinstance(v, TensorDataclass):
                new_dict[k] = v.broadcast_to(batch_shape)
            elif isinstance(v, Dict):
                new_dict[k] = self._broadcast_dict_fields(v, batch_shape)
            else:
                # Don't broadcast the remaining fields
                new_dict[k] = v
        return new_dict

    def __getitem__(self: TensorDataclassT, indices) -> TensorDataclassT:
        if isinstance(indices, (torch.Tensor)):
            return self._apply_fn_to_fields(lambda x: x[indices])
        if isinstance(indices, (int, slice, type(Ellipsis))):
            indices = (indices,)
        assert isinstance(indices, tuple)

        def tensor_fn(x):
            return x[indices + (slice(None),)]

        def dataclass_fn(x):
            return x[indices]

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v[indices + ((slice(None),) * custom_dims)]

        return self._apply_fn_to_fields(tensor_fn, dataclass_fn, custom_tensor_dims_fn=custom_tensor_dims_fn)

    def __setitem__(self, indices, value) -> NoReturn:
        raise RuntimeError("Index assignment is not supported for TensorDataclass")

    def __len__(self) -> int:
        if len(self._shape) == 0:
            raise TypeError("len() of a 0-d tensor")
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
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """
        if isinstance(shape, int):
            shape = (shape,)

        def tensor_fn(x):
            return x.reshape((*shape, x.shape[-1]))

        def dataclass_fn(x):
            return x.reshape(shape)

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v.reshape((*shape, *v.shape[-custom_dims:]))

        return self._apply_fn_to_fields(tensor_fn, dataclass_fn, custom_tensor_dims_fn=custom_tensor_dims_fn)

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
            shape: The new shape of the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """

        def custom_tensor_dims_fn(k, v):
            custom_dims = self._field_custom_dimensions[k]
            return v.broadcast_to((*shape, *v.shape[-custom_dims:]))

        return self._apply_fn_to_fields(
            lambda x: x.broadcast_to((*shape, x.shape[-1])), custom_tensor_dims_fn=custom_tensor_dims_fn
        )

    def to(self: TensorDataclassT, device) -> TensorDataclassT:
        """Returns a new TensorDataclass with the same data but on the specified device.

        Args:
            device: The device to place the tensor dataclass.

        Returns:
            A new TensorDataclass with the same data but on the specified device.
        """
        return self._apply_fn_to_fields(lambda x: x.to(device))

    def pin_memory(self: TensorDataclassT) -> TensorDataclassT:
        """Pins the tensor dataclass memory

        Returns:
            TensorDataclass: A new TensorDataclass with the same data but pinned.
        """
        return self._apply_fn_to_fields(lambda x: x.pin_memory())

    def _apply_fn_to_fields(
        self: TensorDataclassT,
        fn: Callable,
        dataclass_fn: Optional[Callable] = None,
        custom_tensor_dims_fn: Optional[Callable] = None,
    ) -> TensorDataclassT:
        """Applies a function to all fields of the tensor dataclass.

        TODO: Someone needs to make a high level design choice for whether not not we want this
        to apply the function to any fields in arbitray superclasses. This is an edge case until we
        upgrade to python 3.10 and dataclasses can actually be subclassed with vanilla python and no
        janking, but if people try to jank some subclasses that are grandchildren of TensorDataclass
        (imagine if someone tries to subclass the RayBundle) this will matter even before upgrading
        to 3.10 . Currently we aren't going to be able to work properly for grandchildren, but you
        want to use self.__dict__ if you want to apply this to grandchildren instead of our dictionary
        from dataclasses.fields(self) as we do below and in other places.

        Args:
            fn: The function to apply to tensor fields.
            dataclass_fn: The function to apply to TensorDataclass fields.

        Returns:
            A new TensorDataclass with the same data but with a new shape.
        """

        self_dc = self
        assert dataclasses.is_dataclass(self_dc)

        new_fields = self._apply_fn_to_dict(
            {f.name: getattr(self, f.name) for f in dataclasses.fields(self_dc)},
            fn,
            dataclass_fn,
            custom_tensor_dims_fn,
        )

        return dataclasses.replace(self_dc, **new_fields)

    def _apply_fn_to_dict(
        self,
        dict_: Dict,
        fn: Callable,
        dataclass_fn: Optional[Callable] = None,
        custom_tensor_dims_fn: Optional[Callable] = None,
    ) -> Dict:
        """A helper function for _apply_fn_to_fields, applying a function to all fields of dict_

        Args:
            dict_: The dictionary to apply the function to.
            fn: The function to apply to tensor fields.
            dataclass_fn: The function to apply to TensorDataclass fields.

        Returns:
            A new dictionary with the same data but with a new shape. Will deep copy"""

        field_names = dict_.keys()
        new_dict = {}
        for f in field_names:
            v = dict_[f]
            if v is not None:
                if isinstance(v, TensorDataclass) and dataclass_fn is not None:
                    new_dict[f] = dataclass_fn(v)
                # This is the case when we have a custom dimensions tensor
                elif (
                    isinstance(v, torch.Tensor)
                    and f in self._field_custom_dimensions
                    and custom_tensor_dims_fn is not None
                ):
                    new_dict[f] = custom_tensor_dims_fn(f, v)
                elif isinstance(v, (torch.Tensor, TensorDataclass)):
                    new_dict[f] = fn(v)
                elif isinstance(v, Dict):
                    new_dict[f] = self._apply_fn_to_dict(v, fn, dataclass_fn)
                else:
                    new_dict[f] = deepcopy(v)

        return new_dict
