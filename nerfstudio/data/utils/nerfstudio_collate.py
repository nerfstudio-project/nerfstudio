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

"""
Custom collate function that includes cases for nerfstudio types.
"""

import collections
import collections.abc
import re
from typing import Callable, Dict, Union

import torch
import torch.utils.data

from nerfstudio.cameras.cameras import Cameras

# pylint: disable=implicit-str-concat
NERFSTUDIO_COLLATE_ERR_MSG_FORMAT = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts, lists or anything in {}; found {}"
)
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def nerfstudio_collate(
    batch, extra_mappings: Union[Dict[type, Callable], None] = None
):  # pylint: disable=too-many-return-statements
    r"""
    This is the default pytorch collate function, but with support for nerfstudio types. All documentation
    below is copied straight over from pytorch's default_collate function, python version 3.8.13,
    pytorch version '1.12.1+cu113'. Custom nerfstudio types are accounted for at the end, and extra
    mappings can be passed in to handle custom types. These mappings are from types: callable (types
    being like int or float or the return value of type(3.), etc). The only code before we parse for custom types that
    was changed from default pytorch was the addition of the extra_mappings argument, a find and replace operation
    from default_collate to nerfstudio_collate, and the addition of the nerfstudio_collate_err_msg_format variable.


    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, nerfstudio_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[nerfstudio_collate([V1_1, V1_2, ...]),
          nerfstudio_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # Example with a batch of `int`s:
        >>> nerfstudio_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> nerfstudio_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> nerfstudio_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> nerfstudio_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> nerfstudio_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> nerfstudio_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
    """
    if extra_mappings is None:
        extra_mappings = {}
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):  # pylint: disable=no-else-return
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel, device=elem.device)  # pylint: disable=protected-access
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        # pylint: disable=no-else-return, consider-using-in
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem.dtype))

            return nerfstudio_collate([torch.as_tensor(b) for b in batch], extra_mappings=extra_mappings)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type(
                {key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem}
            )
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: nerfstudio_collate([d[key] for d in batch], extra_mappings=extra_mappings) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [nerfstudio_collate(samples, extra_mappings=extra_mappings) for samples in transposed]

    # NerfStudio types supported below

    elif isinstance(elem, Cameras):
        # If a camera, just concatenate along the batch dimension. In the future, this may change to stacking
        assert all((isinstance(cam, Cameras) for cam in batch))
        assert all((cam.distortion_params is None for cam in batch)) or all(
            (cam.distortion_params is not None for cam in batch)
        ), "All cameras must have distortion parameters or none of them should have distortion parameters.\
            Generalized batching will be supported in the future."

        # If no batch dimension exists, then we need to stack everything and create a batch dimension on 0th dim
        if elem.shape == ():
            op = torch.stack
        # If batch dimension exists, then we need to concatenate along the 0th dimension
        else:
            op = torch.cat

        return Cameras(
            op([cameras.camera_to_worlds for cameras in batch], dim=0),
            op([cameras.fx for cameras in batch], dim=0),
            op([cameras.fy for cameras in batch], dim=0),
            op([cameras.cx for cameras in batch], dim=0),
            op([cameras.cy for cameras in batch], dim=0),
            height=op([cameras.height for cameras in batch], dim=0),
            width=op([cameras.width for cameras in batch], dim=0),
            distortion_params=op(
                [
                    cameras.distortion_params
                    if cameras.distortion_params is not None
                    else torch.zeros_like(cameras.distortion_params)
                    for cameras in batch
                ],
                dim=0,
            ),
            camera_type=op([cameras.camera_type for cameras in batch], dim=0),
            times=torch.stack(
                [cameras.times if cameras.times is not None else -torch.ones_like(cameras.times) for cameras in batch],
                dim=0,
            ),
        )

    for type_key in extra_mappings:
        if isinstance(elem, type_key):
            return extra_mappings[type_key](batch)

    raise TypeError(NERFSTUDIO_COLLATE_ERR_MSG_FORMAT.format(elem_type))
