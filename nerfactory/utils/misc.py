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
Miscellaneous helper code.
"""

import hashlib
import json
from math import floor, log
from typing import Any, Callable, Dict, List, Optional, Union

import torch


def get_dict_to_torch(stuff: Any, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None):
    """Set everything in the dict to the specified torch device.

    Args:
        stuff: things to convert to torch
        device: machine to put the "stuff" on
        exclude: list of keys to skip over transferring to device
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            if exclude and k in exclude:
                stuff[k] = v
            else:
                stuff[k] = get_dict_to_torch(v, device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)
    return stuff


def get_dict_to_cpu(stuff: Any):
    """Set everything in the dict to CPU.

    Args:
        stuff: things to place onto cpu
    """
    if isinstance(stuff, dict):
        for k, v in stuff.items():
            stuff[k] = get_dict_to_cpu(v)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.detach().cpu()
    return stuff


def is_not_none(var: Any) -> bool:
    """Return True if the variable var is None."""
    return not isinstance(var, type(None))


def get_masked_dict(d, mask):
    """Return a masked dictionary.
    TODO(ethan): add more asserts/checks so this doesn't have unpredictable behavior.

    Args:
        d: dict to process
        mask: mask to apply to values in dictionary
    """
    masked_dict = {}
    for key, value in d.items():
        masked_dict[key] = value[mask]
    return masked_dict


def get_hash_str_from_dict(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary. Based on
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html

    Args:
        dictionary: dict to process
    """
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


class IterableWrapper:  # pylint: disable=too-few-public-methods
    """A helper that will allow an instance of a class to return multiple kinds of iterables bound
    to different functions of that class.

    To use this, take an instance of a class. From that class, pass in the <instance>.<new_iter_function>
    and <instance>.<new_next_function> to the IterableWrapper constructor. By passing in the instance's
    functions instead of just the class's functions, the self argument should automatically be accounted
    for.

    Args:
        new_iter: function that will be called instead as the __iter__() function
        new_next: function that will be called instead as the __next__() function
        length: length of the iterable. If -1, the iterable will be infinite.


    Attributes:
        new_iter: object's pointer to the function we are calling for __iter__()
        new_next: object's pointer to the function we are calling for __next__()
        length: length of the iterable. If -1, the iterable will be infinite.
        i: current index of the iterable.

    """

    i: int

    def __init__(self, new_iter: Callable, new_next: Callable, length: int = -1):
        self.new_iter = new_iter
        self.new_next = new_next
        self.length = length

    def __next__(self):
        if self.length != -1 and self.i >= self.length:
            raise StopIteration
        self.i += 1
        return self.new_next()

    def __iter__(self):
        self.new_iter()
        self.i = 0
        return self


def human_format(num):
    """Format a number in a more human readable way

    Args:
        num: number to format
    """
    units = ["", "K", "M", "B", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(num, k)))
    return f"{(num / k**magnitude):.2f} {units[magnitude]}"


def scale_dict(dictionary: Dict[Any, Any], coefficients: Dict[str, float]) -> Dict[Any, Any]:
    """Scale a dictionary in-place given a coefficients dictionary.

    Args:
        dictionary: input dict to be scaled.
        coefficients: scalar dict config for holding coefficients.

    Returns:
        Input dict scaled by coefficients.
    """
    for key in dictionary:
        if key in coefficients:
            dictionary[key] *= coefficients[key]
    return dictionary
