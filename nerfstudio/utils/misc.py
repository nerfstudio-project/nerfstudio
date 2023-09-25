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

"""
Miscellaneous helper code.
"""


from inspect import currentframe
import typing
import platform
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import warnings

import torch

T = TypeVar("T")
TKey = TypeVar("TKey")


def get_dict_to_torch(stuff: T, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None) -> T:
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


def get_dict_to_cpu(stuff: T) -> T:
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


def get_masked_dict(d: Dict[TKey, torch.Tensor], mask) -> Dict[TKey, torch.Tensor]:
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


class IterableWrapper:
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


def step_check(step, step_size, run_at_zero=False) -> bool:
    """Returns true based on current step and step interval."""
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0


def update_avg(prev_avg: float, new_val: float, step: int) -> float:
    """helper to calculate the running average

    Args:
        prev_avg (float): previous average value
        new_val (float): new value to update the average with
        step (int): current step number

    Returns:
        float: new updated average
    """
    return (step * prev_avg + new_val) / (step + 1)


def strtobool(val) -> bool:
    """Cheap replacement for `distutils.util.strtobool()` which is deprecated
    FMI https://stackoverflow.com/a/715468
    """
    return val.lower() in ("yes", "y", "true", "t", "on", "1")


def torch_compile(*args, **kwargs) -> Any:
    """
    Safe torch.compile with backward compatibility for PyTorch 1.x
    """
    if not hasattr(torch, "compile"):
        # Backward compatibility for PyTorch 1.x
        warnings.warn(
            "PyTorch 1.x will no longer be supported by Nerstudio. Please upgrade to PyTorch 2.x.", DeprecationWarning
        )
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        else:
            return torch.jit.script
    elif platform.system() == "Windows":
        # torch.compile is not supported on Windows
        # https://github.com/orgs/pytorch/projects/27
        # TODO: @jkulhanek, remove this once torch.compile is supported on Windows
        warnings.warn(
            "Windows does not yet support torch.compile and the performance will be affected.", RuntimeWarning
        )
        if args and isinstance(args[0], torch.nn.Module):
            return args[0]
        else:
            return lambda x: x
    else:
        return torch.compile(*args, **kwargs)


def get_orig_class(obj, default=None):
    """Returns the __orig_class__ class of `obj` even when it is not initialized in __init__ (Python>=3.8).

    Workaround for https://github.com/python/typing/issues/658.
    Inspired by https://github.com/Stewori/pytypes/pull/53.
    """
    try:
        return object.__getattribute__(obj, "__orig_class__")
    except AttributeError:
        cls = object.__getattribute__(obj, "__class__")
        try:
            is_type_generic = isinstance(cls, typing.GenericMeta)  # type: ignore
        except AttributeError:  # Python 3.8
            is_type_generic = issubclass(cls, typing.Generic)
        if is_type_generic:
            frame = currentframe().f_back.f_back  # type: ignore
            try:
                while frame:
                    try:
                        res = frame.f_locals["self"]
                        if res.__origin__ is cls:
                            return res
                    except (KeyError, AttributeError):
                        frame = frame.f_back
            finally:
                del frame
        return default
