# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
Various utilities for working with typing module.
"""

import typing
from inspect import currentframe

try:
    from typing import get_args, get_origin  # pylint: disable=unused-import
except ImportError:
    from typing_extensions import get_args, get_origin  # pylint: disable=unused-import


def get_orig_class(obj):
    """
    Returns __orig_class__, safe to call from inside __init__
    If there is no __orig_class__ it returns None
    """
    # Workaround for https://github.com/python/typing/issues/658
    # Inspired by https://github.com/Stewori/pytypes/pull/53
    try:
        return object.__getattribute__(obj, "__orig_class__")
    except AttributeError:
        cls = object.__getattribute__(obj, "__class__")
        if issubclass(cls, typing.Generic):
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
        return None
