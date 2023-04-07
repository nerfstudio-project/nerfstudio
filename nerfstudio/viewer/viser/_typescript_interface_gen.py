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

"""Generate TypeScript interfaces for all messages."""

import dataclasses
from typing import Any, ClassVar, Type, Union, get_type_hints

import numpy as onp
from typing_extensions import Literal, get_args, get_origin

from ._messages import Message

_raw_type_mapping = {
    bool: "boolean",
    float: "number",
    int: "number",
    str: "string",
    onp.ndarray: "ArrayBuffer",
    Any: "any",
}


def _get_ts_type(typ: Type) -> str:
    if get_origin(typ) is tuple:
        args = get_args(typ)
        if len(args) == 2 and args[1] == ...:
            return _get_ts_type(args[0]) + "[]"
        return "[" + ", ".join(map(_get_ts_type, args)) + "]"
    if get_origin(typ) is Literal:
        return " | ".join(
            map(
                lambda lit: repr(lit).lower() if isinstance(lit, bool) else repr(lit),
                get_args(typ),
            )
        )
    if get_origin(typ) is Union:
        return " | ".join(
            map(
                _get_ts_type,
                get_args(typ),
            )
        )

    if hasattr(typ, "__origin__"):
        typ = typ.__origin__
    if typ in _raw_type_mapping:
        return _raw_type_mapping[typ]

    assert False, f"Unsupported type: {typ}"


def generate_typescript_defs() -> str:
    """Generate TypeScript interfaces for all messages."""

    out_lines = [
        "// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.",
        "// This file should not be manually modified.",
        "",
        "// For numpy arrays, we directly serialize the underlying data buffer.",
        "type ArrayBuffer = Uint8Array;",
        "",
    ]

    message_types = Message.get_subclasses()

    # Generate interfaces for each specific message.
    for cls in message_types:
        out_lines.append(f"export interface {cls.__name__} " + "{")
        field_names = {[f.name for f in dataclasses.fields(cls)]}  # type: ignore
        for name, typ in get_type_hints(cls).items():
            if typ == ClassVar[str]:
                typ = f'"{getattr(cls, name)}"'
            elif name in field_names:
                typ = _get_ts_type(typ)
            else:
                continue
            out_lines.append(f"  {name}: {typ};")
        out_lines.append("}")
    out_lines.append("")

    # Generate union type over all messages.
    out_lines.append("export type Message = ")
    for cls in message_types:
        out_lines.append(f"  | {cls.__name__}")
    out_lines[-1] = out_lines[-1] + ";"

    return "\n".join(out_lines) + "\n"
