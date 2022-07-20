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


from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from pyrad.viewer.server.geometry import (
    Mesh,
    MeshPhongMaterial,
    Object,
    OrthographicCamera,
    PerspectiveCamera,
    Points,
    PointsMaterial,
)
from pyrad.viewer.server.path import Path


class SetObject:  # pylint: disable=too-few-public-methods
    """Set object command class: Instantiates and sets the object according to object type

    Args:
        geometry_or_object (_type_): object to be set
        material (_type_, optional): material to be set. Defaults to None.
        path (str): path of object in tree. Defaults to None.

    Raises:
        ValueError: error if material is not set for object
    """

    __slots__ = ["object", "path"]

    def __init__(self, geometry_or_object, material=None, path=None):
        if isinstance(geometry_or_object, Object):
            if material is not None:
                raise ValueError("Please supply either an Object OR a Geometry and a Material")
            self.object = geometry_or_object
        elif isinstance(geometry_or_object, (OrthographicCamera, PerspectiveCamera)):
            self.object = geometry_or_object
        else:
            if material is None:
                material = MeshPhongMaterial()
            if isinstance(material, PointsMaterial):
                self.object = Points(geometry_or_object, material)
            else:
                self.object = Mesh(geometry_or_object, material)
        if path is not None:
            self.path = path
        else:
            self.path = Path()

    def lower(self) -> Dict[str, Any]:
        """creates properly formatted json

        Returns:
            Dict[str, Any]: json in proper format
        """
        return {"type": "set_object", "object": self.object.lower(), "path": self.path.lower()}


@dataclass
class GetObject:
    """Get object command class

    Args:
        path (str): path of object in tree. Defaults to None.
    """

    __slots__ = ["path"]

    path: str

    def lower(self) -> Dict[str, Any]:
        """creates properly formatted json with object

        Returns:
            Dict[str, Any]: json in proper format
        """
        return {"type": "get_object", "path": self.path.lower()}


@dataclass
class SetTransform:
    """Set transform command class"""

    __slots__ = ["matrix", "path"]

    matrix: np.ndarray
    path: str

    def lower(self):
        """creates properly formatted json with transform matrix

        Returns:
            Dict[str, Any]: json in proper format
        """
        return {"type": "set_transform", "path": self.path.lower(), "matrix": list(self.matrix.T.flatten())}


@dataclass
class SetOutputOptions:
    """Set output options command class"""

    __slots__ = ["output_options", "path"]

    output_options: List[str]
    path: str

    def lower(self):
        """creates properly formatted json with list of possible output options

        Returns:
            Dict[str, Any]: json in proper format
        """
        return {"type": "set_output_options", "path": self.path.lower(), "output_options": list(self.output_options)}


@dataclass
class Delete:
    """Delete current state command class"""

    __slots__ = ["path"]

    path: str

    def lower(self):
        """creates properly formatted json updating tree

        Returns:
            Dict[str, Any]: json in proper format
        """
        return {"type": "delete", "path": self.path.lower()}


@dataclass
class SetProperty:
    """Set property command class"""

    __slots__ = ["path", "key", "value"]

    key: str
    value: list
    path: str

    def lower(self):
        """creates properly formatted json setting property

        Returns:
            Dict[str, Any]: json in proper format
        """
        return {"type": "set_property", "path": self.path.lower(), "property": self.key.lower(), "value": self.value}
