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


class SetObject:
    __slots__ = ["object", "path"]

    def __init__(self, geometry_or_object, material=None, path=None):
        if isinstance(geometry_or_object, Object):
            if material is not None:
                raise (ValueError("Please supply either an Object OR a Geometry and a Material"))
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

    def lower(self):
        return {"type": "set_object", "object": self.object.lower(), "path": self.path.lower()}


class GetObject:
    __slots__ = ["path"]

    def __init__(self, path):
        self.path = path

    def lower(self):
        return {"type": "get_object", "path": self.path.lower()}


class SetTransform:
    __slots__ = ["matrix", "path"]

    def __init__(self, matrix, path):
        self.matrix = matrix
        self.path = path

    def lower(self):
        return {"type": "set_transform", "path": self.path.lower(), "matrix": list(self.matrix.T.flatten())}


class Delete:
    __slots__ = ["path"]

    def __init__(self, path):
        self.path = path

    def lower(self):
        return {"type": "delete", "path": self.path.lower()}


class SetProperty:
    __slots__ = ["path", "key", "value"]

    def __init__(self, key, value, path):
        self.key = key
        self.value = value
        self.path = path

    def lower(self):
        return {"type": "set_property", "path": self.path.lower(), "property": self.key.lower(), "value": self.value}
