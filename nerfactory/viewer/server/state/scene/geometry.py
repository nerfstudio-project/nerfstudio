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

"""Geometry to render meshes, textures, etc.
"""

from __future__ import absolute_import, division, print_function

import base64
import sys
import uuid
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Tuple

import numpy as np
import umsgpack

if sys.version_info >= (3, 0):
    UNICODE = str
    from io import BytesIO, StringIO
else:
    from StringIO import StringIO

    BytesIO = StringIO


class SceneElement:
    """Base class for objects in the scene"""

    def __init__(self):
        self.uuid = UNICODE(uuid.uuid1())

    def intrinsic_transform(self):  # pylint: disable=no-self-use
        """function for intrinsic transformation"""
        return np.identity(4)

    def lower(self, object_data: Dict[str, Any]) -> Dict[str, Any]:  # pylint: disable=no-self-use
        """Formatting contents of object dictionary, returns a dictionary of formatted content

        Args:
            object_data: object being sent via websocket
        """
        return object_data


class ReferenceSceneElement(SceneElement):
    """Generic scene element class"""

    field = None

    def lower_in_object(self, object_data: Dict[str, Any]) -> str:
        """formatting contents of object dictionary, returns a dictionary of formatted content

        Args:
            object_data: object being sent via websocket
        """
        object_data.setdefault(self.field, []).append(self.lower(object_data))
        return self.uuid


class Geometry(ReferenceSceneElement):
    """Geometry class"""

    field = "geometries"


class Material(ReferenceSceneElement):
    """Material class"""

    field = "materials"


class Texture(ReferenceSceneElement):
    """Texture class"""

    field = "textures"


class Image(ReferenceSceneElement):
    """Image class"""

    field = "images"


class Box(Geometry):
    """Box object

    Args:
        lengths: length of side of box
    """

    def __init__(self, lengths: List[float]):
        super().__init__()
        self.lengths = lengths

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "BoxGeometry",
            "width": self.lengths[0],
            "height": self.lengths[1],
            "depth": self.lengths[2],
        }


class Sphere(Geometry):
    """Sphere object

    Args:
        radius: radius of sphere
    """

    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "SphereGeometry",
            "radius": self.radius,
            "widthSegments": 20,
            "heightSegments": 20,
        }


class Ellipsoid(Sphere):
    """
    An Ellipsoid is treated as a Sphere of unit radius, with an affine
    transformation applied to distort it into the ellipsoidal shape

    Args:
        radii: the radii of epllipsoid
    """

    def __init__(self, radii: List[float]):
        super().__init__(1.0)
        self.radii = radii

    def intrinsic_transform(self):
        return np.diag(np.hstack((self.radii, 1.0)))


class PlaneGeometry(Geometry):
    """Plane Class

    Args:
        lengths: the height and width of plane
    """

    def __init__(self, lengths: List[float]):
        super().__init__()
        self.lengths = lengths

    def lower(self, object_data):
        return {"uuid": self.uuid, "type": "PlaneGeometry", "width": self.lengths[0], "height": self.lengths[1]}


class Cylinder(Geometry):
    """A cylinder of the given height and radius. By Three.js convention, the axis of
    rotational symmetry is aligned with the y-axis.

    Args:
        height: height of cylinder
        radius: radius for both top and bottom if top/bottom not given. Defaults to 1.0.
        radius_top: radius of top. Defaults to None.
        radius_bottom: radius of bottom. Defaults to None.
    """

    def __init__(
        self,
        height: int,
        radius: float = 1.0,
        radius_top: Optional[float] = None,
        radius_bottom: Optional[float] = None,
    ):
        super().__init__()
        if radius_top is not None and radius_bottom is not None:
            self.radius_top = radius_top
            self.radius_bottom = radius_bottom
        else:
            self.radius_top = radius
            self.radius_bottom = radius
        self.height = height
        self.radial_segments = 50

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "type": "CylinderGeometry",
            "radiusTop": self.radius_top,
            "radiusBottom": self.radius_bottom,
            "height": self.height,
            "radialSegments": self.radial_segments,
        }


class GenericMaterial(Material):
    """Generic Material base class

    Args:
       color: color of material. Defaults to 0xFFFFFF.
       reflectivity: reflectivity index. Defaults to 0.5.
       map: texture map of object. Defaults to None.
       transparent: transparency value. Defaults to None.
       opacity: opacity value. Defaults to 1.0.
       linewidth: width of lines. Defaults to 1.0.
       wireframe: whether to render wireframe around object. Defaults to False.
       wireframe_linewidth: width of wireframe lines. Defaults to 1.0.
       vertex_colors: whether to render vertex colors. Defaults to False.
    """

    def __init__(
        self,
        color: int = 0xFFFFFF,
        reflectivity: float = 0.5,
        map: Optional[Texture] = None,  # pylint: disable=redefined-builtin
        side: int = 2,
        transparent: Optional[bool] = None,
        opacity: float = 1.0,
        linewidth: float = 1.0,
        wireframe: bool = False,
        wireframe_linewidth: float = 1.0,
        vertex_colors: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.color = color
        self.reflectivity = reflectivity
        self.map = map
        self.side = side
        self.transparent = transparent
        self.opacity = opacity
        self.linewidth = linewidth
        self.wireframe = wireframe
        self.wireframe_linewidth = wireframe_linewidth
        self.vertex_colors = vertex_colors
        self.properties = kwargs
        self._type = self._type

    def lower(self, object_data):
        # Three.js allows a material to have an opacity which is != 1,
        # but to still be non-transparent, in which case the opacity only
        # serves to desaturate the material's color. That's a pretty odd
        # combination of things to want, so by default we just use the
        # opacity value to decide whether to set transparent to True or
        # False.
        if self.transparent is None:
            transparent = bool(self.opacity != 1)
        else:
            transparent = self.transparent
        data = {
            "uuid": self.uuid,
            "type": self._type,
            "color": self.color,
            "reflectivity": self.reflectivity,
            "side": self.side,
            "transparent": transparent,
            "opacity": self.opacity,
            "linewidth": self.linewidth,
            "wireframe": bool(self.wireframe),
            "wireframeLinewidth": self.wireframe_linewidth,
            "vertexColors": (2 if self.vertex_colors else 0),  # three.js wants an enum
        }
        data.update(self.properties)
        if self.map is not None:
            data["map"] = self.map.lower_in_object(object_data)
        return data


class MeshBasicMaterial(GenericMaterial):
    """Basic material mesh class"""

    _type = "MeshBasicMaterial"


class MeshPhongMaterial(GenericMaterial):
    """Phong material mesh class"""

    _type = "MeshPhongMaterial"


class MeshLambertMaterial(GenericMaterial):
    """Lambert material mesh class"""

    _type = "MeshLambertMaterial"


class MeshToonMaterial(GenericMaterial):
    """Toon material mesh class"""

    _type = "MeshToonMaterial"


class LineBasicMaterial(GenericMaterial):
    """Basic Line Material class"""

    _type = "LineBasicMaterial"


class PngImage(Image):
    """Png Image"""

    def __init__(self, data):
        super().__init__()
        self.data = data

    @staticmethod
    def from_file(fname: str):
        """Read in png image from file

        Args:
            fname: file name to read image from
        """
        with open(fname, "rb") as f:
            return PngImage(f.read())

    def lower(self, object_data):
        return {
            "uuid": self.uuid,
            "url": UNICODE("data:image/png;base64," + base64.b64encode(self.data).decode("ascii")),
        }


class GenericTexture(Texture):
    """Generic texture class"""

    def __init__(self, properties: Iterable):
        super().__init__()
        self.properties = properties

    def lower(self, object_data):
        data = {"uuid": self.uuid}
        data.update(self.properties)
        if "image" in data:
            image = data["image"]
            data["image"] = image.lower_in_object(object_data)
        return data


class ImageTexture(Texture):
    """Image Texture class

    Args:
        image: image represented as byte array
        wrap: wrap format. Defaults to None.
        repeat: repeat format. Defaults to None.
    """

    def __init__(self, image: bytearray, wrap: List[int] = None, repeat: List[int] = None, **kwargs):
        super().__init__()
        if wrap is None:
            wrap = [1001, 1001]
        if repeat is None:
            repeat = [1, 1]
        self.image = image
        self.wrap = wrap
        self.repeat = repeat
        self.properties = kwargs

    def lower(self, object_data):
        data = {
            "uuid": self.uuid,
            "wrap": self.wrap,
            "repeat": self.repeat,
            "image": self.image.lower_in_object(object_data),
        }
        data.update(self.properties)
        return data


class Object(SceneElement):
    """Object element within scene

    Args:
        geometry: geometry of the object
        material: material composition of object. Defaults to MeshPhongMaterial().
    """

    def __init__(self, geometry: Geometry, material: GenericMaterial = MeshPhongMaterial()):
        super().__init__()
        self.geometry = geometry
        self.material = material
        self._type = self._type

    # pylint: arguments-differ
    def lower(self):
        data = {
            "metadata": {
                "version": 4.5,
                "type": "Object",
            },
            "geometries": [],
            "materials": [],
            "object": {
                "uuid": self.uuid,
                "type": self._type,
                "geometry": self.geometry.uuid,
                "material": self.material.uuid,
                "matrix": list(self.geometry.intrinsic_transform().flatten()),
            },
        }
        self.geometry.lower_in_object(data)
        self.material.lower_in_object(data)
        return data


class Mesh(Object):
    """Mesh class"""

    _type = "Mesh"


class OrthographicCamera(SceneElement):
    """Orthographic camera class
    Args:
        left: left bounds
        right: right bounds
        top: top bounds
        bottom: bottom bounds
        near: near plane
        far: far plane
        zoom: zoom param. Defaults to 1.
    """

    def __init__(self, left: float, right: float, top: float, bottom: float, near: float, far: float, zoom: int = 1):
        super(OrthographicCamera, self).__init__()
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.near = near
        self.far = far
        self.zoom = zoom

    def lower(self, object_data):
        data = {
            "object": {
                "uuid": self.uuid,
                "type": "OrthographicCamera",
                "left": self.left,
                "right": self.right,
                "top": self.top,
                "bottom": self.bottom,
                "near": self.near,
                "far": self.far,
                "zoom": self.zoom,
            }
        }
        return data


class PerspectiveCamera(SceneElement):
    """
    The PerspectiveCamera is the default camera used by the nerfactory viewer. See
    https://threejs.org/docs/#api/en/cameras/PerspectiveCamera for more
    information.

    Args:
        fov   : Camera frustum vertical field of view, from bottom to top of view, in degrees. Default is 50.
        aspect: Camera frustum aspect ratio, usually the canvas width / canvas height. Default is 1 (square canvas).
        near  : Camera frustum near plane. Default is 0.1. The valid range is greater than 0 and less than the current
                value of the far plane. Note that, unlike for the OrthographicCamera, 0 is not a valid value for a
                PerspectiveCamera's near plane.
        far   : Camera frustum far plane. Default is 2000.
        zoom  : Gets or sets the zoom factor of the camera. Default is 1.
        film_gauge: Film size used for the larger axis. Default is 35 (millimeters). This parameter does not influence
                   the projection matrix unless .filmOffset is set to a nonzero value.
        film_offset: Horizontal off-center offset in the same unit as .filmGauge. Default is 0.
        focus: Object distance used for stereoscopy and depth-of-field effects. This parameter does not influence
               the projection matrix unless a StereoCamera is being used. Default is 10.
    """

    def __init__(
        self,
        fov: float = 50,
        aspect: float = 1,
        near: float = 0.1,
        far: float = 2000,
        zoom: int = 1,
        film_gauge: float = 35,
        film_offset: float = 0,
        focus: float = 10,
    ):
        super(PerspectiveCamera, self).__init__()
        self.fov = fov
        self.aspect = aspect
        self.far = far
        self.near = near
        self.zoom = zoom
        self.film_gauge = film_gauge
        self.film_offset = film_offset
        self.focus = focus

    def lower(self, object_data):
        data = {
            "object": {
                "uuid": self.uuid,
                "type": "PerspectiveCamera",
                "aspect": self.aspect,
                "far": self.far,
                "filmGauge": self.film_gauge,
                "filmOffset": self.film_offset,
                "focus": self.focus,
                "fov": self.fov,
                "near": self.near,
                "zoom": self.zoom,
            }
        }
        return data


def item_size(array: np.ndarray) -> int:
    """Returns the size of the 1 or 2d numpy array

    Args:
        array: np array for which to get the size of

    Raises:
        ValueError: if n dimensions is not == 1 or 2
    """
    if array.ndim == 1:
        return 1
    if array.ndim == 2:
        return array.shape[0]
    raise ValueError(f"I can only pack 1- or 2-dimensional numpy arrays, but this one has {array.ndim:d} dimensions")


def threejs_type(dtype: type) -> Tuple[str, int]:
    """Converts the np type to 3js type, returns str name and hex of equivalent 3js type

    Args:
        dtype: np type

    Raises:
        ValueError: if unsupported datatype
    """
    if dtype == np.uint8:
        return "Uint8Array", 0x12
    if dtype == np.int32:
        return "Int32Array", 0x15
    if dtype == np.uint32:
        return "Uint32Array", 0x16
    if dtype == np.float32:
        return "Float32Array", 0x17

    raise ValueError("Unsupported datatype: " + str(dtype))


def pack_numpy_array(x: np.ndarray) -> Dict[str, Any]:
    """convert numpy array to byte-based message with context, returns serialized message

    Args:
        x: array to pack into message format
    """
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    typename, extcode = threejs_type(x.dtype)
    return {
        "itemSize": item_size(x),
        "type": typename,
        "array": umsgpack.Ext(extcode, x.tobytes("F")),
        "normalized": False,
    }


def data_from_stream(stream: BinaryIO) -> str:
    """Read in data from some binary io stream and returns the serialized stream

    Args:
        stream: the stream in which data is being sent over

    Raises:
        ValueError: invalid stream type
    """
    if sys.version_info >= (3, 0):
        if isinstance(stream, BytesIO):
            data = stream.read().decode(encoding="utf-8")
        elif isinstance(stream, StringIO):
            data = stream.read()
        else:
            raise ValueError(f"Stream must be instance of StringIO or BytesIO, not {type(stream)}")
    else:
        data = stream.read()
    return data


class MeshGeometry(Geometry):
    """Basic mesh geometry class

    Args:
        contents: the stream of contents representing the mesh
        mesh_format: specification of mesh format
    """

    def __init__(self, contents: bytearray, mesh_format: str):
        super().__init__()
        self.contents = contents
        self.mesh_format = mesh_format

    def lower(self, object_data):
        return {"type": "_meshfile", "uuid": self.uuid, "format": self.mesh_format, "data": self.contents}


class ObjMeshGeometry(MeshGeometry):
    """Basic mesh geometry class

    Args:
        contents: the stream of contents representing the mesh
    """

    def __init__(self, contents: bytearray):
        super().__init__(contents, "obj")

    @staticmethod
    def from_file(fname: str) -> MeshGeometry:
        """Reading mesh geometry from file. Returns mesh geometry object

        Args:
            fname: file name to read from
        """
        with open(fname, "r", encoding="utf8") as f:
            return MeshGeometry(f.read(), "obj")

    @staticmethod
    def from_stream(f: BinaryIO) -> MeshGeometry:
        """Reads mesh geometry from io stream. Returns mesh geometry object

        Args:
            f: binary io stream to read from
        """
        return MeshGeometry(data_from_stream(f), "obj")


class DaeMeshGeometry(MeshGeometry):
    """Basic DAE mesh geometry class

    Args:
        contents: the stream of contents representing the mesh
    """

    def __init__(self, contents: bytearray):
        super().__init__(contents, "dae")

    @staticmethod
    def from_file(fname: str) -> MeshGeometry:
        """Reading mesh geometry from file. Returns mesh geometry object

        Args:
            fname: file name to read from
        """
        with open(fname, "r", encoding="utf8") as f:
            return MeshGeometry(f.read(), "dae")

    @staticmethod
    def from_stream(f: BinaryIO) -> MeshGeometry:
        """Reads mesh geometry from io stream. Returns mesh geometry object

        Args:
            f: binary io stream to read from
        """
        return MeshGeometry(data_from_stream(f), "dae")


class StlMeshGeometry(MeshGeometry):
    """Basic STL mesh geometry class

    Args:
        contents: the stream of contents representing the mesh
    """

    def __init__(self, contents: bytearray):
        super().__init__(contents, "stl")

    @staticmethod
    def from_file(fname: str) -> MeshGeometry:
        """Reading mesh geometry from file. Returns mesh geometry object

        Args:
            fname: file name to read from
        """
        with open(fname, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            _, extcode = threejs_type(np.uint8)
            encoded = umsgpack.Ext(extcode, arr.tobytes())
            return MeshGeometry(encoded, "stl")

    @staticmethod
    def from_stream(f: BinaryIO) -> MeshGeometry:
        """Reads mesh geometry from io stream. Returns mesh geometry object

        Args:
            f: binary io stream to read from
        """
        if sys.version_info >= (3, 0):
            if isinstance(f, BytesIO):
                arr = np.frombuffer(f.read(), dtype=np.uint8)
            elif isinstance(f, StringIO):
                arr = np.frombuffer(bytes(f.read(), "utf-8"), dtype=np.uint8)
            else:
                raise ValueError(f"Stream must be instance of StringIO or BytesIO, not {type(f)}")
        else:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
        _, extcode = threejs_type(np.uint8)
        encoded = umsgpack.Ext(extcode, arr.tobytes())
        return MeshGeometry(encoded, "stl")


class PlyMeshGeometry(MeshGeometry):
    """Basic PLY mesh geometry class

    Args:
        contents: the stream of contents representing the mesh
    """

    def __init__(self, contents: bytearray):
        super().__init__(contents, "ply")

    @staticmethod
    def from_file(fname: str) -> MeshGeometry:
        """Reading mesh geometry from file. Returns mesh geometry object

        Args:
            fname: file name to read from
        """
        with open(fname, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
            _, extcode = threejs_type(np.uint8)
            encoded = umsgpack.Ext(extcode, arr.tobytes())
            return MeshGeometry(encoded, "ply")

    @staticmethod
    def from_stream(f: BinaryIO) -> MeshGeometry:
        """Reads mesh geometry from io stream. Returns mesh geometry object

        Args:
            f: binary io stream to read from
        """
        if sys.version_info >= (3, 0):
            if isinstance(f, BytesIO):
                arr = np.frombuffer(f.read(), dtype=np.uint8)
            elif isinstance(f, StringIO):
                arr = np.frombuffer(bytes(f.read(), "utf-8"), dtype=np.uint8)
            else:
                raise ValueError(f"Stream must be instance of StringIO or BytesIO, not {type(f)}")
        else:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
        _, extcode = threejs_type(np.uint8)
        encoded = umsgpack.Ext(extcode, arr.tobytes())
        return MeshGeometry(encoded, "stl")


class TriangularMeshGeometry(Geometry):
    """
    A mesh consisting of an arbitrary collection of triangular faces.
    To construct one, you need to pass in a collection of vertices as an Nx3 array
    and a collection of faces as an Mx3 array. Each element of `faces`
    should be a collection of 3 indices into the `vertices` array.
    For example, to create a square made out of two adjacent triangles, we could do:
    vertices = np.array([
        [0, 0, 0],  # the first vertex is at [0, 0, 0]
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],  # The first face consists of vertices 0, 1, and 2
        [3, 0, 2]
    ])
    mesh = TriangularMeshGeometry(vertices, faces)
    """

    __slots__ = ["vertices", "faces"]

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        super().__init__()

        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.uint32)
        assert vertices.shape[1] == 3, "`vertices` must be an Nx3 array"
        assert faces.shape[1] == 3, "`faces` must be an Mx3 array"
        self.vertices = vertices
        self.faces = faces

    def lower(self, object_data) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "type": "BufferGeometry",
            "data": {
                "attributes": {"position": pack_numpy_array(self.vertices.T)},
                "index": pack_numpy_array(self.faces.T),
            },
        }


class PointsGeometry(Geometry):
    """Points Geometry class

    Args:
        position: position of the point to draw
        color: color to render the point
    """

    def __init__(self, position: np.ndarray, color: Optional[np.ndarray] = None):
        super().__init__()
        self.position = position
        self.color = color

    def lower(self, object_data) -> Dict[str, Any]:
        attrs = {"position": pack_numpy_array(self.position)}
        if self.color is not None:
            attrs["color"] = pack_numpy_array(self.color)
        return {"uuid": self.uuid, "type": "BufferGeometry", "data": {"attributes": attrs}}


class PointsMaterial(Material):
    """Points Material class

    Args:
        size: size of point to render
        color: color to render the point
        sizeAttenuation: whether to attenuate size
    """

    def __init__(self, size: float = 0.001, color: int = 0xFFFFFF, size_attenuation: bool = True):
        super().__init__()
        self.size = size
        self.color = color
        self.size_attenuation = size_attenuation

    def lower(self, object_data) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "type": "PointsMaterial",
            "color": self.color,
            "size": self.size,
            "sizeAttenuation": bool(self.size_attenuation),
            "vertexColors": 2,
        }


class Points(Object):
    """Points object class"""

    _type = "Points"


def PointCloud(position: np.ndarray, color: np.ndarray, **kwargs) -> Points:  # pylint: disable=invalid-name
    """Creates a point cloud with the specified position of points

    Args:
        position: position of points
        color: color to render point cloud
    """
    return Points(PointsGeometry(position, color), PointsMaterial(**kwargs))


class Line(Object):
    """Line class"""

    _type = "Line"


class LineSegments(Object):
    """Line Segment class"""

    _type = "LineSegments"


class LineLoop(Object):
    """Line Loop class"""

    _type = "LineLoop"


def triad(scale: float = 1.0) -> LineSegments:
    """
    A visual representation of the origin of a coordinate system, drawn as three
    lines in red, green, and blue along the x, y, and z axes. The `scale` parameter
    controls the length of the three lines.
    Returns an `Object` which can be passed to `set_object()`

    Args:
        scale: scale size of triad
    """
    return LineSegments(
        PointsGeometry(
            position=np.array([[0, 0, 0], [scale, 0, 0], [0, 0, 0], [0, scale, 0], [0, 0, 0], [0, 0, scale]])
            .astype(np.float32)
            .T,
            color=np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]])
            .astype(np.float32)
            .T,
        ),
        LineBasicMaterial(vertexColors=True),
    )


def camera(scale: float = 1.0) -> LineSegments:
    """
    A visual representation of the origin of a coordinate system, drawn as three
    lines in red, green, and blue along the x, y, and z axes. The `scale` parameter
    controls the length of the three lines.
    Returns an `Object` which can be passed to `set_object()`

    Args:
        scale: scale size of camera
    """
    return LineSegments(
        PointsGeometry(
            position=np.array([[0, 0, 0], [scale, 0, 0], [0, 0, 0], [0, scale, 0], [0, 0, 0], [0, 0, scale]])
            .astype(np.float32)
            .T,
            color=np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]])
            .astype(np.float32)
            .T,
        ),
        LineBasicMaterial(vertexColors=True),
    )
