"""
Test the viewer comms code.
"""


import random
import time

import cv2
import numpy as np
import umsgpack
from tqdm import tqdm

import nerfactory.viewer.server.geometry as g
from nerfactory.cameras import camera_paths
from nerfactory.cameras.cameras import get_camera
from nerfactory.data.datasets import Blender
from nerfactory.utils.io import get_absolute_path
from nerfactory.viewer.server.transformations import (
    get_rotation_matrix,
    get_translation_matrix,
)
from nerfactory.viewer.server.viewer_utils import get_default_vis, set_camera


def test_drawing():
    """Test drawing objects in the viewer."""
    vis = get_default_vis()
    vis.delete()

    v = vis["shapes"]
    v.set_transform(get_translation_matrix([1.0, 0, 0]))
    v["box"].set_object(g.Box([1.0, 0.2, 0.3]))
    v["box"].delete()
    v["box"].set_object(g.Box([0.1, 0.2, 0.3]))
    v["box"].set_transform(get_translation_matrix([0.05, 0.1, 0.15]))
    v["cylinder"].set_object(g.Cylinder(0.2, 0.1), g.MeshLambertMaterial(color=0x22DD22))
    v["cylinder"].set_transform(get_translation_matrix([0, 0.5, 0.1]).dot(get_rotation_matrix(-np.pi / 2, [1, 0, 0])))
    v["sphere"].set_object(g.Mesh(g.Sphere(0.15), g.MeshLambertMaterial(color=0xFF11DD)))
    v["sphere"].set_transform(get_translation_matrix([0, 1, 0.15]))
    v["ellipsoid"].set_object(g.Ellipsoid([0.3, 0.1, 0.1]))
    v["ellipsoid"].set_transform(get_translation_matrix([0, 1.5, 0.1]))

    v["transparent_ellipsoid"].set_object(
        g.Mesh(g.Ellipsoid([0.3, 0.1, 0.1]), g.MeshLambertMaterial(color=0xFFFFFF, opacity=0.5))
    )
    v["transparent_ellipsoid"].set_transform(get_translation_matrix([0, 2.0, 0.1]))

    v = vis["meshes/convex"]
    v["obj"].set_object(
        g.Mesh(g.ObjMeshGeometry.from_file(get_absolute_path("tests/viewer/data/mesh_0_convex_piece_0.obj")))
    )
    v["stl_ascii"].set_object(
        g.Mesh(g.StlMeshGeometry.from_file(get_absolute_path("tests/viewer/data/mesh_0_convex_piece_0.stl_ascii")))
    )
    v["stl_ascii"].set_transform(get_translation_matrix([0, -0.5, 0]))
    v["stl_binary"].set_object(
        g.Mesh(g.StlMeshGeometry.from_file(get_absolute_path("tests/viewer/data/mesh_0_convex_piece_0.stl_binary")))
    )
    v["stl_binary"].set_transform(get_translation_matrix([0, -1, 0]))
    v["dae"].set_object(
        g.Mesh(g.DaeMeshGeometry.from_file(get_absolute_path("tests/viewer/data/mesh_0_convex_piece_0.dae")))
    )
    v["dae"].set_transform(get_translation_matrix([0, -1.5, 0]))

    v = vis["points"]
    v.set_transform(get_translation_matrix([0, 2, 0]))
    verts = np.random.rand(3, 1000000)
    colors = verts
    v["random"].set_object(g.PointCloud(verts, colors))
    v["random"].set_transform(get_translation_matrix([-0.5, -0.5, 0]))

    v = vis["lines"]
    v.set_transform(get_translation_matrix([-2, -3, 0]))

    vertices = np.random.random((3, 10)).astype(np.float32)
    v["line_segments"].set_object(g.LineSegments(g.PointsGeometry(vertices)))

    v["line"].set_object(g.Line(g.PointsGeometry(vertices)))
    v["line"].set_transform(get_translation_matrix([0, 1, 0]))

    v["line_loop"].set_object(g.LineLoop(g.PointsGeometry(vertices)))
    v["line_loop"].set_transform(get_translation_matrix([0, 2, 0]))

    v["line_loop_with_material"].set_object(g.LineLoop(g.PointsGeometry(vertices), g.LineBasicMaterial(color=0xFF0000)))
    v["line_loop_with_material"].set_transform(get_translation_matrix([0, 3, 0]))

    colors = vertices  # Color each line by treating its xyz coordinates as RGB colors
    v["line_with_vertex_colors"].set_object(
        g.Line(g.PointsGeometry(vertices, colors), g.LineBasicMaterial(vertexColors=True))
    )
    v["line_with_vertex_colors"].set_transform(get_translation_matrix([0, 4, 0]))

    v["triad"].set_object(
        g.LineSegments(
            g.PointsGeometry(
                position=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
                .astype(np.float32)
                .T,
                color=np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]])
                .astype(np.float32)
                .T,
            ),
            g.LineBasicMaterial(vertexColors=True),
        )
    )
    v["triad"].set_transform(get_translation_matrix([0, 5, 0]))

    v["triad_function"].set_object(g.triad(0.5))
    v["triad_function"].set_transform(get_translation_matrix([0, 6, 0]))


def test_streaming():
    """Test streaming of data to the viewer."""


def test_rendering():
    """Test rendering."""


def test_camera_trajectory():
    """Test generating camera trajectory."""
    vis = get_default_vis()
    # vis.delete()

    # sample poses from a dataset
    dataset = Blender(data_directory="data/blender/lego")
    dataset_inputs = dataset.get_dataset_inputs(split="train")
    num_cameras = len(dataset_inputs.intrinsics)
    intrinsics = dataset_inputs.intrinsics
    camera_to_world = dataset_inputs.camera_to_world
    idx0, idx1 = random.sample(range(num_cameras), k=2)
    camera_a = get_camera(intrinsics[idx0], camera_to_world[idx0], None)
    camera_b = get_camera(intrinsics[idx1], camera_to_world[idx1], None)

    num_steps = 100
    fps = 30
    estimated_time = num_steps / fps
    print("estimated_time:", estimated_time)

    camera_path = camera_paths.get_interpolated_camera_path(camera_a, camera_b, steps=num_steps)

    start = time.time()
    for camera in camera_path.cameras:
        set_camera(vis, camera)
        time.sleep(1 / fps)
    time_elapsed = time.time() - start
    print("time_elapsed:", time_elapsed)


def test_send_image_stream():
    """Test sending image stream."""
    vis = get_default_vis()
    vis_background = vis["/Background"]

    video_filename = get_absolute_path("data/instant_ngp/bear/bear.MOV")
    images = []
    cap = cv2.VideoCapture(video_filename)
    ret, image = cap.read()
    while ret:
        image = image[:, :, ::-1]  # to RGB
        image = cv2.resize(image, (200, 300))
        images.append(image)
        ret, image = cap.read()
        # break

    fps = 30
    for image in tqdm(images):
        rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        encoded = umsgpack.Ext(0x12, rgba.tobytes())
        vis_background.set_image(encoded)
        time.sleep(1 / fps)


def test_perspective_camera():
    """Test perspective camera in viewer."""
    vis = get_default_vis()
    vis.set_object(g.Box([0.5, 0.5, 0.5]))
    camera = g.PerspectiveCamera(fov=90)
    vis["/Cameras/default/rotated"].set_object(camera)
    vis["/Cameras/default"].set_transform(get_translation_matrix([1, -1, 0.5]))
    vis["/Cameras/default/rotated/<object>"].set_property("position", [0, 0, 0])


if __name__ == "__main__":
    # TODO: start the TCP server
    test_drawing()
    time.sleep(5)
    test_camera_trajectory()
    test_send_image_stream()
    # test_perspective_camera()
    # TODO: close the TCP server
