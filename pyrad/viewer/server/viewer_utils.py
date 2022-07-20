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

"""Code to interface with the `vis/` (the JS visualizer).
"""

import copy
import logging
import random
import sys
import threading
import time
from typing import List

import numpy as np
import torch
from pyrad.utils.decorators import check_visualizer_enabled, decorate_all

import pyrad.viewer.server.cameras as c
import pyrad.viewer.server.geometry as g
from pyrad.cameras.cameras import Camera, get_camera, get_intrinsics_from_intrinsics_matrix
from pyrad.cameras.rays import RayBundle
from pyrad.data.image_dataset import ImageDataset
from pyrad.data.structs import DatasetInputs
from pyrad.graphs.base import Graph
from pyrad.utils import profiler
from pyrad.utils.config import ViewerConfig
from pyrad.viewer.server.transformations import get_translation_matrix
from pyrad.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from pyrad.viewer.server.visualizer import Viewer


class IOChangeException(Exception):
    """Basic camera exception to interrupt visualizer"""


class SetTrace:
    """Basic trace function"""

    def __init__(self, func):
        self.func = func

    def __enter__(self):
        sys.settrace(self.func)
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.settrace(None)


class RenderThread(threading.Thread):
    """Thread that does all the rendering calls while listening for interrupts"""

    def __init__(self, state: "VisualizerState", graph: Graph, camera_ray_bundle: RayBundle):
        threading.Thread.__init__(self)
        self.state = state
        self.graph = graph
        self.camera_ray_bundle = camera_ray_bundle
        self.exc = None

    def run(self):
        outputs = None
        try:
            with SetTrace(self.state.check_interrupt):
                outputs = self.graph.get_outputs_for_camera_ray_bundle(self.camera_ray_bundle)
        except IOChangeException as e:
            self.exc = e

        if outputs:
            self.graph.process_outputs_as_images(outputs)

        self.state.check_done_render = True
        self.state.check_interrupt_vis = False

    def join(self, timeout=None):
        threading.Thread.join(self)
        if self.exc:
            raise self.exc


class CheckThread(threading.Thread):
    """Thread the constantly checks for io changes and sets a flag indicating interrupt"""

    def __init__(self, state):
        threading.Thread.__init__(self)
        self.state = state

    def run(self):
        self.state.check_done_render = False
        is_interrupt = False
        while not self.state.check_done_render:
            # check camera
            data = self.state.vis["/Cameras/Main Camera"].get_object()
            if data is not None:
                camera_object = data["object"]["object"]
                if self.state.prev_camera_matrix is None or not np.allclose(
                    camera_object["matrix"], self.state.prev_camera_matrix
                ):
                    is_interrupt = True
                    self.state.prev_camera_matrix = camera_object["matrix"]

            # check output type
            data = self.state.vis["/Output Type"].get_object()
            if data is None:
                output_type = "default"
            else:
                output_type = data["output_type"]
            if self.state.prev_output_type != output_type:
                is_interrupt = True

            # check max render
            data = self.state.vis["/Max Resolution"].get_object()
            if data is not None:
                max_resolution = int(data["max_resolution"])
                if self.state.max_resolution != max_resolution:
                    is_interrupt = True
                    self.state.max_resolution = max_resolution

            # check min render
            data = self.state.vis["/Min Resolution"].get_object()
            if data is not None:
                min_resolution = int(data["min_resolution"])
                if self.state.min_resolution != min_resolution:
                    is_interrupt = True
                    self.state.min_resolution = min_resolution

            self.state.check_interrupt_vis = is_interrupt
            time.sleep(0.001)


@decorate_all([check_visualizer_enabled])
class VisualizerState:
    """Class to hold state for visualizer variables"""

    def __init__(self, config: ViewerConfig):
        self.config = config

        self.vis = None
        if self.config.enable:
            zmq_url = self.config.zmq_url
            self.vis = Viewer(zmq_url=zmq_url)
            self.vis.delete()
        else:
            logging.info("Continuing without viewer.")

        # visualizer specific variables
        self.prev_camera_matrix = None
        self.prev_output_type = "default"
        self.min_resolution = 50
        self.max_resolution = 1000
        self.res_upscale_factor = 1
        self.check_interrupt_vis = False
        self.check_done_render = True
        self.last_render_time = time.time()
        self.min_wait_time = 5.0

        self.outputs_set = False

    def init_scene(self, image_dataset: ImageDataset, dataset_inputs: DatasetInputs) -> None:
        """initializes the scene with the datasets"""
        self._draw_scene_in_viewer(image_dataset, dataset_inputs)

    def update_scene(self, step: int, graph: Graph) -> None:
        """updates the scene based on the graph weights"""
        data = self.vis["/Training State"].get_object()

        if data is None or not data["training_state"]:
            # in training mode, render every few steps
            if self._is_render_step(step):
                self._render_image_in_viewer(graph)
        else:
            # in pause training mode, enter render loop with set graph
            local_step = step
            run_loop = data["training_state"]
            while run_loop:
                if self._is_render_step(local_step):
                    self._render_image_in_viewer(graph)
                data = self.vis["/Training State"].get_object()
                run_loop = data["training_state"]
                local_step += 1

    def check_interrupt(self, frame, event, arg):
        """raises interrupt when flag has been set and not already on lowest resolution"""
        if event == "line":
            if self.check_interrupt_vis and self.res_upscale_factor > 1:
                self.res_upscale_factor = 1
                raise IOChangeException
        return self.check_interrupt

    def _is_render_step(self, step: int, default_steps: int = 5) -> bool:
        """dynamically calculate when to render grapic based on resolution of image"""
        if step != 0:
            if self.res_upscale_factor == 1:
                return True
            steps_per_render_image = min(default_steps * self.res_upscale_factor, 100)
            steps_condition = step % steps_per_render_image == 0
            if steps_condition:
                if self.res_upscale_factor > 3:
                    if time.time() - self.last_render_time >= self.min_wait_time:
                        self.last_render_time = time.time()
                        return True  # if higher res, and minimum wait time achieved
                    return False  # if higher res, and minimum wait time NOT achieved
                self.last_render_time = time.time()
                return True  # if not higher res, but steps met
        return False  # if init

    def _draw_scene_in_viewer(self, image_dataset: ImageDataset, dataset_inputs: DatasetInputs) -> None:
        """Draw some images and the scene aabb in the viewer."""
        indices = random.sample(range(len(image_dataset)), k=10)
        for idx in indices:
            image = image_dataset[idx]["image"]
            camera = get_camera(dataset_inputs.intrinsics[idx], dataset_inputs.camera_to_world[idx], None)
            pose = camera.get_camera_to_world().double().numpy()
            K = camera.get_intrinsics_matrix().double().numpy()
            draw_camera_frustum(
                self.vis,
                image=(image.double().numpy() * 255.0),
                pose=pose,
                K=K,
                height=1.0,
                name=f"image_dataset/{idx:06d}",
                displayed_focal_length=0.5,
                realistic=False,
            )
        aabb = dataset_inputs.scene_bounds.aabb
        draw_aabb(self.vis, aabb, name="dataset_inputs_train/scene_bounds/aabb")

        # set the main camera intrinsics to one from the dataset
        K = camera.get_intrinsics_matrix()
        set_persp_intrinsics_matrix(self.vis, K.double().numpy())

    @profiler.time_function
    def _render_image_in_viewer(self, graph: Graph) -> None:
        """
        Draw an image using the current camera pose from the viewer.
        The image is sent of a TCP connection and then uses WebRTC to send it to the viewer.
        """
        # check and perform camera updates
        data = self.vis["/Cameras/Main Camera"].get_object()
        if data is None:
            return
        camera_object = data["object"]["object"]
        # hacky way to prevent overflow check to see if < 100; TODO(make less hacky)
        if self.prev_camera_matrix is not None and np.allclose(camera_object["matrix"], self.prev_camera_matrix):
            self.res_upscale_factor = min(self.res_upscale_factor * 2, 100)
        else:
            self.prev_camera_matrix = camera_object["matrix"]
            self.res_upscale_factor = 1

        # check and perform output type updates
        data = self.vis["/Output Type"].get_object()
        if data is None:
            output_type = "default"
        else:
            output_type = data["output_type"]
        self.prev_output_type = output_type

        # check and perform min/max update
        data = self.vis["/Max Resolution"].get_object()
        if data is not None:
            self.max_resolution = int(data["max_resolution"])

        data = self.vis["/Min Resolution"].get_object()
        if data is not None:
            self.min_resolution = int(data["min_resolution"])

        image_height = min(
            self.min_resolution * self.res_upscale_factor,
            self.max_resolution,
        )
        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            camera_object, image_height=image_height
        )

        camera_to_world = camera_to_world_h[:3, :]
        camera_to_world = torch.stack(
            [
                camera_to_world[0, :],
                camera_to_world[2, :],
                camera_to_world[1, :],
            ],
            dim=0,
        )
        intrinsics = get_intrinsics_from_intrinsics_matrix(intrinsics_matrix)
        camera = get_camera(intrinsics, camera_to_world)
        camera_ray_bundle = camera.get_camera_ray_bundle(device=graph.get_device())
        camera_ray_bundle.num_rays_per_chunk = self.config.num_rays_per_chunk

        graph.eval()
        check_thread = CheckThread(state=self)
        render_thread = RenderThread(state=self, graph=graph, camera_ray_bundle=camera_ray_bundle)
        check_thread.start()
        render_thread.start()

        try:
            render_thread.join()
        except Exception:  # pylint: disable=broad-except
            pass

        try:
            check_thread.join()
        except Exception:  # pylint: disable=broad-except
            pass

        graph.train()
        outputs = graph.vis_outputs
        if outputs is not None:
            if not self.outputs_set:
                set_output_options(self.vis, list(outputs.keys()))
                self.outputs_set = True
            # gross hack to get the image key, depending on which keys the graph uses
            if output_type == "default":
                output_type = "rgb" if "rgb" in outputs else "rgb_fine"
            image_output = outputs[output_type].cpu().numpy() * 255
            image = (image_output).astype("uint8")
            self.vis["/Cameras/Main Camera"].set_image(image)


def get_default_vis() -> Viewer:
    """Returns the default Visualizer."""
    zmq_url = "tcp://0.0.0.0:6000"
    viewer = Viewer(zmq_url=zmq_url)
    return viewer


def set_output_options(vis: Viewer, output_options: List[str]):
    """Sets the possible list of output options for user to toggle"""
    vis["output_options"].set_output_options(output_options)


def show_box_test(vis: Viewer):
    """Simple test to draw a box and make sure everything is working."""
    vis["box"].set_object(g.Box([1.0, 1.0, 1.0]), material=g.MeshPhongMaterial(color=0xFF0000))


def show_ply(vis: Viewer, ply_path: str, name: str = "ply", color=None):
    """Show the PLY file in the 3D viewer. Specify the full filename as input."""
    assert ply_path.endswith(".ply")
    if color:
        material = g.MeshPhongMaterial(color=color)
    else:
        material = g.MeshPhongMaterial(vertexColors=True)
    vis[name].set_object(g.PlyMeshGeometry.from_file(ply_path), material)


def show_obj(vis: Viewer, obj_path: str, name: str = "obj", color=None):
    """Show the PLY file in the 3D viewer. Specify the full filename as input."""
    assert obj_path.endswith(".obj")
    if color:
        material = g.MeshPhongMaterial(color=color)
    else:
        material = g.MeshPhongMaterial(vertexColors=True)
    vis[name].set_object(g.ObjMeshGeometry.from_file(obj_path), material)


def draw_camera_frustum(
    vis: Viewer,
    image=np.random.rand(100, 100, 3) * 255.0,
    pose=get_translation_matrix([0, 0, 0]),
    K=None,
    name="0000000",
    displayed_focal_length=None,
    shift_forward=None,
    height=None,
    realistic=True,
):
    """Draw the camera in the scene."""

    assert K[0, 0] == K[1, 1]
    focal_length = K[0, 0]
    pp_w = K[0, 2]
    pp_h = K[1, 2]

    if displayed_focal_length:
        assert height is None or not realistic
    if height:
        assert displayed_focal_length is None or not realistic

    if height:
        dfl = height / (2.0 * (pp_h / focal_length))
        width = 2.0 * (pp_w / focal_length) * dfl
        if displayed_focal_length is None:
            displayed_focal_length = dfl
    elif displayed_focal_length:
        width = 2.0 * (pp_w / focal_length) * displayed_focal_length
        height = 2.0 * (pp_h / focal_length) * displayed_focal_length
    else:
        assert not realistic

    if pose.shape == (3, 4):
        pose = np.concatenate([pose, np.zeros_like(pose[:1])], axis=0)
        pose[3, 3] = 1.0

    # draw the frustum
    g_frustum = c.frustum(scale=1.0, focal_length=displayed_focal_length, width=width, height=height)
    vis[name + "/frustum"].set_object(g_frustum)
    if not realistic:
        vis[name + "/frustum"].set_transform(get_translation_matrix([0, 0, displayed_focal_length]))

    # draw the image plane
    g_image_plane = c.ImagePlane(image, width=width, height=height)
    vis[name + "/image_plane"].set_object(g_image_plane)
    if realistic:
        vis[name + "/image_plane"].set_transform(get_translation_matrix([0, 0, -displayed_focal_length]))

    if shift_forward:
        matrix = get_translation_matrix([0, 0, displayed_focal_length])
        matrix2 = get_translation_matrix([0, 0, -shift_forward])
        vis[name + "/frustum"].set_transform(matrix2 @ matrix)
        vis[name + "/image_plane"].set_transform(matrix2)

    # set the transform of the camera
    vis[name].set_transform(pose)


def set_persp_intrinsics_matrix(vis, K):
    pp_w = K[0, 2]
    pp_h = K[1, 2]
    assert K[0, 0] == K[1, 1]
    focal_length = K[0, 0]
    x = pp_h / (focal_length)
    fov = 2.0 * np.arctan(x) * (180.0 / np.pi)
    vis["/Cameras/Main Camera/<object>"].set_property("fov", fov)
    vis["/Cameras/Main Camera/<object>"].set_property("aspect", float(pp_w / pp_h))  # three.js expects width/height


def set_persp_pose(vis, pose, colmap=True):
    pose_processed = copy.deepcopy(pose)
    if colmap:
        pose_processed[:, 1:3] *= -1
    vis["/Cameras/Main Camera/<object>"].set_transform(pose_processed)


def set_persp_camera(vis, pose, K, colmap=True):
    """Assumes simple pinhole model for intrinsics.
    Args:
        colmap: whether to use the colmap camera coordinate convention or not
    """
    set_persp_intrinsics_matrix(vis, K)
    set_persp_pose(vis, pose, colmap=colmap)


def set_camera(vis, camera: Camera):
    pose = camera.get_camera_to_world_h()
    K = camera.get_intrinsics_matrix()
    set_persp_camera(vis, pose=pose.double().numpy(), K=K.double().numpy())


def draw_aabb(vis, aabb, name="aabb"):
    """Draw the axis-aligned bounding box."""
    lengths = aabb[1] - aabb[0]
    vis[name].set_object(g.Box(lengths.tolist()), material=g.MeshPhongMaterial(color=0xFF0000, opacity=0.1))
    center = aabb[0] + lengths / 2.0
    vis[name].set_transform(get_translation_matrix(center.tolist()))
