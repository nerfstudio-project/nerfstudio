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

import logging
import sys
import threading
import time
from typing import Any, Dict

import numpy as np
import torch

from nerfactory.cameras.cameras import get_camera, get_intrinsics_from_intrinsics_matrix
from nerfactory.cameras.rays import RayBundle
from nerfactory.data.image_dataset import ImageDataset
from nerfactory.data.structs import DatasetInputs
from nerfactory.graphs.base import Graph
from nerfactory.utils import profiler
from nerfactory.utils.config import ViewerConfig
from nerfactory.utils.decorators import check_visualizer_enabled, decorate_all
from nerfactory.utils.writer import GLOBAL_BUFFER
from nerfactory.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfactory.viewer.server.visualizer import Viewer


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
    """Thread that does all the rendering calls while listening for interrupts

    Args:
        state: current visualizer state object
        graph: current checkpoint of model
        camera_ray_bundle: input rays to pass through the graph to render out
    """

    def __init__(self, state: "VisualizerState", graph: Graph, camera_ray_bundle: RayBundle):
        threading.Thread.__init__(self)
        self.state = state
        self.graph = graph
        self.camera_ray_bundle = camera_ray_bundle
        self.exc = None
        self.vis_outputs = None

    def run(self):
        """run function that renders out images given the current graph and ray bundles.
        Interlaced with a trace function that checks to see if any I/O changes were registered.
        Exits and continues program if IOChangeException thrown.
        """
        outputs = None
        try:
            with SetTrace(self.state.check_interrupt):
                outputs = self.graph.get_outputs_for_camera_ray_bundle(self.camera_ray_bundle)
        except IOChangeException as e:
            self.exc = e

        if outputs:
            self.graph.process_outputs_as_images(outputs)
            self.vis_outputs = outputs

        self.state.check_done_render = True
        self.state.check_interrupt_vis = False

    def join(self, timeout=None):
        threading.Thread.join(self)
        if self.exc:
            raise self.exc


class CheckThread(threading.Thread):
    """Thread the constantly checks for io changes and sets a flag indicating interrupt

    Args:
        state: current visualizer state object
    """

    def __init__(self, state):
        threading.Thread.__init__(self)
        self.state = state

    def run(self):
        """Run function that checks to see if any of the existing state has changed
        (e.g. camera pose/output type/resolutions).
        Sets the visualizer state flag to true to signal
        to render thread that an interrupt was registered.
        """
        self.state.check_done_render = False
        while not self.state.check_done_render:
            # check camera
            data = self.state.vis["renderingState/camera"].read()
            if data is not None:
                camera_object = data["object"]
                if self.state.prev_camera_matrix is None or not np.allclose(
                    camera_object["matrix"], self.state.prev_camera_matrix
                ):
                    self.state.check_interrupt_vis = True
                    return

            # check output type
            output_type = self.state.vis["renderingState/output_choice"].read()
            if output_type is None:
                output_type = "default"
            if self.state.prev_output_type != output_type:
                self.state.check_interrupt_vis = True
                return

            # check max render
            max_resolution = self.state.vis["renderingState/maxResolution"].read()
            if max_resolution is not None:
                if self.state.max_resolution != max_resolution:
                    self.state.check_interrupt_vis = True
                    return

            # check min render
            min_resolution = self.state.vis["renderingState/minResolution"].read()
            if min_resolution is not None:
                if self.state.min_resolution != min_resolution:
                    self.state.check_interrupt_vis = True
                    return


@decorate_all([check_visualizer_enabled])
class VisualizerState:
    """Class to hold state for visualizer variables

    Args:
        config: viewer setup configuration
    """

    def __init__(self, config: ViewerConfig):
        self.config = config

        self.vis = None
        if self.config.enable:
            zmq_url = self.config.zmq_url
            self.vis = Viewer(zmq_url=zmq_url)
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
        self.min_wait_time = 0.5  # 1.0 is on high side and will cause lag

        self.outputs_set = False

    def init_scene(self, image_dataset: ImageDataset, dataset_inputs: DatasetInputs) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            image_dataset: dataset to render in the scene
            dataset_inputs: inputs to the image dataset and ray generator
        """

        # clear the current scene
        self.vis["sceneState/sceneBounds"].delete()
        self.vis["sceneState/cameras"].delete()

        # draw the training cameras and images
        image_indices = range(len(image_dataset))
        for idx in image_indices:
            image = image_dataset[idx]["image"]
            camera = get_camera(dataset_inputs.intrinsics[idx], dataset_inputs.camera_to_world[idx], None)
            bgr = image[..., [2, 1, 0]]
            self.vis[f"sceneState/cameras/{idx:06d}"].write(camera.to_json(image=bgr, resize_shape=(100, 100)))

        # draw the scene bounds (i.e., the bounding box)
        json_ = dataset_inputs.scene_bounds.to_json()
        self.vis["sceneState/sceneBounds"].write(json_)

        # set the properties of the camera
        # self.vis["renderingState/camera"].write(json_)

        # set the main camera intrinsics to one from the dataset
        # K = camera.get_intrinsics_matrix()
        # set_persp_intrinsics_matrix(self.vis, K.double().numpy())

    def update_scene(self, step: int, graph: Graph) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            graph: the current checkpoint of the model
        """

        is_training = self.vis["renderingState/isTraining"].read()

        if is_training is None or is_training:
            # in training mode, render every few steps
            if self._is_render_step(step):
                self._render_image_in_viewer(graph)
        else:
            # in pause training mode, enter render loop with set graph
            local_step = step
            run_loop = not is_training
            while run_loop:
                if self._is_render_step(local_step):
                    self._render_image_in_viewer(graph)
                is_training = self.vis["renderingState/isTraining"].read()
                run_loop = not is_training
                local_step += 1

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.check_interrupt_vis and self.res_upscale_factor > 1:
                self.res_upscale_factor = 1
                raise IOChangeException
        return self.check_interrupt

    def _is_render_step(self, step: int, default_steps: int = 2, max_steps: int = 10) -> bool:
        """dynamically calculate when to render grapic based on resolution of image"""
        if step != 0:
            if self.res_upscale_factor == 1:
                return True
            steps_per_render_image = min(default_steps * self.res_upscale_factor, max_steps)
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

    def _send_output_to_viewer(self, outputs: Dict[str, Any]):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the graph
        """
        if not self.outputs_set:
            # set_output_options(self.vis, list(outputs.keys()))
            self.vis["renderingState/output_options"].write(list(outputs.keys()))
            self.outputs_set = True
        # gross hack to get the image key, depending on which keys the graph uses
        if self.prev_output_type == "default":
            self.prev_output_type = "rgb" if "rgb" in outputs else "rgb_fine"
        image_output = outputs[self.prev_output_type].cpu().numpy() * 255
        image = (image_output).astype("uint8")
        self.vis.set_image(image)

    def _update_viewer_stats(self, render_time, image_height) -> None:
        is_training = self.vis["renderingState/isTraining"].read()
        if is_training is None or is_training:
            eval_fps = f"{1 / render_time:.2f} fps at {image_height} res"
            self.vis["renderingState/eval_fps"].write(eval_fps)
            self.vis["renderingState/train_eta"].write(GLOBAL_BUFFER["viewer/train_eta"])
        else:
            self.vis["renderingState/eval_fps"].write("Paused")
            self.vis["renderingState/train_eta"].write("Paused")

    @profiler.time_function
    def _render_image_in_viewer(self, graph: Graph) -> None:
        """
        Draw an image using the current camera pose from the viewer.
        The image is sent of a TCP connection and then uses WebRTC to send it to the viewer.
        """
        # check and perform camera updates
        data = self.vis["renderingState/camera"].read()
        if data is None:
            return
        camera_object = data["object"]
        # hacky way to prevent overflow check to see if < 100; TODO(make less hacky)
        if self.prev_camera_matrix is not None and np.allclose(camera_object["matrix"], self.prev_camera_matrix):
            self.res_upscale_factor = min(self.res_upscale_factor * 2, 100)
        else:
            self.prev_camera_matrix = camera_object["matrix"]
            self.res_upscale_factor = 1

        # check and perform output type updates
        output_type = self.vis["renderingState/output_choice"].read()
        if output_type is None:
            output_type = "default"
        self.prev_output_type = output_type

        # check and perform min/max update
        max_resolution = self.vis["renderingState/maxResolution"].read()
        if max_resolution:
            self.max_resolution = max_resolution

        min_resolution = self.vis["renderingState/minResolution"].read()
        if min_resolution:
            self.min_resolution = min_resolution

        damped_upsacale_factor = 1 if self.res_upscale_factor < 8 else self.res_upscale_factor
        image_height = min(
            self.min_resolution * damped_upsacale_factor,
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
        camera_ray_bundle = camera.get_camera_ray_bundle(device=graph.device)
        camera_ray_bundle.num_rays_per_chunk = self.config.num_rays_per_chunk

        graph.eval()

        check_thread = CheckThread(state=self)
        render_thread = RenderThread(state=self, graph=graph, camera_ray_bundle=camera_ray_bundle)
        start_time = time.time()
        check_thread.start()
        render_thread.start()
        try:
            render_thread.join()
            check_thread.join()
        except Exception:  # pylint: disable=broad-except
            pass
        render_duration = time.time() - start_time

        graph.train()

        outputs = render_thread.vis_outputs
        if outputs is not None:
            self._send_output_to_viewer(outputs)
            self._update_viewer_stats(render_duration, image_height)


def get_default_vis() -> Viewer:
    """Returns the default Visualizer."""
    zmq_url = "tcp://0.0.0.0:6000"
    viewer = Viewer(zmq_url=zmq_url)
    return viewer
