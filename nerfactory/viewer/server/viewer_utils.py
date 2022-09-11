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

import enum
import logging
import sys
import threading
import time
from typing import Any, Dict

import numpy as np
import torch
from rich import print  # pylint: disable=redefined-builtin

from nerfactory.cameras.cameras import Cameras
from nerfactory.cameras.rays import RayBundle
from nerfactory.configs import base as cfg
from nerfactory.datamanagers.datasets import InputDataset
from nerfactory.models.base import Model
from nerfactory.utils import profiler, visualization, writer
from nerfactory.utils.decorators import check_visualizer_enabled, decorate_all
from nerfactory.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfactory.viewer.server.subprocess import run_viewer_bridge_server_as_subprocess
from nerfactory.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfactory.viewer.server.visualizer import Viewer


class OutputTypes(str, enum.Enum):
    """Noncomprehsnive list of output render types"""

    INIT = "init"
    RGB = "rgb"
    RGB_FINE = "rgb_fine"


class ColormapTypes(str, enum.Enum):
    """Noncomprehsnive list of colormap render types"""

    INIT = "init"
    DEFAULT = "default"
    TURBO = "turbo"
    DEPTH = "depth"
    SEMANTIC = "semantic"
    BOOLEAN = "boolean"


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

    def __init__(self, state: "VisualizerState", graph: Model, camera_ray_bundle: RayBundle):
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
                output_type = OutputTypes.INIT
            if self.state.prev_output_type != output_type:
                self.state.check_interrupt_vis = True
                return

            # check colormap type
            colormap_type = self.state.vis["renderingState/colormap_choice"].read()
            if colormap_type is None:
                colormap_type = ColormapTypes.INIT
            if self.state.prev_colormap_type != colormap_type:
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

    def __init__(self, config: cfg.ViewerConfig):
        self.config = config

        self.vis = None
        self.viewer_url = None
        if self.config.enable:
            if self.config.launch_bridge_server:
                # start the viewer bridge server
                zmq_port = int(self.config.zmq_url.split(":")[-1])
                websocket_port = self.config.websocket_port
                self.config.log_filename.parent.mkdir(exist_ok=True)
                run_viewer_bridge_server_as_subprocess(
                    zmq_port, websocket_port, log_filename=str(self.config.log_filename)
                )
                # TODO(ethan): move this into the writer such that it's at the bottom
                # of the logging stack and easy to see and click
                # TODO(ethan): log the output of the viewer bridge server in a file where the training logs go
                print("\n")
                self.viewer_url = (
                    f"https://viewer.nerfactory.com/branch/master/?websocket_url=localhost:{websocket_port}"
                )
                viewer_url_local = f"http://localhost:4000/?websocket_url=localhost:{websocket_port}"
                pub_open_viewer_instructions_string = f"[Public] Open the viewer at {self.viewer_url}"
                dev_open_viewer_instructions_string = f"[Local] Open the viewer at {viewer_url_local}"
                print("-" * len(pub_open_viewer_instructions_string))
                print(pub_open_viewer_instructions_string)
                print(dev_open_viewer_instructions_string)
                print("-" * len(pub_open_viewer_instructions_string))
                print("\n")
            self.vis = Viewer(zmq_url=self.config.zmq_url)
        else:
            logging.info("Continuing without viewer.")

        # visualizer specific variables
        self.prev_camera_matrix = None
        self.prev_output_type = OutputTypes.INIT
        self.prev_colormap_type = ColormapTypes.INIT
        self.output_type_changed = True
        self.min_resolution = 50
        self.max_resolution = 1000
        self.res_upscale_factor = 1
        self.check_interrupt_vis = False
        self.check_done_render = True
        self.last_render_time = time.time()
        self.min_wait_time = 0.5  # 1.0 is on high side and will cause lag
        self.step = 0

        self.outputs_set = False

    def init_scene(self, dataset: InputDataset, start_train=True) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            start_train: whether to start train when viewer init;
                if False, only displays dataset until resume train is toggled
        """
        # clear the current scene
        self.vis["sceneState/sceneBounds"].delete()
        self.vis["sceneState/cameras"].delete()

        # draw the training cameras and images
        image_indices = range(len(dataset))
        for idx in image_indices:
            image = dataset[idx]["image"]
            bgr = image[..., [2, 1, 0]]
            camera_json = dataset.dataset_inputs.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            self.vis[f"sceneState/cameras/{idx:06d}"].write(camera_json)

        # draw the scene bounds (i.e., the bounding box)
        json_ = dataset.dataset_inputs.scene_bounds.to_json()
        self.vis["sceneState/sceneBounds"].write(json_)

        # set the initial state whether to train or not
        self.vis["renderingState/isTraining"].write(start_train)

        # set the properties of the camera
        # self.vis["renderingState/camera"].write(json_)

        # set the main camera intrinsics to one from the dataset
        # K = camera.get_intrinsics_matrix()
        # set_persp_intrinsics_matrix(self.vis, K.double().numpy())

    def update_scene(self, step: int, graph: Model) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            graph: the current checkpoint of the model
        """

        is_training = self.vis["renderingState/isTraining"].read()
        self.step = step

        if is_training is None or is_training:
            # in training mode, render every few steps
            if self._is_render_step(step):
                self._render_image_in_viewer(graph)
        else:
            # in pause training mode, enter render loop with set graph
            local_step = step
            run_loop = not is_training
            while run_loop:
                if self._is_render_step(local_step) and step > 0:
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
        """dynamically calculate when to render grapic based on resolution of image

        Args:
            step: current train iteration step
            default_steps: base multiple number of steps
            max_steps: maximum number of steps in between renders
        """
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

    def _apply_colormap(self, outputs: Dict[str, Any], stuff_colors: torch.Tensor = None, eps=1e-6):
        """Determines which colormap to use based on set colormap type

        Args:
            outputs: the output tensors for which to apply colormaps on
            stuff_colors: is only set if colormap is for semantics. Defaults to None.
            eps: epsilon to handle floating point comparisons
        """
        # default for rgb images
        if self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[self.prev_output_type].shape[-1] == 3:
            return outputs[self.prev_output_type]

        # rendering depth outputs
        if self.prev_colormap_type == ColormapTypes.DEPTH or (
            self.prev_colormap_type == ColormapTypes.DEFAULT
            and outputs[self.prev_output_type].dtype == torch.float
            and (torch.max(outputs[self.prev_output_type]) - 1.0) > eps  # handle floating point arithmetic
        ):
            return visualization.apply_depth_colormap(
                outputs[self.prev_output_type], accumulation=outputs["accumulation"]
            )

        # rendering accumulation outputs
        if self.prev_colormap_type == ColormapTypes.TURBO or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[self.prev_output_type].dtype == torch.float
        ):
            return visualization.apply_colormap(outputs[self.prev_output_type])

        # rendering semantic outputs
        if self.prev_colormap_type == ColormapTypes.SEMANTIC or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[self.prev_output_type].dtype == torch.int
        ):
            logits = outputs[self.prev_output_type]
            labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)  # type: ignore
            assert stuff_colors is not None
            return stuff_colors[labels]

        # rendering boolean outputs
        if self.prev_colormap_type == ColormapTypes.BOOLEAN or (
            self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[self.prev_output_type].dtype == torch.bool
        ):
            return visualization.apply_boolean_colormap(outputs[self.prev_output_type])

        raise NotImplementedError

    def _send_output_to_viewer(self, outputs: Dict[str, Any], stuff_colors: torch.Tensor = None, eps=1e-6):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the graph
            stuff_colors: is only set if colormap is for semantics. Defaults to None.
            eps: epsilon to handle floating point comparisons
        """
        if not self.outputs_set:
            self.vis["renderingState/output_options"].write(list(outputs.keys()))
            self.outputs_set = True
        # gross hack to get the image key, depending on which keys the graph uses
        if self.prev_output_type == OutputTypes.INIT:
            self.prev_output_type = OutputTypes.RGB if OutputTypes.RGB in outputs else OutputTypes.RGB_FINE
        # re-register colormaps and send to viewer
        if self.output_type_changed or self.prev_colormap_type == ColormapTypes.INIT:
            self.prev_colormap_type = ColormapTypes.DEFAULT
            colormap_options = [ColormapTypes.DEFAULT]
            if (
                outputs[self.prev_output_type].shape[-1] != 3
                and outputs[self.prev_output_type].dtype == torch.float
                and (torch.max(outputs[self.prev_output_type]) - 1.0) <= eps  # handle floating point arithmetic
            ):
                # accumulation can also include depth
                colormap_options.extend(["depth"])
            self.output_type_changed = False
            self.vis["renderingState/colormap_choice"].write(self.prev_colormap_type)
            self.vis["renderingState/colormap_options"].write(colormap_options)
        selected_output = (self._apply_colormap(outputs, stuff_colors) * 255).type(torch.uint8)
        image = selected_output.cpu().numpy()
        self.vis.set_image(image)

    def _update_viewer_stats(self, render_time: float, num_rays: int, image_height: int) -> None:
        """Function that calculates and populates all the rendering statistics accordingly

        Args:
            render_time: total time spent rendering current view
            num_rays: number of rays rendered
            image_height: resolution of the current view
        """
        writer.put_time(
            name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=self.step, avg_over_steps=True
        )
        is_training = self.vis["renderingState/isTraining"].read()
        if is_training is None or is_training:
            # process the  current rendering fps
            eval_fps = f"{1 / render_time:.2f} fps at {image_height} res"
            self.vis["renderingState/eval_fps"].write(eval_fps)
            # process remaining training ETA
            self.vis["renderingState/train_eta"].write(GLOBAL_BUFFER["events"].get(EventName.ETA.value, "Starting"))
            # process ratio time spent on vis vs train
            if EventName.ITER_VIS_TIME.value in GLOBAL_BUFFER["events"]:
                vis_time = GLOBAL_BUFFER["events"][EventName.ITER_VIS_TIME.value]["avg"]
                train_time = GLOBAL_BUFFER["events"][EventName.ITER_TRAIN_TIME.value]["avg"]
                vis_train_ratio = f"{int(vis_time / train_time * 100)}% spent on viewer"
                self.vis["renderingState/vis_train_ratio"].write(vis_train_ratio)
            else:
                self.vis["renderingState/vis_train_ratio"] = "Starting"
        else:
            self.vis["renderingState/eval_fps"].write("Paused")
            self.vis["renderingState/train_eta"].write("Paused")
            self.vis["renderingState/vis_train_ratio"].write("100% spent on viewer")

    @profiler.time_function
    def _render_image_in_viewer(self, graph: Model) -> None:
        """
        Draw an image using the current camera pose from the viewer.
        The image is sent of a TCP connection and then uses WebRTC to send it to the viewer.

        Args:
            graph: current checkpoint of model
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
        output_type = OutputTypes.INIT if output_type is None else output_type
        self.output_type_changed = self.prev_output_type != output_type
        self.prev_output_type = output_type

        # check and perform colormap type updates
        colormap_type = self.vis["renderingState/colormap_choice"].read()
        colormap_type = ColormapTypes.INIT if colormap_type is None else colormap_type
        self.prev_colormap_type = colormap_type

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

        camera = Cameras(
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2],
            camera_to_worlds=camera_to_world[None, ...],
        )
        camera = camera.to(graph.device)

        camera_ray_bundle = camera.generate_rays(camera_indices=0)
        camera_ray_bundle.num_rays_per_chunk = self.config.num_rays_per_chunk

        graph.eval()

        check_thread = CheckThread(state=self)
        render_thread = RenderThread(state=self, graph=graph, camera_ray_bundle=camera_ray_bundle)

        with TimeWriter(None, None, write=False) as vis_t:
            check_thread.start()
            render_thread.start()
            try:
                render_thread.join()
                check_thread.join()
            except Exception:  # pylint: disable=broad-except
                pass
        graph.train()
        outputs = render_thread.vis_outputs
        if outputs is not None:
            stuff_colors = graph.stuff_colors if hasattr(graph, "stuff_colors") else None
            self._send_output_to_viewer(outputs, stuff_colors=stuff_colors)
            self._update_viewer_stats(vis_t.duration, num_rays=len(camera_ray_bundle), image_height=image_height)
