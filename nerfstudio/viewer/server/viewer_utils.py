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

# pylint: disable=too-many-lines

"""Code to interface with the `vis/` (the JS viewer)."""
from __future__ import annotations

import enum
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, profiler, writer
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.server.control_panel import ControlPanel
from nerfstudio.viewer.server.gui_utils import get_viewer_elements
from nerfstudio.viewer.server.subprocess import (
    get_free_port,
    run_viewer_bridge_server_as_subprocess,
)
from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfstudio.viewer.server.viewer_param import ViewerElement
from nerfstudio.viewer.server.visualizer import Viewer
from nerfstudio.viewer.viser import ViserServer
from nerfstudio.viewer.viser._messages import (
    CameraMessage,
    CameraPathOptionsRequest,
    CameraPathPayloadMessage,
    IsTrainingMessage,
    Message,
)

CONSOLE = Console(width=120)


def get_viewer_version() -> str:
    """Get the version of the viewer."""
    json_filename = os.path.join(os.path.dirname(__file__), "../app/package.json")
    version = load_from_json(Path(json_filename))["version"]
    return version


@check_main_thread
def setup_viewer(config: cfg.ViewerConfig, log_filename: Path, datapath: Path):
    """Sets up the viewer if enabled

    Args:
        config: the configuration to instantiate viewer
        log_filename: the log filename to write to
        datapath: the path to the dataset
    """
    viewer_state = ViewerState(config, log_filename=log_filename, datapath=datapath)
    banner_messages = [f"Viewer at: {viewer_state.viewer_url}"]
    return viewer_state, banner_messages


class OutputTypes(str, enum.Enum):
    """Noncomprehensive list of output render types"""

    INIT = "init"
    RGB = "rgb"
    RGB_FINE = "rgb_fine"
    ACCUMULATION = "accumulation"
    ACCUMULATION_FINE = "accumulation_fine"


class ColormapTypes(str, enum.Enum):
    """List of colormap render types"""

    DEFAULT = "default"
    TURBO = "turbo"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    INFERNO = "inferno"
    CIVIDIS = "cividis"


class IOChangeException(Exception):
    """Basic camera exception to interrupt viewer"""


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
        state: current viewer state object
        model: current checkpoint of model
        camera_ray_bundle: input rays to pass through the model to render out
    """

    def __init__(self, state: "ViewerState", model: Model, camera_ray_bundle: RayBundle):
        threading.Thread.__init__(self)
        self.state = state
        self.model = model
        self.camera_ray_bundle = camera_ray_bundle
        self.exc = None
        self.vis_outputs = None

    def run(self):
        """run function that renders out images given the current model and ray bundles.
        Interlaced with a trace function that checks to see if any I/O changes were registered.
        Exits and continues program if IOChangeException thrown.
        """
        outputs = None
        try:
            with SetTrace(self.state.check_interrupt):
                if self.state.control_panel.crop_viewport:
                    color = self.state.control_panel.background_color
                    if color is None:
                        background_color = torch.tensor([0.0, 0.0, 0.0], device=self.model.device)
                    else:
                        background_color = torch.tensor(
                            [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0], device=self.model.device
                        )
                    with renderers.background_color_override_context(background_color), torch.no_grad():
                        outputs = self.model.get_outputs_for_camera_ray_bundle(self.camera_ray_bundle)
                else:
                    with torch.no_grad():
                        outputs = self.model.get_outputs_for_camera_ray_bundle(self.camera_ray_bundle)
        except Exception as e:  # pylint: disable=broad-except
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
        state: current viewer state object
    """

    def __init__(self, state):
        threading.Thread.__init__(self)
        self.state = state

    def run(self):
        """Run function that checks to see if any of the existing state has changed
        (e.g. camera pose/output type/resolutions).
        Sets the viewer state flag to true to signal
        to render thread that an interrupt was registered.
        """
        self.state.check_done_render = False
        while not self.state.check_done_render:
            # Interrupt on the start of a camera move, but not during a camera move
            if self.state.camera_message is not None:
                if self.state.camera_moving:
                    if self.state.prev_moving:
                        self.state.check_interrupt_vis = False
                        self.state.prev_moving = True
                    else:
                        self.state.check_interrupt_vis = True
                        self.state.prev_moving = True
                        return
                else:
                    self.state.prev_moving = False

            # check output type
            output_type = self.state.vis["renderingState/output_choice"].read()
            if output_type is None:
                output_type = OutputTypes.INIT
            if self.state.prev_output_type != output_type:
                self.state.check_interrupt_vis = True
                return

            # check colormap type
            colormap_type = self.state.vis["renderingState/colormap_choice"].read()
            if self.state.prev_colormap_type != colormap_type:
                self.state.check_interrupt_vis = True
                return

            colormap_range = self.state.vis["renderingState/colormap_range"].read()
            if self.state.prev_colormap_range != colormap_range:
                self.state.check_interrupt_vis = True
                return


@decorate_all([check_main_thread])
class ViewerState:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
    """

    vis: Any

    def __init__(self, config: cfg.ViewerConfig, log_filename: Path, datapath: Path):
        self.config = config
        self.vis = None
        self.viewer_url = None
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        if self.config.launch_bridge_server:
            # start the viewer bridge server
            if self.config.websocket_port is None:
                websocket_port = get_free_port(default_port=self.config.websocket_port_default)
            else:
                websocket_port = self.config.websocket_port
            self.log_filename.parent.mkdir(exist_ok=True)
            zmq_port = run_viewer_bridge_server_as_subprocess(
                websocket_port,
                zmq_port=self.config.zmq_port,
                ip_address=self.config.ip_address,
                log_filename=str(self.log_filename),
            )
            # TODO(ethan): log the output of the viewer bridge server in a file where the training logs go
            CONSOLE.line()
            version = get_viewer_version()
            websocket_url = f"ws://localhost:{websocket_port}"
            self.viewer_url = f"https://viewer.nerf.studio/versions/{version}/?websocket_url={websocket_url}"
            CONSOLE.rule(characters="=")
            CONSOLE.print(f"[Public] Open the viewer at {self.viewer_url}")
            CONSOLE.rule(characters="=")
            CONSOLE.line()
            self.vis = Viewer(zmq_port=zmq_port, ip_address=self.config.ip_address)
        else:
            assert self.config.zmq_port is not None
            self.vis = Viewer(zmq_port=self.config.zmq_port, ip_address=self.config.ip_address)

        # viewer specific variables
        self.prev_output_type = OutputTypes.INIT
        self.prev_colormap_type = None
        self.prev_colormap_invert = False
        self.prev_colormap_normalize = False
        self.prev_colormap_range = [0, 1]
        self.prev_moving = False
        self.output_type_changed = True
        self.check_interrupt_vis = False
        self.check_done_render = True
        self.step = 0
        self.static_fps = 1
        self.moving_fps = 24
        self.camera_moving = False
        self.prev_camera_timestamp = 0

        self.output_list = None
        self.is_training = self.config.start_train
        self.camera_message = None

        # TODO: host and port should not be hardcoded. This should eventually replace
        # the ZMQ + websocket logic above.
        self.viser_server = ViserServer(host="localhost", port=8080)

        self.viser_server.register_handler(IsTrainingMessage, self._handle_is_training)
        self.viser_server.register_handler(CameraMessage, self._handle_camera_update)
        self.viser_server.register_handler(CameraPathOptionsRequest, self._handle_camera_path_option_request)
        self.viser_server.register_handler(CameraPathPayloadMessage, self._handle_camera_path_payload)

        self.control_panel = ControlPanel(self._interrupt_render, self._crop_params_update)
        self.control_panel.install(self.viser_server)

    def _interrupt_render(self) -> None:
        """Interrupt current render."""
        self.check_interrupt_vis = True

    def _crop_params_update(self) -> None:
        """Update crop parameters"""
        crop_min = torch.tensor(self.control_panel.crop_min, dtype=torch.float32)
        crop_max = torch.tensor(self.control_panel.crop_max, dtype=torch.float32)
        scene_box = SceneBox(aabb=torch.stack([crop_min, crop_max], dim=0))
        self.viser_server.update_scene_box(scene_box)
        crop_scale = (crop_max - crop_min) / 2.0
        crop_center = (crop_max + crop_min) / 2.0
        self.viser_server.send_crop_params(
            crop_enabled=self.control_panel.crop_viewport,
            crop_bg_color=self.control_panel.background_color,
            crop_scale=tuple(crop_scale.tolist()),
            crop_center=tuple(crop_center.tolist()),
        )

    def _handle_is_training(self, message: Message) -> None:
        """Handle is_training message from viewer."""
        assert isinstance(message, IsTrainingMessage)
        self.is_training = message.is_training

    def _handle_camera_update(self, message: Message) -> None:
        """Handle camera update message from viewer."""
        assert isinstance(message, CameraMessage)
        self.camera_message = message
        self.camera_moving = message.is_moving

    def _handle_camera_path_option_request(self, message: Message) -> None:
        """Handle camera path option request message from viewer."""
        assert isinstance(message, CameraPathOptionsRequest)
        camera_path_dir = self.datapath / "camera_paths"
        if camera_path_dir.exists():
            all_path_dict = {}
            for path in camera_path_dir.iterdir():
                if path.suffix == ".json":
                    all_path_dict[path.stem] = load_from_json(path)
            self.viser_server.send_camera_paths(all_path_dict)

    def _handle_camera_path_payload(self, message: Message) -> None:
        """Handle camera path payload message from viewer."""
        assert isinstance(message, CameraPathPayloadMessage)
        camera_path_filename = message.camera_path_filename + ".json"
        camera_path = message.camera_path
        camera_paths_directory = self.datapath / "camera_paths"
        camera_paths_directory.mkdir(parents=True, exist_ok=True)
        write_to_json(camera_paths_directory / camera_path_filename, camera_path)

    def _pick_drawn_image_idxs(self, total_num: int) -> list[int]:
        """Determine indicies of images to display in viewer.

        Args:
            total_num: total number of training images.

        Returns:
            List of indices from [0, total_num-1].
        """
        if self.config.max_num_display_images < 0:
            num_display_images = total_num
        else:
            num_display_images = min(self.config.max_num_display_images, total_num)
        # draw indices, roughly evenly spaced
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32).tolist()

    def init_scene(self, dataset: InputDataset, start_train=True) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            start_train: whether to start train when viewer init;
                if False, only displays dataset until resume train is toggled
        """
        self.viser_server.send_file_path_info(
            config_base_dir=self.log_filename.parents[0],
            data_base_dir=self.datapath,
            export_path_name=self.log_filename.parent.stem,
        )

        # draw the training cameras and images
        image_indices = self._pick_drawn_image_idxs(len(dataset))
        for idx in image_indices:
            image = dataset[idx]["image"]
            bgr = image[..., [2, 1, 0]]
            camera_json = dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            self.viser_server.add_dataset_image(idx=f"{idx:06d}", json=camera_json)

        # draw the scene box (i.e., the bounding box)
        self.viser_server.update_scene_box(dataset.scene_box)

        # set the initial state whether to train or not
        self.viser_server.set_is_training(start_train)

    def _update_render_aabb(self, model):
        """
        update the render aabb box for the viewer:

        Args:
            model: the model to render
        """

        if self.control_panel.crop_viewport:
            crop_min = torch.tensor(self.control_panel.crop_min, dtype=torch.float32)
            crop_max = torch.tensor(self.control_panel.crop_max, dtype=torch.float32)

            if isinstance(model.render_aabb, SceneBox):
                model.render_aabb.aabb[0] = crop_min
                model.render_aabb.aabb[1] = crop_max
            else:
                model.render_aabb = SceneBox(aabb=torch.stack([crop_min, crop_max], dim=0))
        else:
            model.render_aabb = None

    def update_scene(self, step: int, pipeline: Pipeline, num_rays_per_batch: int) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            pipeline: the method pipeline
            num_rays_per_batch: number of rays per batch
        """
        if not hasattr(self, "viewer_elements"):

            def nested_folder_install(folder_labels: List[str], element: ViewerElement):
                if len(folder_labels) == 0:
                    element.install(self.viser_server)
                else:
                    with self.viser_server.gui_folder(folder_labels[0]):
                        nested_folder_install(folder_labels[1:], element)

            self.viewer_elements = get_viewer_elements(pipeline)
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, element)
        model = pipeline.model

        is_training = self.is_training
        self.step = step

        self._legacy_messages()
        if self.camera_message is None:
            return

        if is_training is None or is_training:
            # in training mode

            if self.camera_moving:
                # if the camera is moving, then we pause training and update camera continuously

                while self.camera_moving:
                    self._render_image_in_viewer(model, is_training)
                    self._legacy_messages()
            else:
                # if the camera is not moving, then we approximate how many training steps need to be taken
                # to render at a FPS defined by self.static_fps.

                if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                    train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                    target_train_util = self.control_panel.train_util
                    if target_train_util is None:
                        target_train_util = 0.9

                    batches_per_sec = train_rays_per_sec / num_rays_per_batch

                    num_steps = max(int(1 / self.static_fps * batches_per_sec), 1)
                else:
                    num_steps = 1

                if step % num_steps == 0:
                    self._render_image_in_viewer(model, is_training)

        else:
            # in pause training mode, enter render loop with set model
            local_step = step
            run_loop = not self.is_training
            while run_loop:
                # if self._is_render_step(local_step) and step > 0:
                if step > 0:
                    self._legacy_messages()
                    self._render_image_in_viewer(model, self.is_training)
                run_loop = not self.is_training
                local_step += 1

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.check_interrupt_vis:
                raise IOChangeException
        return self.check_interrupt

    def _legacy_messages(self):
        """Gets the camera object from the viewer and updates the movement state if it has changed."""

        output_type = self.vis["renderingState/output_choice"].read()
        if output_type is None:
            output_type = OutputTypes.INIT
        if self.prev_output_type != output_type:
            self.camera_moving = True

        colormap_type = self.vis["renderingState/colormap_choice"].read()
        if self.prev_colormap_type != colormap_type:
            self.camera_moving = True

        colormap_range = self.vis["renderingState/colormap_range"].read()
        if self.prev_colormap_range != colormap_range:
            self.camera_moving = True

        colormap_invert = self.vis["renderingState/colormap_invert"].read()
        if self.prev_colormap_invert != colormap_invert:
            self.camera_moving = True

        colormap_normalize = self.vis["renderingState/colormap_normalize"].read()
        if self.prev_colormap_normalize != colormap_normalize:
            self.camera_moving = True

    def _apply_colormap(self, outputs: Dict[str, Any], colors: torch.Tensor = None, eps=1e-6):
        """Determines which colormap to use based on set colormap type

        Args:
            outputs: the output tensors for which to apply colormaps on
            colors: is only set if colormap is for semantics. Defaults to None.
            eps: epsilon to handle floating point comparisons
        """
        if self.output_list:
            reformatted_output = self._process_invalid_output(self.prev_output_type)

        # default for rgb images
        if self.prev_colormap_type == ColormapTypes.DEFAULT and outputs[reformatted_output].shape[-1] == 3:
            return outputs[reformatted_output]

        # rendering depth outputs
        if outputs[reformatted_output].shape[-1] == 1 and outputs[reformatted_output].dtype == torch.float:
            output = outputs[reformatted_output]
            if self.prev_colormap_normalize:
                output = output - torch.min(output)
                output = output / (torch.max(output) + eps)
            output = output * (self.prev_colormap_range[1] - self.prev_colormap_range[0]) + self.prev_colormap_range[0]
            output = torch.clip(output, 0, 1)
            if self.prev_colormap_invert:
                output = 1 - output
            if self.prev_colormap_type == ColormapTypes.DEFAULT:
                return colormaps.apply_colormap(output, cmap=ColormapTypes.TURBO.value)
            return colormaps.apply_colormap(output, cmap=self.prev_colormap_type)

        # rendering semantic outputs
        if outputs[reformatted_output].dtype == torch.int:
            logits = outputs[reformatted_output]
            labels = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)  # type: ignore
            assert colors is not None
            return colors[labels]

        # rendering boolean outputs
        if outputs[reformatted_output].dtype == torch.bool:
            return colormaps.apply_boolean_colormap(outputs[reformatted_output])

        raise NotImplementedError

    def _send_output_to_viewer(self, outputs: Dict[str, Any], colors: torch.Tensor = None):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
            colors: is only set if colormap is for semantics. Defaults to None.
        """
        if self.output_list is None:
            self.output_list = list(outputs.keys())
            viewer_output_list = list(np.copy(self.output_list))
            # remapping rgb_fine -> rgb for all cases just so that we dont have 2 of them in the options
            if OutputTypes.RGB_FINE in self.output_list:
                viewer_output_list.remove(OutputTypes.RGB_FINE)
            viewer_output_list.insert(0, OutputTypes.RGB)
            # remove semantics, which crashes viewer; semantics_colormap is OK
            if "semantics" in self.output_list:
                viewer_output_list.remove("semantics")
            self.vis["renderingState/output_options"].write(viewer_output_list)

        reformatted_output = self._process_invalid_output(self.prev_output_type)
        # re-register colormaps and send to viewer
        if self.output_type_changed or self.prev_colormap_type is None:
            self.prev_colormap_type = ColormapTypes.DEFAULT
            colormap_options = []
            self.vis["renderingState/colormap_options"].write(list(ColormapTypes))
            if outputs[reformatted_output].shape[-1] == 3:
                colormap_options = [ColormapTypes.DEFAULT]
            if outputs[reformatted_output].shape[-1] == 1 and outputs[reformatted_output].dtype == torch.float:
                self.prev_colormap_type = ColormapTypes.TURBO
                colormap_options = list(ColormapTypes)[1:]
            self.output_type_changed = False
            self.vis["renderingState/colormap_choice"].write(self.prev_colormap_type)
            self.vis["renderingState/colormap_options"].write(colormap_options)
        selected_output = (self._apply_colormap(outputs, colors) * 255).type(torch.uint8)

        self.viser_server.set_background_image(
            selected_output.cpu().numpy(), file_format=self.config.image_format, quality=self.config.jpeg_quality
        )

    def _update_viewer_stats(self, render_time: float, num_rays: int, image_height: int, image_width: int) -> None:
        """Function that calculates and populates all the rendering statistics accordingly

        Args:
            render_time: total time spent rendering current view
            num_rays: number of rays rendered
            image_height: resolution of the current view
            image_width: resolution of the current view
        """
        writer.put_time(
            name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=self.step, avg_over_steps=True
        )
        self.vis["renderingState/eval_res"].write(f"{image_height}x{image_width}px")
        if self.is_training is None or self.is_training:
            # process remaining training ETA
            self.vis["renderingState/train_eta"].write(GLOBAL_BUFFER["events"].get(EventName.ETA.value, "Starting"))
            # process ratio time spent on vis vs train
            if (
                EventName.ITER_VIS_TIME.value in GLOBAL_BUFFER["events"]
                and EventName.ITER_TRAIN_TIME.value in GLOBAL_BUFFER["events"]
            ):
                vis_time = GLOBAL_BUFFER["events"][EventName.ITER_VIS_TIME.value]["avg"]
                train_time = GLOBAL_BUFFER["events"][EventName.ITER_TRAIN_TIME.value]["avg"]
                vis_train_ratio = f"{int(vis_time / train_time * 100)}% spent on viewer"
                self.vis["renderingState/vis_train_ratio"].write(vis_train_ratio)
            else:
                self.vis["renderingState/vis_train_ratio"].write("Starting")
        else:
            self.vis["renderingState/train_eta"].write("Paused")
            self.vis["renderingState/vis_train_ratio"].write("100% spent on viewer")

    def _calculate_image_res(self, aspect_ratio: float, is_training: bool) -> Optional[Tuple[int, int]]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            apect_ratio: the aspect ratio of the current view
            is_training: whether or not we are training
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        if self.camera_moving or not is_training:
            target_train_util = 0
        else:
            target_train_util = self.control_panel.train_util
            if target_train_util is None:
                target_train_util = 0.9

        if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
            train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
        elif not is_training:
            train_rays_per_sec = (
                80000  # TODO(eventually find a way to not hardcode. case where there are no prior training steps)
            )
        else:
            return None, None
        if EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
            vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
        else:
            vis_rays_per_sec = train_rays_per_sec

        current_fps = self.moving_fps if self.camera_moving else self.static_fps

        # calculate number of rays that can be rendered given the target fps
        num_vis_rays = vis_rays_per_sec / current_fps * (1 - target_train_util)

        max_res = self.control_panel.max_res
        if not self.camera_moving and not is_training:
            image_height = max_res
        else:
            image_height = (num_vis_rays / aspect_ratio) ** 0.5
            image_height = int(round(image_height, -1))
            image_height = max(min(max_res, image_height), 30)
        image_width = int(image_height * aspect_ratio)
        if image_width > max_res:
            image_width = max_res
            image_height = int(image_width / aspect_ratio)
        return image_height, image_width

    def _process_invalid_output(self, output_type: str) -> str:
        """Check to see whether we are in the corner case of RGB; if still invalid, throw error
        Returns correct string mapping given improperly formatted output_type.

        Args:
            output_type: reformatted output type
        """
        if output_type == OutputTypes.INIT:
            output_type = OutputTypes.RGB

        # check if rgb or rgb_fine should be the case TODO: add other checks here
        attempted_output_type = output_type
        if output_type not in self.output_list and output_type == OutputTypes.RGB:
            output_type = OutputTypes.RGB_FINE

        # check if output_type is not in list
        if output_type not in self.output_list:
            assert (
                NotImplementedError
            ), f"Output {attempted_output_type} not in list. Tried to reformat as {output_type} but still not found."
        return output_type

    @profiler.time_function
    def _render_image_in_viewer(self, model: Model, is_training: bool) -> None:
        # pylint: disable=too-many-statements
        """
        Draw an image using the current camera pose from the viewer.
        The image is sent over a TCP connection.

        Args:
            model: current checkpoint of model
            is_training: whether or not we are training
        """
        # Check that timestamp is newer than the last one
        camera_message = self.camera_message
        if int(camera_message.timestamp) < self.prev_camera_timestamp:
            return

        self.prev_camera_timestamp = camera_message.timestamp

        # check and perform output type updates
        output_type = self.vis["renderingState/output_choice"].read()
        output_type = OutputTypes.INIT if output_type is None else output_type
        self.output_type_changed = self.prev_output_type != output_type
        self.prev_output_type = output_type

        # check and perform colormap type updates
        colormap_type = self.vis["renderingState/colormap_choice"].read()
        self.prev_colormap_type = colormap_type

        colormap_invert = self.vis["renderingState/colormap_invert"].read()
        self.prev_colormap_invert = colormap_invert

        colormap_normalize = self.vis["renderingState/colormap_normalize"].read()
        self.prev_colormap_normalize = colormap_normalize

        colormap_range = self.vis["renderingState/colormap_range"].read()
        self.prev_colormap_range = colormap_range

        # update render aabb
        try:
            self._update_render_aabb(model)
        except RuntimeError as e:
            self.vis["renderingState/log_errors"].write("Got an Error while trying to update aabb crop")
            print(f"Error: {e}")

            time.sleep(0.5)  # sleep to allow buffer to reset

        # Calculate camera pose and intrinsics
        try:
            image_height, image_width = self._calculate_image_res(camera_message.aspect, is_training)
        except ZeroDivisionError as e:
            self.vis["renderingState/log_errors"].write("Error: Screen too small; no rays intersecting scene.")
            time.sleep(0.03)  # sleep to allow buffer to reset
            print(f"Error: {e}")
            return

        if image_height is None:
            return

        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            camera_message, image_height=image_height, image_width=image_width
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

        camera_type_msg = camera_message.camera_type
        if camera_type_msg == "perspective":
            camera_type = CameraType.PERSPECTIVE
        elif camera_type_msg == "fisheye":
            camera_type = CameraType.FISHEYE
        elif camera_type_msg == "equirectangular":
            camera_type = CameraType.EQUIRECTANGULAR
        else:
            camera_type = CameraType.PERSPECTIVE

        camera = Cameras(
            fx=intrinsics_matrix[0, 0],
            fy=intrinsics_matrix[1, 1],
            cx=intrinsics_matrix[0, 2],
            cy=intrinsics_matrix[1, 2],
            camera_type=camera_type,
            camera_to_worlds=camera_to_world[None, ...],
            times=torch.tensor([0.0]),
        )
        camera = camera.to(model.device)

        camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=model.render_aabb)

        model.eval()

        check_thread = CheckThread(state=self)
        render_thread = RenderThread(state=self, model=model, camera_ray_bundle=camera_ray_bundle)

        check_thread.daemon = True
        render_thread.daemon = True

        with TimeWriter(None, None, write=False) as vis_t:
            check_thread.start()
            render_thread.start()
            try:
                render_thread.join()
                check_thread.join()
            except IOChangeException:
                del camera_ray_bundle
                torch.cuda.empty_cache()
            except RuntimeError as e:
                self.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )
                print(f"Error: {e}")
                del camera_ray_bundle
                torch.cuda.empty_cache()
                time.sleep(0.5)  # sleep to allow buffer to reset

        model.train()
        outputs = render_thread.vis_outputs
        if outputs is not None:
            colors = model.colors if hasattr(model, "colors") else None
            self._send_output_to_viewer(outputs, colors=colors)
            self._update_viewer_stats(
                vis_t.duration, num_rays=len(camera_ray_bundle), image_height=image_height, image_width=image_width
            )
