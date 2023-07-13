# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

""" This file contains the render state machine, which is responsible for deciding when to render the image """
from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, get_args

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.utils import colormaps, writer
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.viser.messages import CameraMessage

if TYPE_CHECKING:
    from nerfstudio.viewer.server.viewer_state import ViewerState

RenderStates = Literal["low_move", "low_static", "high"]
RenderActions = Literal["rerender", "move", "static", "step"]


@dataclass
class RenderAction:
    """Message to the render state machine"""

    action: RenderActions
    """The action to take """
    cam_msg: CameraMessage
    """The camera message to render"""


class RenderStateMachine(threading.Thread):
    """The render state machine is responsible for deciding how to render the image.
    It decides the resolution and whether to interrupt the current render.

    Args:
        viewer: the viewer state
    """

    def __init__(self, viewer: ViewerState):
        threading.Thread.__init__(self)
        self.transitions: Dict[RenderStates, Dict[RenderActions, RenderStates]] = {
            s: {} for s in get_args(RenderStates)
        }
        # by default, everything is a self-transition
        for a in get_args(RenderActions):
            for s in get_args(RenderStates):
                self.transitions[s][a] = s
        # then define the actions between states
        self.transitions["low_move"]["static"] = "low_static"
        self.transitions["low_static"]["static"] = "high"
        self.transitions["low_static"]["step"] = "high"
        self.transitions["low_static"]["move"] = "low_move"
        self.transitions["high"]["move"] = "low_move"
        self.transitions["high"]["rerender"] = "low_static"
        self.next_action: Optional[RenderAction] = None
        self.state: RenderStates = "low_static"
        self.render_trigger = threading.Event()
        self.target_fps = 24
        self.viewer = viewer
        self.interrupt_render_flag = False
        self.daemon = True
        self.output_keys = {}

    def action(self, action: RenderAction):
        """Takes an action and updates the state machine

        Args:
            action: the action to take
        """
        if self.next_action is None:
            self.next_action = action
        elif action.action == "step" and (
            self.state == "low_move" or self.next_action.action in ("move", "static", "rerender")
        ):
            # ignore steps if:
            #  1. we are in low_moving state
            #  2. the current next_action is move, static, or rerender
            return
        elif self.next_action == "rerender":
            # never overwrite rerenders
            pass
        else:
            #  minimal use case, just set the next action
            self.next_action = action

        # handle interrupt logic
        if self.state == "high" and self.next_action.action in ("move", "rerender"):
            self.interrupt_render_flag = True
        self.render_trigger.set()

    def _render_img(self, cam_msg: CameraMessage):
        """Takes the current camera, generates rays, and renders the image

        Args:
            cam_msg: the camera message to render
        """

        # initialize the camera ray bundle
        viewer_utils.update_render_aabb(
            crop_viewport=self.viewer.control_panel.crop_viewport,
            crop_min=self.viewer.control_panel.crop_min,
            crop_max=self.viewer.control_panel.crop_max,
            model=self.viewer.get_model(),
        )

        image_height, image_width = self._calculate_image_res(cam_msg.aspect)

        camera: Optional[Cameras] = self.viewer.get_camera(image_height, image_width)
        assert camera is not None, "render called before viewer connected"

        with self.viewer.train_lock if self.viewer.train_lock is not None else contextlib.nullcontext():
            camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=self.viewer.get_model().render_aabb)

            with TimeWriter(None, None, write=False) as vis_t:
                self.viewer.get_model().eval()
                step = self.viewer.step
                if self.viewer.control_panel.crop_viewport:
                    color = self.viewer.control_panel.background_color
                    if color is None:
                        background_color = torch.tensor([0.0, 0.0, 0.0], device=self.viewer.pipeline.model.device)
                    else:
                        background_color = torch.tensor(
                            [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
                            device=self.viewer.get_model().device,
                        )
                    with background_color_override_context(background_color), torch.no_grad():
                        outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                else:
                    with torch.no_grad():
                        outputs = self.viewer.get_model().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                self.viewer.get_model().train()
        num_rays = len(camera_ray_bundle)
        render_time = vis_t.duration
        if writer.is_initialized():
            writer.put_time(
                name=EventName.VIS_RAYS_PER_SEC, duration=num_rays / render_time, step=step, avg_over_steps=True
            )
        self.viewer.viser_server.send_status_message(eval_res=f"{image_height}x{image_width}px", step=step)
        return outputs

    def run(self):
        """Main loop for the render thread"""
        while True:
            self.render_trigger.wait()
            self.render_trigger.clear()
            action = self.next_action
            assert action is not None, "Action should never be None at this point"
            self.next_action = None
            if self.state == "high" and action.action == "static":
                # if we are in high res and we get a static action, we don't need to do anything
                continue
            self.state = self.transitions[self.state][action.action]
            try:
                with viewer_utils.SetTrace(self.check_interrupt):
                    outputs = self._render_img(action.cam_msg)
            except viewer_utils.IOChangeException:
                # if we got interrupted, don't send the output to the viewer
                continue
            self._send_output_to_viewer(outputs)
            # if we rendered a static low res, we need to self-trigger a static high-res
            if self.state == "low_static":
                self.action(RenderAction("static", action.cam_msg))

    def check_interrupt(self, frame, event, arg):
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.interrupt_render_flag:
                self.interrupt_render_flag = False
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(self, outputs: Dict[str, Any]):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
        """
        output_keys = set(outputs.keys())
        if self.output_keys != output_keys:
            self.output_keys = output_keys
            self.viewer.viser_server.send_output_options_message(list(outputs.keys()))
            self.viewer.control_panel.update_output_options(list(outputs.keys()))

        output_render = self.viewer.control_panel.output_render
        self.viewer.update_colormap_options(
            dimensions=outputs[output_render].shape[-1], dtype=outputs[output_render].dtype
        )
        selected_output = colormaps.apply_colormap(
            image=outputs[self.viewer.control_panel.output_render],
            colormap_options=self.viewer.control_panel.colormap_options,
        )

        if self.viewer.control_panel.split:
            split_output_render = self.viewer.control_panel.split_output_render
            self.viewer.update_split_colormap_options(
                dimensions=outputs[split_output_render].shape[-1], dtype=outputs[split_output_render].dtype
            )
            split_output = colormaps.apply_colormap(
                image=outputs[self.viewer.control_panel.split_output_render],
                colormap_options=self.viewer.control_panel.split_colormap_options,
            )
            split_index = min(
                int(self.viewer.control_panel.split_percentage * selected_output.shape[1]),
                selected_output.shape[1] - 1,
            )
            selected_output = torch.cat([selected_output[:, :split_index], split_output[:, split_index:]], dim=1)
            selected_output[:, split_index] = torch.tensor([0.133, 0.157, 0.192], device=selected_output.device)

        selected_output = (selected_output * 255).type(torch.uint8)

        self.viewer.viser_server.set_background_image(
            selected_output.cpu().numpy(),
            file_format=self.viewer.config.image_format,
            quality=self.viewer.config.jpeg_quality,
        )

    def _calculate_image_res(self, aspect_ratio: float) -> Tuple[int, int]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            aspect_ratio: the aspect ratio of the current view
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        if aspect_ratio == 0:
            aspect_ratio = 0.001
        max_res = self.viewer.control_panel.max_res
        if self.state == "high":
            # high res is always static
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        elif self.state in ("low_move", "low_static"):
            if writer.is_initialized() and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
            else:
                vis_rays_per_sec = 100000
            target_fps = self.target_fps
            num_vis_rays = vis_rays_per_sec / target_fps
            image_height = (num_vis_rays / aspect_ratio) ** 0.5
            image_height = int(round(image_height, -1))
            image_height = max(min(max_res, image_height), 30)
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        else:
            raise ValueError(f"Invalid state: {self.state}")

        return image_height, image_width
