from __future__ import annotations

import threading
import time
from asyncio import Future
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Literal, get_args

import nerfstudio.viewer.server.viewer_utils as viewer_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.utils import colormaps
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfstudio.viewer.server.viewer_utils import SetTrace
from nerfstudio.viewer.viser._messages import CameraMessage

RenderStates = Literal["low_move", "low_static", "high"]
RenderActions = Literal["rerender", "move", "static", "step"]


@dataclass
class RenderAction:
    action: RenderActions
    cam_msg: CameraMessage


class RenderStateMachine(threading.Thread):
    def __init__(self, viewer):
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
        self.state = "low_static"
        self.render_trigger = threading.Event()
        self.target_fps = 24
        self.viewer = viewer
        self.interrupt_render_flag = False
        self.daemon = True

    def action(self, action: RenderAction):
        # TODO we need to implement framerate logic for the step action
        if action.action == "step" and self.state != "high":
            # ignore steps unless we're in high state, otherwise sometimes we get stuck inside a low state
            return
        self.next_action = action
        if self.state == "high" and action.action in ("move", "rerender"):
            self.interrupt_render_flag = True
        self.render_trigger.set()

    def _render_img(self, cam_msg: CameraMessage):
        """
        This thing actually takes the current camera, generates rays, and renders the colors
        """

        # initialize the camera ray bundle
        try:
            viewer_utils.update_render_aabb(
                crop_viewport=self.viewer.control_panel.crop_viewport,
                crop_min=self.viewer.control_panel.crop_min,
                crop_max=self.viewer.control_panel.crop_max,
                model=self.viewer.get_model(),
            )
        except RuntimeError as e:
            print(f"Error: {e}")

            time.sleep(0.5)  # sleep to allow buffer to reset

        image_height, image_width = self._calculate_image_res(cam_msg.aspect)

        intrinsics_matrix, camera_to_world_h = get_intrinsics_matrix_and_camera_to_world_h(
            cam_msg, image_height=image_height, image_width=image_width
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

        camera_type_msg = cam_msg.camera_type
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
        camera = camera.to(self.viewer.get_model().device)
        camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=self.viewer.get_model().render_aabb)
        #  TODO this rendering isn't threadsafe with training (setting .eval() can break training), we might want
        #  a mutex around training/rendering in the future.
        with TimeWriter(None, None, write=False) as vis_t:
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
        self.viewer._update_viewer_stats(
            vis_t.duration, num_rays=len(camera_ray_bundle), image_height=image_height, image_width=image_width
        )
        return outputs

    def run(self):
        while True:
            self.render_trigger.wait()
            self.render_trigger.clear()
            action = self.next_action
            if self.state == "high" and action.action == "static":
                # if we are in high res and we get a static action, we don't need to do anything
                continue
            print("Trying to render from state", self.state, "with action", action.action)
            self.state = self.transitions[self.state][action.action]
            try:
                with SetTrace(self.check_interrupt):
                    outputs = self._render_img(action.cam_msg)
            except viewer_utils.IOChangeException:
                # if we got interrupted, don't send the output to the viewer
                continue
            # TODO below line seems messy, what is it? (why is colors sometimes defined and sometimes not??)
            colors = getattr(self.viewer.get_model(), "colors", None)
            self._send_output_to_viewer(outputs, colors)  # TODO what is colors?
            # if we rendered a static low res, we need to self-trigger a static high-res
            if self.state == "low_static":
                self.action(RenderAction("static", action.cam_msg))

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.interrupt_render_flag:
                self.interrupt_render_flag = False
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(self, outputs: Dict[str, Any], colors: torch.Tensor = None):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
            colors: is only set if colormap is for semantics. Defaults to None.
        """
        if self.viewer.output_list is None:
            self.viewer.output_list = list(outputs.keys())
            viewer_output_list = list(np.copy(self.viewer.output_list))
            # remove semantics, which crashes viewer; semantics_colormap is OK
            if "semantics" in self.viewer.output_list:
                viewer_output_list.remove("semantics")
            self.viewer.control_panel.update_output_options(viewer_output_list)

        output_render = self.viewer.control_panel.output_render
        # re-register colormaps and send to viewer
        if self.viewer.output_type_changed:
            colormap_options = []
            if outputs[output_render].shape[-1] == 3:
                colormap_options = [viewer_utils.ColormapTypes.DEFAULT.value]
            if outputs[output_render].shape[-1] == 1 and outputs[output_render].dtype == torch.float:
                colormap_options = [c.value for c in list(viewer_utils.ColormapTypes)[1:]]
            self.viewer.output_type_changed = False
            self.viewer.control_panel.update_colormap_options(colormap_options)
        selected_output = (viewer_utils.apply_colormap(self.viewer.control_panel, outputs, colors) * 255).type(
            torch.uint8
        )

        self.viewer.viser_server.set_background_image(
            selected_output.cpu().numpy(),
            file_format=self.viewer.config.image_format,
            quality=self.viewer.config.jpeg_quality,
        )

    def _calculate_image_res(self, aspect_ratio: float) -> Optional[Tuple[int, int]]:
        """Calculate the maximum image height that can be rendered in the time budget

        Args:
            apect_ratio: the aspect ratio of the current view
            is_training: whether or not we are training
        Returns:
            image_height: the maximum image height that can be rendered in the time budget
            image_width: the maximum image width that can be rendered in the time budget
        """
        max_res = self.viewer.control_panel.max_res
        print("State in image res", self.state)
        if self.state == "high":
            # high res is always static
            image_height = max_res
            image_width = int(image_height * aspect_ratio)
            if image_width > max_res:
                image_width = max_res
                image_height = int(image_width / aspect_ratio)
        elif self.state in ("low_move", "low_static"):
            target_train_util = 0
            if EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                train_rays_per_sec = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
            else:
                train_rays_per_sec = 80000
            if EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]:
                vis_rays_per_sec = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
            else:
                vis_rays_per_sec = train_rays_per_sec
            target_fps = self.target_fps
            num_vis_rays = vis_rays_per_sec / target_fps * (1 - target_train_util)
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
