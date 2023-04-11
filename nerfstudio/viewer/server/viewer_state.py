""" Manage the state of the viewer """
from __future__ import annotations

import enum
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from rich.console import Console

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, profiler, writer
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.control_panel import ControlPanel
from nerfstudio.viewer.server.gui_utils import get_viewer_elements
from nerfstudio.viewer.server.subprocess import get_free_port
from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfstudio.viewer.server.viewer_param import ViewerElement
from nerfstudio.viewer.viser import ViserServer
from nerfstudio.viewer.viser._messages import (
    CameraMessage,
    CameraPathOptionsRequest,
    CameraPathPayloadMessage,
    IsTrainingMessage,
    Message,
    SaveCheckpointMessage,
)

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer

CONSOLE = Console(width=120)


@decorate_all([check_main_thread])
class ViewerState:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use

    Attributes:
        viewer_url: url to open viewer
    """

    viewer_url: str

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer: Optional[Trainer] = None,
    ):
        self.config = config
        self.trainer = trainer
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath

        if self.config.websocket_port is None:
            websocket_port = get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        CONSOLE.line()
        self.viewer_url = viewer_utils.get_viewer_url(websocket_port)
        CONSOLE.rule(characters="=")
        CONSOLE.print(f"[Public] Open the viewer at {self.viewer_url}")
        CONSOLE.rule(characters="=")
        CONSOLE.line()

        # viewer specific variables
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

        self.viser_server = ViserServer(host="localhost", port=websocket_port)

        self.viser_server.register_handler(IsTrainingMessage, self._handle_is_training)
        self.viser_server.register_handler(SaveCheckpointMessage, self._handle_save_checkpoint)
        self.viser_server.register_handler(CameraMessage, self._handle_camera_update)
        self.viser_server.register_handler(CameraPathOptionsRequest, self._handle_camera_path_option_request)
        self.viser_server.register_handler(CameraPathPayloadMessage, self._handle_camera_path_payload)

        self.control_panel = ControlPanel(self._interrupt_render, self._crop_params_update, self._output_type_change)
        self.control_panel.install(self.viser_server)

        def nested_folder_install(folder_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
            else:
                with self.viser_server.gui_folder(folder_labels[0]):
                    nested_folder_install(folder_labels[1:], element)

        self.viewer_elements = get_viewer_elements(self.pipeline)
        for param_path, element in self.viewer_elements:
            folder_labels = param_path.split("/")[:-1]
            nested_folder_install(folder_labels, element)

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _interrupt_render(self, _) -> None:
        """Interrupt current render."""
        self.check_interrupt_vis = True
        self._render_image_in_viewer(self.pipeline.model, self.is_training)

    def _crop_params_update(self, _) -> None:
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

    def _handle_save_checkpoint(self, message: Message) -> None:
        """Handle is_training message from viewer."""
        assert isinstance(message, SaveCheckpointMessage)
        if self.trainer is not None:
            self.trainer.save_checkpoint(self.step)

    def _handle_camera_update(self, message: Message) -> None:
        """Handle camera update message from viewer."""
        assert isinstance(message, CameraMessage)
        self.camera_message = message
        self.camera_moving = message.is_moving
        if self.camera_moving:
            if self.prev_moving:
                self.check_interrupt_vis = False
                self.prev_moving = True
            else:
                self.check_interrupt_vis = True
                self.prev_moving = True
        else:
            self.prev_moving = False

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

    def update_scene(self, step: int, num_rays_per_batch: int) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            pipeline: the method pipeline
            num_rays_per_batch: number of rays per batch
        """
        model = self.pipeline.model

        is_training = self.is_training
        self.step = step

        if self.camera_message is None:
            return

        if is_training is None or is_training:
            # in training mode

            if self.camera_moving:
                # if the camera is moving, then we pause training and update camera continuously

                while self.camera_moving:
                    self._render_image_in_viewer(model, is_training)
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
            while not self.is_training:
                # if self._is_render_step(local_step) and step > 0:
                if step > 0:
                    self._render_image_in_viewer(model, self.is_training)
                local_step += 1

    def check_interrupt(self, frame, event, arg):  # pylint: disable=unused-argument
        """Raises interrupt when flag has been set and not already on lowest resolution.
        Used in conjunction with SetTrace.
        """
        if event == "line":
            if self.check_interrupt_vis:
                raise viewer_utils.IOChangeException
        return self.check_interrupt

    def _send_output_to_viewer(self, outputs: Dict[str, Any], colors: torch.Tensor = None):
        """Chooses the correct output and sends it to the viewer

        Args:
            outputs: the dictionary of outputs to choose from, from the model
            colors: is only set if colormap is for semantics. Defaults to None.
        """
        if self.output_list is None:
            self.output_list = list(outputs.keys())
            viewer_output_list = list(np.copy(self.output_list))
            # remove semantics, which crashes viewer; semantics_colormap is OK
            if "semantics" in self.output_list:
                viewer_output_list.remove("semantics")
            self.control_panel.update_output_options(viewer_output_list)

        output_render = self.control_panel.output_render
        # re-register colormaps and send to viewer
        if self.output_type_changed:
            colormap_options = []
            if outputs[output_render].shape[-1] == 3:
                colormap_options = [viewer_utils.ColormapTypes.DEFAULT.value]
            if outputs[output_render].shape[-1] == 1 and outputs[output_render].dtype == torch.float:
                colormap_options = [c.value for c in list(viewer_utils.ColormapTypes)[1:]]
            self.output_type_changed = False
            self.control_panel.update_colormap_options(colormap_options)
        selected_output = (viewer_utils.apply_colormap(self.control_panel, outputs, colors) * 255).type(torch.uint8)

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
        vis_train_ratio = "Starting"
        if self.is_training is None or self.is_training:
            # process ratio time spent on vis vs train
            if (
                EventName.ITER_VIS_TIME.value in GLOBAL_BUFFER["events"]
                and EventName.ITER_TRAIN_TIME.value in GLOBAL_BUFFER["events"]
            ):
                vis_time = GLOBAL_BUFFER["events"][EventName.ITER_VIS_TIME.value]["avg"]
                train_time = GLOBAL_BUFFER["events"][EventName.ITER_TRAIN_TIME.value]["avg"]
                vis_train_ratio = f"{int(vis_time / train_time * 100)}% spent on viewer"
        else:
            vis_train_ratio = "100% spent on viewer"
        self.viser_server.send_status_message(
            eval_res=f"{image_height}x{image_width}px", vis_train_ratio=vis_train_ratio
        )

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

        # update render aabb
        try:
            viewer_utils.update_render_aabb(
                crop_viewport=self.control_panel.crop_viewport,
                crop_min=self.control_panel.crop_min,
                crop_max=self.control_panel.crop_max,
                model=model,
            )
        except RuntimeError as e:
            print(f"Error: {e}")

            time.sleep(0.5)  # sleep to allow buffer to reset

        # Calculate camera pose and intrinsics
        try:
            image_height, image_width = self._calculate_image_res(camera_message.aspect, is_training)
        except ZeroDivisionError as e:
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

        render_thread = viewer_utils.RenderThread(state=self, model=model, camera_ray_bundle=camera_ray_bundle)
        render_thread.daemon = True

        with TimeWriter(None, None, write=False) as vis_t:
            render_thread.start()
            try:
                render_thread.join()
            except viewer_utils.IOChangeException:
                del camera_ray_bundle
                torch.cuda.empty_cache()
            except RuntimeError as e:
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
