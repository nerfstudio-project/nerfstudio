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

""" Manage the state of the viewer """
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional

import numpy as np
import torch
from rich import box, style
from rich.panel import Panel
from rich.table import Table

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.control_panel import ControlPanel
from nerfstudio.viewer.server.gui_utils import parse_object
from nerfstudio.viewer.server.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfstudio.viewer.server.viewer_elements import ViewerControl, ViewerElement
from nerfstudio.viewer.viser import ViserServer
from nerfstudio.viewer.viser.messages import (
    CameraMessage,
    CameraPathOptionsRequest,
    CameraPathPayloadMessage,
    ClickMessage,
    CropParamsMessage,
    NerfstudioMessage,
    SaveCheckpointMessage,
    TimeConditionMessage,
    TrainingStateMessage,
)

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


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
        train_lock: Optional[threading.Lock] = None,
    ):
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        self.viewer_url = viewer_utils.get_viewer_url(websocket_port)
        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("HTTP", f"[link={self.viewer_url}][blue]{self.viewer_url}[/link]")

        CONSOLE.print(Panel(table, title="[bold][yellow]Viewer[/bold]", expand=False))

        self.include_time = self.pipeline.datamanager.includes_time

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = "training"
        self._prev_train_state: Literal["training", "paused", "completed"] = "training"

        self.camera_message = None

        self.viser_server = ViserServer(host=config.websocket_host, port=websocket_port)

        self.viser_server.register_handler(TrainingStateMessage, self._handle_training_state_message)
        self.viser_server.register_handler(SaveCheckpointMessage, self._handle_save_checkpoint)
        self.viser_server.register_handler(CameraMessage, self._handle_camera_update)
        self.viser_server.register_handler(CameraPathOptionsRequest, self._handle_camera_path_option_request)
        self.viser_server.register_handler(CameraPathPayloadMessage, self._handle_camera_path_payload)
        self.viser_server.register_handler(CropParamsMessage, self._handle_crop_params_message)
        self.viser_server.register_handler(ClickMessage, self._handle_click_message)
        if self.include_time:
            self.viser_server.use_time_conditioning()
            self.viser_server.register_handler(TimeConditionMessage, self._handle_time_condition_message)

        self.control_panel = ControlPanel(
            self.viser_server,
            self.include_time,
            self._interrupt_render,
            self._crop_params_update,
            self._output_type_change,
            self._output_split_type_change,
        )

        def nested_folder_install(folder_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb(element), self._interrupt_render(element)]
            else:
                with self.viser_server.gui_folder(folder_labels[0]):
                    nested_folder_install(folder_labels[1:], element)

        self.viewer_elements = []
        if self.trainer is not None:
            self.viewer_elements.extend(parse_object(self.trainer, ViewerElement, "Trainer"))
        else:
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Pipeline"))
        for param_path, element in self.viewer_elements:
            folder_labels = param_path.split("/")[:-1]
            nested_folder_install(folder_labels, element)

        # scrape the trainer/pipeline for any ViewerControl objects to initialize them
        if self.trainer is not None:
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(self.trainer, ViewerControl, "Trainer")
            ]
        else:
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(self.trainer, ViewerControl, "Pipeline")
            ]

        for c in self.viewer_controls:
            c._setup(self)
        self.render_statemachine = RenderStateMachine(self)
        self.render_statemachine.start()

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _output_split_type_change(self, _):
        self.output_split_type_changed = True

    def _interrupt_render(self, _) -> None:
        """Interrupt current render."""
        if self.camera_message is not None:
            self.render_statemachine.action(RenderAction("rerender", self.camera_message))

    def _crop_params_update(self, _) -> None:
        """Update crop parameters"""
        crop_min = torch.tensor(self.control_panel.crop_min, dtype=torch.float32)
        crop_max = torch.tensor(self.control_panel.crop_max, dtype=torch.float32)
        scene_box = SceneBox(aabb=torch.stack([crop_min, crop_max], dim=0))
        self.viser_server.update_scene_box(scene_box)
        crop_scale = crop_max - crop_min
        crop_center = (crop_max + crop_min) / 2.0
        self.viser_server.send_crop_params(
            crop_enabled=self.control_panel.crop_viewport,
            crop_bg_color=self.control_panel.background_color,
            crop_scale=tuple(crop_scale.tolist()),
            crop_center=tuple(crop_center.tolist()),
        )
        if self.camera_message is not None:
            self.render_statemachine.action(RenderAction("rerender", self.camera_message))

    def _handle_training_state_message(self, message: NerfstudioMessage) -> None:
        """Handle training state message from viewer."""
        assert isinstance(message, TrainingStateMessage)
        self.train_btn_state = message.training_state
        self.training_state = message.training_state
        self.viser_server.set_training_state(message.training_state)

    def _handle_save_checkpoint(self, message: NerfstudioMessage) -> None:
        """Handle save checkpoint message from viewer."""
        assert isinstance(message, SaveCheckpointMessage)
        if self.trainer is not None:
            self.trainer.save_checkpoint(self.step)

    def _handle_camera_update(self, message: NerfstudioMessage) -> None:
        """Handle camera update message from viewer."""
        assert isinstance(message, CameraMessage)
        self.camera_message = message
        if message.is_moving:
            self.render_statemachine.action(RenderAction("move", self.camera_message))
            if self.training_state == "training":
                self.training_state = "paused"
        else:
            self.render_statemachine.action(RenderAction("static", self.camera_message))
            self.training_state = self.train_btn_state

    def _handle_camera_path_option_request(self, message: NerfstudioMessage) -> None:
        """Handle camera path option request message from viewer."""
        assert isinstance(message, CameraPathOptionsRequest)
        camera_path_dir = self.datapath / "camera_paths"
        if camera_path_dir.exists():
            all_path_dict = {}
            for path in camera_path_dir.iterdir():
                if path.suffix == ".json":
                    all_path_dict[path.stem] = load_from_json(path)
            self.viser_server.send_camera_paths(all_path_dict)

    def _handle_camera_path_payload(self, message: NerfstudioMessage) -> None:
        """Handle camera path payload message from viewer."""
        assert isinstance(message, CameraPathPayloadMessage)
        camera_path_filename = message.camera_path_filename + ".json"
        camera_path = message.camera_path
        camera_paths_directory = self.datapath / "camera_paths"
        camera_paths_directory.mkdir(parents=True, exist_ok=True)
        write_to_json(camera_paths_directory / camera_path_filename, camera_path)

    def _handle_crop_params_message(self, message: NerfstudioMessage) -> None:
        """Handle crop parameters message from viewer."""
        assert isinstance(message, CropParamsMessage)
        self.control_panel.crop_viewport = message.crop_enabled
        self.control_panel.background_color = message.crop_bg_color
        center = np.array(message.crop_center)
        scale = np.array(message.crop_scale)
        crop_min = center - scale / 2.0
        crop_max = center + scale / 2.0
        self.control_panel.crop_min = tuple(crop_min.tolist())
        self.control_panel.crop_max = tuple(crop_max.tolist())

    def _handle_click_message(self, message: NerfstudioMessage) -> None:
        """Handle click message from viewer."""
        assert isinstance(message, ClickMessage)
        for controls in self.viewer_controls:
            controls.on_click(message)

    def _handle_time_condition_message(self, message: NerfstudioMessage) -> None:
        """Handle time conditioning message from viewer."""
        assert isinstance(message, TimeConditionMessage)
        self.control_panel.time = message.time

    @property
    def training_state(self) -> Literal["training", "paused", "completed"]:
        """Get training state flag."""
        if self.trainer is not None:
            return self.trainer.training_state
        return self.train_btn_state

    @training_state.setter
    def training_state(self, training_state: Literal["training", "paused", "completed"]) -> None:
        """Set training state flag."""
        if self.trainer is not None:
            self.trainer.training_state = training_state

    def get_camera(self, image_height: int, image_width: int) -> Optional[Cameras]:
        """
        Return a Cameras object representing the camera for the viewer given the provided image height and width
        """
        cam_msg: Optional[CameraMessage] = self.camera_message
        if cam_msg is None:
            return None
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
            times=torch.tensor([self.control_panel.time], dtype=torch.float32),
        )
        camera = camera.to(self.get_model().device)
        return camera

    def _pick_drawn_image_idxs(self, total_num: int) -> np.ndarray:
        """Determine indices of images to display in viewer.

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
        return np.linspace(0, total_num - 1, num_display_images, dtype=np.int32)

    def init_scene(
        self,
        train_dataset: InputDataset,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        self.viser_server.send_file_path_info(
            config_base_dir=self.log_filename.parents[0],
            data_base_dir=self.datapath,
            export_path_name=self.log_filename.parent.stem,
        )

        # total num of images
        num_images = len(train_dataset)
        if eval_dataset is not None:
            num_images += len(eval_dataset)

        # draw the training cameras and images
        image_indices = self._pick_drawn_image_idxs(num_images)
        for idx in image_indices[image_indices < len(train_dataset)].tolist():
            image = train_dataset[idx]["image"]
            bgr = image[..., [2, 1, 0]]
            camera_json = train_dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
            self.viser_server.add_dataset_image(idx=f"{idx:06d}", json=camera_json)

        # draw the eval cameras and images
        if eval_dataset is not None:
            image_indices = image_indices[image_indices >= len(train_dataset)] - len(train_dataset)
            for idx in image_indices.tolist():
                image = eval_dataset[idx]["image"]
                bgr = image[..., [2, 1, 0]]
                # color the eval image borders red
                # TODO: color the threejs frustum instead of changing the image itself like we are doing here
                t = int(min(image.shape[:2]) * 0.1)  # border thickness as 10% of min height or width resolution
                bc = torch.tensor((0, 0, 1.0))
                bgr[:t, :, :] = bc
                bgr[-t:, :, :] = bc
                bgr[:, -t:, :] = bc
                bgr[:, :t, :] = bc

                camera_json = eval_dataset.cameras.to_json(camera_idx=idx, image=bgr, max_size=100)
                self.viser_server.add_dataset_image(idx=f"{idx+len(train_dataset):06d}", json=camera_json)

        # draw the scene box (i.e., the bounding box)
        self.viser_server.update_scene_box(train_dataset.scene_box)

        # set the initial state whether to train or not
        self.train_btn_state = train_state
        self.viser_server.set_training_state(train_state)

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if self.camera_message is None:
            return

        if (
            self.trainer is not None
            and self.trainer.training_state == "training"
            and self.control_panel.train_util != 1
        ):
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.control_panel.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                self.render_statemachine.action(RenderAction("step", self.camera_message))

    def update_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_type_changed:
            self.control_panel.update_colormap_options(dimensions, dtype)
            self.output_type_changed = False

    def update_split_colormap_options(self, dimensions: int, dtype: type) -> None:
        """update the colormap options based on the current render

        Args:
            dimensions: the number of dimensions of the render
            dtype: the data type of the render
        """
        if self.output_split_type_changed:
            self.control_panel.update_split_colormap_options(dimensions, dtype)
            self.output_split_type_changed = False

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model

    def training_complete(self) -> None:
        """Called when training is complete."""
        self.training_state = "completed"
        self.viser_server.set_training_state("completed")
