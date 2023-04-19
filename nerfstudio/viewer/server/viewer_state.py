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

""" Manage the state of the viewer """
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from rich import box, style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.control_panel import ControlPanel
from nerfstudio.viewer.server.gui_utils import get_viewer_elements
from nerfstudio.viewer.server.render_state_machine import (
    RenderAction,
    RenderStateMachine,
)
from nerfstudio.viewer.server.viewer_elements import ViewerElement
from nerfstudio.viewer.viser import ViserServer
from nerfstudio.viewer.viser.messages import (
    CameraMessage,
    CameraPathOptionsRequest,
    CameraPathPayloadMessage,
    CropParamsMessage,
    IsTrainingMessage,
    NerfstudioMessage,
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

        # viewer specific variables
        self.prev_moving = False
        self.output_type_changed = True
        self.check_done_render = True
        self.step = 0
        self.camera_moving = False
        self.prev_camera_timestamp = 0
        self.static_fps = 1.0
        self.train_btn_state = True

        self.camera_message = None

        self.viser_server = ViserServer(host="0.0.0.0", port=websocket_port)

        self.viser_server.register_handler(IsTrainingMessage, self._handle_is_training)
        self.viser_server.register_handler(SaveCheckpointMessage, self._handle_save_checkpoint)
        self.viser_server.register_handler(CameraMessage, self._handle_camera_update)
        self.viser_server.register_handler(CameraPathOptionsRequest, self._handle_camera_path_option_request)
        self.viser_server.register_handler(CameraPathPayloadMessage, self._handle_camera_path_payload)
        self.viser_server.register_handler(CropParamsMessage, self._handle_crop_params_message)

        self.control_panel = ControlPanel(self._interrupt_render, self._crop_params_update, self._output_type_change)
        self.control_panel.install(self.viser_server)

        def nested_folder_install(folder_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [self._interrupt_render(element), prev_cb(element)]
            else:
                with self.viser_server.gui_folder(folder_labels[0]):
                    nested_folder_install(folder_labels[1:], element)

        self.viewer_elements = get_viewer_elements(self.pipeline)
        for param_path, element in self.viewer_elements:
            folder_labels = param_path.split("/")[:-1]
            nested_folder_install(folder_labels, element)

        self.render_statemachine = RenderStateMachine(self)
        self.render_statemachine.start()

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _interrupt_render(self, _) -> None:
        """Interrupt current render."""
        if self.camera_message is not None:
            self.render_statemachine.action(RenderAction("rerender", self.camera_message))

    def _crop_params_update(self, _) -> None:
        """Update crop parameters"""
        self.render_statemachine.action(RenderAction("rerender", self.camera_message))
        crop_min = torch.tensor(self.control_panel.crop_min, dtype=torch.float32)
        crop_max = torch.tensor(self.control_panel.crop_max, dtype=torch.float32)
        scene_box = SceneBox(aabb=torch.stack([crop_min, crop_max], dim=0))
        self.viser_server.update_scene_box(scene_box)
        crop_scale = crop_max - crop_min
        crop_center = crop_max + crop_min
        self.viser_server.send_crop_params(
            crop_enabled=self.control_panel.crop_viewport,
            crop_bg_color=self.control_panel.background_color,
            crop_scale=tuple(crop_scale.tolist()),
            crop_center=tuple(crop_center.tolist()),
        )

    def _handle_is_training(self, message: NerfstudioMessage) -> None:
        """Handle is_training message from viewer."""
        assert isinstance(message, IsTrainingMessage)
        self.is_training = message.is_training
        self.train_btn_state = message.is_training
        self.viser_server.set_is_training(message.is_training)

    def _handle_save_checkpoint(self, message: NerfstudioMessage) -> None:
        """Handle is_training message from viewer."""
        assert isinstance(message, SaveCheckpointMessage)
        if self.trainer is not None:
            self.trainer.save_checkpoint(self.step)

    def _handle_camera_update(self, message: NerfstudioMessage) -> None:
        """Handle camera update message from viewer."""
        assert isinstance(message, CameraMessage)
        self.camera_message = message
        self.camera_moving = message.is_moving
        if message.is_moving:
            self.render_statemachine.action(RenderAction("move", self.camera_message))
            self.is_training = False
        else:
            self.render_statemachine.action(RenderAction("static", self.camera_message))
            self.is_training = self.train_btn_state

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

    @property
    def is_training(self) -> bool:
        """Get is_training flag from viewer."""
        if self.trainer is not None:
            return self.trainer.is_training
        return False

    @is_training.setter
    def is_training(self, is_training: bool) -> None:
        """Set is_training flag in viewer."""
        if self.trainer is not None:
            self.trainer.is_training = is_training

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
        self.train_btn_state = start_train
        self.viser_server.set_is_training(start_train)

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if self.camera_message is None:
            return

        if self.trainer is not None and self.trainer.is_training and self.control_panel.train_util != 1:
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

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model
