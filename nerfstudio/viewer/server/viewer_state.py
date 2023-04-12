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
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.io import load_from_json, write_to_json
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer.server.control_panel import ControlPanel
from nerfstudio.viewer.server.gui_utils import get_viewer_elements
from nerfstudio.viewer.server.render_state_machine import (
    RenderAction,
    RenderStateMachine,
)
from nerfstudio.viewer.server.subprocess import get_free_port
from nerfstudio.viewer.server.utils import get_intrinsics_matrix_and_camera_to_world_h
from nerfstudio.viewer.server.viewer_param import ViewerElement
from nerfstudio.viewer.viser import ViserServer
from nerfstudio.viewer.viser._messages import (
    CameraMessage,
    CameraPathOptionsRequest,
    CameraPathPayloadMessage,
    CropParamsMessage,
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
        train_lock: Optional[threading.Lock] = None,
    ):
        self.config = config
        self.trainer = trainer
        self.train_lock = train_lock
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
        self.check_done_render = True
        self.step = 0
        self.camera_moving = False
        self.prev_camera_timestamp = 0
        self.static_fps = 1.0
        self.train_btn_state = True

        self.output_list = None
        self.camera_message = None

        self.viser_server = ViserServer(host="localhost", port=websocket_port)

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
        self.render_statemachine.action(RenderAction("rerender", self.camera_message))

    def _crop_params_update(self, _) -> None:
        """Update crop parameters"""
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

    def _handle_is_training(self, message: Message) -> None:
        """Handle is_training message from viewer."""
        assert isinstance(message, IsTrainingMessage)
        self.is_training = message.is_training
        self.train_btn_state = message.is_training
        self.viser_server.set_is_training(message.is_training)

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
        if message.is_moving:
            self.render_statemachine.action(RenderAction("move", self.camera_message))
            self.is_training = False
        else:
            self.render_statemachine.action(RenderAction("static", self.camera_message))
            self.is_training = self.train_btn_state

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

    def _handle_crop_params_message(self, message: Message) -> None:
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

    def update_scene(self, step: int, num_rays_per_batch: int) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            pipeline: the method pipeline
            num_rays_per_batch: number of rays per batch
        """
        self.step = step

        if self.camera_message is None:
            return

        if self.trainer is not None and self.trainer.is_training:
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
                self.render_statemachine.action(RenderAction("step", self.camera_message))

    def get_model(self) -> Model:
        """Returns the model."""
        return self.pipeline.model
