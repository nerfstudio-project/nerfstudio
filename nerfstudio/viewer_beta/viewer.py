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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import numpy as np
import torch
import torchvision
import viser
import viser.theme
import viser.transforms as vtf

from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.decorators import check_main_thread, decorate_all
from nerfstudio.utils.writer import GLOBAL_BUFFER, EventName
from nerfstudio.viewer.server import viewer_utils
from nerfstudio.viewer_beta.control_panel import ControlPanel
from nerfstudio.viewer_beta.export_panel import populate_export_tab
from nerfstudio.viewer_beta.render_panel import populate_render_tab
from nerfstudio.viewer_beta.render_state_machine import RenderAction, RenderStateMachine
from nerfstudio.viewer_beta.utils import CameraState, parse_object
from nerfstudio.viewer_beta.viewer_elements import ViewerControl, ViewerElement

if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


@decorate_all([check_main_thread])
class Viewer:
    """Class to hold state for viewer variables

    Args:
        config: viewer setup configuration
        log_filename: filename to log viewer output to
        datapath: path to data
        pipeline: pipeline object to use
        trainer: trainer object to use
        share: print a shareable URL

    Attributes:
        viewer_url: url to open viewer
        viser_server: the viser server
    """

    viewer_url: str
    viser_server: viser.ViserServer
    camera_state: Optional[CameraState] = None

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
        share: bool = False,
    ):
        self.config = config
        self.trainer = trainer
        self.last_step = 0
        self.train_lock = train_lock
        self.pipeline = pipeline
        self.log_filename = log_filename
        self.datapath = datapath.parent if datapath.is_file() else datapath
        self.include_time = self.pipeline.datamanager.includes_time

        if self.config.websocket_port is None:
            websocket_port = viewer_utils.get_free_port(default_port=self.config.websocket_port_default)
        else:
            websocket_port = self.config.websocket_port
        self.log_filename.parent.mkdir(exist_ok=True)

        self.viewer_url = viewer_utils.get_viewer_url(websocket_port)

        # viewer specific variables
        self.output_type_changed = True
        self.output_split_type_changed = True
        self.step = 0
        self.train_btn_state: Literal["training", "paused", "completed"] = "training"
        self._prev_train_state: Literal["training", "paused", "completed"] = "training"

        self.client: Optional[viser.ClientHandle] = None
        self.viser_server = viser.ViserServer(host=config.websocket_host, port=websocket_port, share=share)
        buttons = (
            viser.theme.TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://nerf.studio",
            ),
            viser.theme.TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/nerfstudio-project/nerfstudio",
            ),
            viser.theme.TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://docs.nerf.studio",
            ),
        )
        image = viser.theme.TitlebarImage(
            image_url_light="https://docs.nerf.studio/en/latest/_static/imgs/logo.png",
            image_url_dark="https://docs.nerf.studio/en/latest/_static/imgs/logo-dark.png",
            image_alt="NerfStudio Logo",
            href="https://docs.nerf.studio/",
        )
        titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
        self.viser_server.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.render_statemachine = RenderStateMachine(self, VISER_NERFSTUDIO_SCALE_RATIO)
        self.viser_server.on_client_connect(self.handle_new_client)

        tabs = self.viser_server.add_gui_tab_group()
        control_tab = tabs.add_tab("Control", viser.Icon.SETTINGS)
        with control_tab:
            self.control_panel = ControlPanel(
                self.viser_server,
                self.include_time,
                VISER_NERFSTUDIO_SCALE_RATIO,
                self._interrupt_render,
                self._crop_params_update,
                self._output_type_change,
                self._output_split_type_change,
                self._toggle_training_state,
                self.set_camera_visibility,
            )
        config_path = self.log_filename.parents[0] / "config.yml"
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            populate_render_tab(self.viser_server, config_path, self.datapath, self.control_panel)

        with tabs.add_tab("Export", viser.Icon.PACKAGE_EXPORT):
            populate_export_tab(self.viser_server, self.control_panel, config_path)

        def nested_folder_install(folder_labels: List[str], element: ViewerElement):
            if len(folder_labels) == 0:
                element.install(self.viser_server)
                # also rewire the hook to rerender
                prev_cb = element.cb_hook
                element.cb_hook = lambda element: [prev_cb(element), self._interrupt_render(element)]
            else:
                with self.viser_server.add_gui_folder(folder_labels[0]):
                    nested_folder_install(folder_labels[1:], element)

        with control_tab:
            self.viewer_elements = []
            self.viewer_elements.extend(parse_object(pipeline, ViewerElement, "Custom Elements"))
            for param_path, element in self.viewer_elements:
                folder_labels = param_path.split("/")[:-1]
                nested_folder_install(folder_labels, element)

            # scrape the trainer/pipeline for any ViewerControl objects to initialize them
            self.viewer_controls: List[ViewerControl] = [
                e for (_, e) in parse_object(self.trainer, ViewerControl, "Custom Elements")
            ]
        for c in self.viewer_controls:
            c._setup(self)
        self.render_statemachine.start()

    def handle_new_client(self, client: viser.ClientHandle) -> None:
        self.client = client
        self.last_move_time = 0

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            assert self.client is not None
            with client.atomic():
                self.last_move_time = time.time()
                R = vtf.SO3(wxyz=self.client.camera.wxyz)
                R = R @ vtf.SO3.from_x_radians(np.pi)
                R = torch.tensor(R.as_matrix())
                pos = torch.tensor(self.client.camera.position, dtype=torch.float64) / VISER_NERFSTUDIO_SCALE_RATIO
                c2w = torch.concatenate([R, pos[:, None]], dim=1)
                self.camera_state = CameraState(fov=self.client.camera.fov, aspect=self.client.camera.aspect, c2w=c2w)
                self.render_statemachine.action(RenderAction("move", self.camera_state))

    def set_camera_visibility(self, visible: bool) -> None:
        """Toggle the visibility of the training cameras."""
        with self.viser_server.atomic():
            for idx in self.camera_handles:
                self.camera_handles[idx].visible = visible

    def update_camera_poses(self):
        # Update the train camera locations based on optimization
        assert self.camera_handles is not None
        idxs = list(self.camera_handles.keys())
        if hasattr(self.pipeline.datamanager, "train_camera_optimizer"):
            camera_optimizer = self.pipeline.datamanager.train_camera_optimizer
        else:
            camera_optimizer = self.pipeline.model.camera_optimizer
        with torch.no_grad():
            assert isinstance(camera_optimizer, CameraOptimizer)
            c2ws_delta = camera_optimizer(torch.tensor(idxs, device=camera_optimizer.device)).cpu().numpy()
        for idx in idxs:
            # both are numpy arrays
            c2w_orig = self.original_c2w[idx]
            c2w_delta = c2ws_delta[idx, ...]
            c2w = c2w_orig @ np.concatenate((c2w_delta, np.array([[0, 0, 0, 1]])), axis=0)
            R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
            R = R @ vtf.SO3.from_x_radians(np.pi)
            self.camera_handles[idx].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
            self.camera_handles[idx].wxyz = R.wxyz

    def _interrupt_render(self, _) -> None:
        """Interrupt current render."""
        if self.camera_state is not None:
            self.render_statemachine.action(RenderAction("rerender", self.camera_state))

    def _toggle_training_state(self, _) -> None:
        """Toggle the trainer's training state."""
        if self.trainer is not None:
            if self.trainer.training_state == "training":
                self.trainer.training_state = "paused"
            elif self.trainer.training_state == "paused":
                self.trainer.training_state = "training"

    def _crop_params_update(self, _) -> None:
        """Update crop parameters"""
        if self.camera_state is not None:
            self.render_statemachine.action(RenderAction("move", self.camera_state))

    def _output_type_change(self, _):
        self.output_type_changed = True

    def _output_split_type_change(self, _):
        self.output_split_type_changed = True

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
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        image_indices = self._pick_drawn_image_idxs(len(train_dataset))
        for idx in image_indices:
            image = train_dataset[idx]["image"]
            camera = train_dataset.cameras[idx]
            image_uint8 = (image * 255).detach().type(torch.uint8)
            image_uint8 = image_uint8.permute(2, 0, 1)
            image_uint8 = torchvision.transforms.functional.resize(image_uint8, 100)  # type: ignore
            image_uint8 = image_uint8.permute(1, 2, 0)
            image_uint8 = image_uint8.cpu().numpy()
            c2w = camera.camera_to_worlds.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                scale=0.1,
                aspect=float(camera.cx[0] / camera.cy[0]),
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
            )

            @camera_handle.on_click
            def _(event: viser.ClickEvent[viser.CameraFrustumHandle]) -> None:
                assert self.client is not None
                with self.client.atomic():
                    self.client.camera.position = event.target.position
                    self.client.camera.wxyz = event.target.wxyz

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9

    def update_scene(self, step: int, num_rays_per_batch: Optional[int] = None) -> None:
        """updates the scene based on the graph weights

        Args:
            step: iteration step of training
            num_rays_per_batch: number of rays per batch, used during training
        """
        self.step = step

        if self.camera_state is None:
            return
        # this stops training while moving to make the response smoother
        while time.time() - self.last_move_time < 0.1:
            time.sleep(0.05)
        # self.render_statemachine.action(RenderAction("static", self.camera_state))
        if self.trainer is not None and self.trainer.training_state == "training" and self.train_util != 1:
            if (
                EventName.TRAIN_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
                and EventName.VIS_RAYS_PER_SEC.value in GLOBAL_BUFFER["events"]
            ):
                train_s = GLOBAL_BUFFER["events"][EventName.TRAIN_RAYS_PER_SEC.value]["avg"]
                vis_s = GLOBAL_BUFFER["events"][EventName.VIS_RAYS_PER_SEC.value]["avg"]
                train_util = self.train_util
                vis_n = self.control_panel.max_res**2
                train_n = num_rays_per_batch
                train_time = train_n / train_s
                vis_time = vis_n / vis_s

                render_freq = train_util * vis_time / (train_time - train_util * train_time)
            else:
                render_freq = 30
            if step > self.last_step + render_freq:
                self.last_step = step
                self.render_statemachine.action(RenderAction("step", self.camera_state))
                self.update_camera_poses()
                self.control_panel.update_step(step)

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
