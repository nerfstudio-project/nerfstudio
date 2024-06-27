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

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import viser

from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.viewer.control_panel import ControlPanel
from nerfstudio.viewer.viewer_elements import ViewerPlot

@dataclasses.dataclass
class MetricsPanel:
    server: viser.ViserServer
    control_panel: ControlPanel
    config_path: Path
    viewer_model: Model

def populate_metrics_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
    config_path: Path,
    viewer_model: Model,
) -> None:
    viewing_gsplat = isinstance(viewer_model, SplatfactoModel)

    with server.add_gui_folder("Training Metrics"):
        populate_train_metrics_tab(server, control_panel, config_path, viewing_gsplat)
    # with server.add_gui_folder("Training Loss"):
    #     populate_train_loss_tab(server, control_panel, config_path, viewing_gsplat)
    #training rays?
    # with server.add_gui_folder("Eval Metrics"):
    #     populate_eval_metrics_tab(server, control_panel, config_path, viewing_gsplat)
    # with server.add_gui_folder("Eval Metrics (All Images)"):
    #     populate_eval_metrics_all_images_tab(server, control_panel, config_path, viewing_gsplat)
    # with server.add_gui_folder("Eval Loss"):
    #     populate_eval_loss_tab(server, control_panel, config_path, viewing_gsplat)


def populate_train_metrics_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
    config_path: Path,
    viewing_gsplat: bool,
) -> None:
    ViewerPlot()