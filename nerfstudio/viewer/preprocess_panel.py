"""Prepross panel for the viewer"""
"""(In Progress: JiChen)"""

# from __future__ import annotations

from pathlib import Path

import viser
import viser.transforms as vtf
from typing_extensions import Literal

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.viewer.control_panel import ControlPanel


def populate_preprocess_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
    config_path: Path,
    viewer_model: Model,
) -> None:
    viewing_gsplat = isinstance(viewer_model, SplatfactoModel)
    if not viewing_gsplat:
        crop_output = server.gui.add_checkbox("Use Crop", False)

        @crop_output.on_update
        def _(_) -> None:
            control_panel.crop_viewport = crop_output.value

    with server.gui.add_folder("Upload Images"):
        populate_upload_images_tab(server, control_panel)
    with server.gui.add_folder("Upload Video"):
        populate_upload_video_tab(server, control_panel)
    with server.gui.add_folder("Filter blurred images"):
        populate_filter_blurred_image_tab(server, control_panel)
    with server.gui.add_folder("Colmap"):
        populate_colmap_tab(server, control_panel)


# 上傳圖片欄位
def populate_upload_images_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
) -> None:
    server.gui.add_markdown("<small>上傳圖片</small> ")
    input_dir = server.gui.add_text("Input Directory", initial_value="images_path/")
    generate_command = server.gui.add_button("Upload", icon=viser.Icon.TERMINAL_2)
    @generate_command.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None

# 上傳影片欄位
def populate_upload_video_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
) -> None:
    server.gui.add_markdown("<small>上傳影片</small> ")
    input_dir = server.gui.add_text("Input Directory", initial_value="video_path/")
    generate_command = server.gui.add_button("Upload", icon=viser.Icon.TERMINAL_2)
    @generate_command.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None

# 模糊影像處理欄位
def populate_filter_blurred_image_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
) -> None:
    server.gui.add_markdown("<small>過濾輸入的圖片集</small> ")
    image_num = server.gui.add_text("Num of Images:", initial_value="0")
    thresholding = server.gui.add_slider(
                    "Thresholding",
                    0.0,
                    10.0,
                    step=0.1,
                    initial_value=2.5,
                )
    blurred_image_num = server.gui.add_text("Num of Images:", initial_value="0")
    filter_blurred_images = server.gui.add_button("Filter Blurred Image", icon=viser.Icon.TERMINAL_2)
    @filter_blurred_images.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None

    delete_blurred_images = server.gui.add_button("Delete Blurred Image", icon=viser.Icon.TERMINAL_2)
    @delete_blurred_images.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None

    fix_blurred_images = server.gui.add_button("Fix Blurred Image", icon=viser.Icon.TERMINAL_2)
    @fix_blurred_images.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None

# Colmap生成點雲欄位
def populate_colmap_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
) -> None:
    server.gui.add_markdown("<small>進行Colmap，生成稀疏點雲</small> ")
    intput_dir = server.gui.add_text("Input Directory", initial_value="input_path/")
    run_colmap = server.gui.add_button("Run Colmap", icon=viser.Icon.TERMINAL_2)
    @run_colmap.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
