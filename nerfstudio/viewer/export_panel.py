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

from pathlib import Path
from typing import cast

import viser
import viser.transforms as vtf
import yaml
from typing_extensions import Literal, Tuple

from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.base_model import Model
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.viewer.control_panel import ControlPanel


def populate_export_tab(
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

    server.gui.add_markdown("<small>Export available after a checkpoint is saved (default minimum 2000 steps)</small>")
    with server.gui.add_folder("Splat"):
        populate_splat_tab(server, control_panel, config_path, viewing_gsplat)
    with server.gui.add_folder("Point Cloud"):
        populate_point_cloud_tab(server, control_panel, config_path, viewing_gsplat)
    with server.gui.add_folder("Mesh"):
        populate_mesh_tab(server, control_panel, config_path, viewing_gsplat)


def show_command_modal(
    client: viser.ClientHandle,
    what: Literal["mesh", "point cloud", "splat"],
    command: str,
) -> None:
    """Show a modal to each currently connected client.

    In the future, we should only show the modal to the client that pushes the
    generation button.
    """
    with client.gui.add_modal(what.title() + " Export") as modal:
        client.gui.add_markdown(
            "\n".join(
                [
                    f"To export a {what}, run the following from the command line:",
                    "",
                    "```",
                    command,
                    "```",
                ]
            )
        )
        close_button = client.gui.add_button("Close")

        @close_button.on_click
        def _(_) -> None:
            modal.close()


def get_crop_string(obb: OrientedBox, crop_viewport: bool) -> str:
    """Takes in an oriented bounding box and returns a string of the form "--obb_{center,rotation,scale}
    and each arg formatted with spaces around it
    """
    if not crop_viewport:
        return ""
    rpy = vtf.SO3.from_matrix(obb.R.numpy(force=True)).as_rpy_radians()
    rpy = [rpy.roll, rpy.pitch, rpy.yaw]
    pos = obb.T.squeeze().tolist()
    scale = obb.S.squeeze().tolist()
    rpystring = " ".join([f"{x:.10f}" for x in rpy])
    posstring = " ".join([f"{x:.10f}" for x in pos])
    scalestring = " ".join([f"{x:.10f}" for x in scale])
    return f"--obb_center {posstring} --obb_rotation {rpystring} --obb_scale {scalestring}"


Vec3f = Tuple[float, float, float]


def get_crop_tuple(obb: OrientedBox, crop_viewport: bool):
    """Takes in an oriented bounding box and returns tuples for obb_{center,rotation,scale}."""
    if not crop_viewport:
        return None, None, None
    rpy = vtf.SO3.from_matrix(obb.R.numpy(force=True)).as_rpy_radians()
    obb_rotation = [rpy.roll, rpy.pitch, rpy.yaw]
    obb_center = obb.T.squeeze().tolist()
    obb_scale = obb.S.squeeze().tolist()
    return (
        cast(Vec3f, tuple(obb_center)),
        cast(Vec3f, tuple(obb_rotation)),
        cast(Vec3f, tuple(obb_scale)),
    )


def populate_point_cloud_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
    config_path: Path,
    viewing_gsplat: bool,
) -> None:
    if not viewing_gsplat:
        server.gui.add_markdown("<small>Render depth, project to an oriented point cloud, and filter</small> ")
        num_points = server.gui.add_number("# Points", initial_value=1_000_000, min=1, max=None, step=1)
        world_frame = server.gui.add_checkbox(
            "Save in world frame",
            False,
            hint=(
                "If checked, saves the point cloud in the same frame as the original dataset. Otherwise, uses the "
                "scaled and reoriented coordinate space expected by the NeRF models."
            ),
        )
        remove_outliers = server.gui.add_checkbox("Remove outliers", True)
        normals = server.gui.add_dropdown(
            "Normals",
            # TODO: options here could depend on what's available to the model.
            ("open3d", "model_output"),
            initial_value="open3d",
            hint="Normal map source.",
        )

        output_dir = server.gui.add_text("Output Directory", initial_value="exports/pcd/")
        export_button = server.gui.add_button("Export", icon=viser.Icon.FILE_EXPORT)
        download_button = server.gui.add_button("Download Point Cloud", icon=viser.Icon.DOWNLOAD, disabled=True)
        generate_command = server.gui.add_button("Generate Command", icon=viser.Icon.TERMINAL_2)

        @export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            if config.load_dir is None:
                notif = client.add_notification(
                    title="Export unsuccessful",
                    body="Cannot export before 2000 training steps.",
                    loading=False,
                    with_close_button=True,
                    color="red",
                )
                return

            notif = client.add_notification(
                title="Exporting point cloud",
                body="File will be saved under " + str(output_dir.value),
                loading=True,
                with_close_button=False,
            )

            if control_panel.crop_obb is not None and control_panel.crop_viewport:
                obb_center, obb_rotation, obb_scale = get_crop_tuple(
                    control_panel.crop_obb, control_panel.crop_viewport
                )
            else:
                obb_center, obb_rotation, obb_scale = None, None, None

            from nerfstudio.scripts.exporter import ExportPointCloud

            export = ExportPointCloud(
                load_config=config_path,
                output_dir=Path(output_dir.value),
                num_points=num_points.value,
                remove_outliers=remove_outliers.value,
                normal_method=normals.value,
                save_world_frame=world_frame.value,
                obb_center=obb_center,
                obb_rotation=obb_rotation,
                obb_scale=obb_scale,
            )
            export.main()

            if export._complete:
                notif.title = "Export complete!"
                notif.body = "File saved under " + str(output_dir.value)
                notif.loading = False
                notif.with_close_button = True

                download_button.disabled = False

        @download_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            with open(str(output_dir.value) + "point_cloud.ply", "rb") as ply_file:
                ply_bytes = ply_file.read()

            client.send_file_download("point_cloud.ply", ply_bytes)

        @generate_command.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            command = " ".join(
                [
                    "ns-export pointcloud",
                    f"--load-config {config_path}",
                    f"--output-dir {output_dir.value}",
                    f"--num-points {num_points.value}",
                    f"--remove-outliers {remove_outliers.value}",
                    f"--normal-method {normals.value}",
                    f"--save-world-frame {world_frame.value}",
                    get_crop_string(control_panel.crop_obb, control_panel.crop_viewport),
                ]
            )
            show_command_modal(event.client, "point cloud", command)

    else:
        server.gui.add_markdown("<small>Point cloud export is not currently supported with Gaussian Splatting</small>")


def populate_mesh_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
    config_path: Path,
    viewing_gsplat: bool,
) -> None:
    if not viewing_gsplat:
        server.gui.add_markdown(
            "<small>Render depth, project to an oriented point cloud, and run Poisson surface reconstruction</small>"
        )

        normals = server.gui.add_dropdown(
            "Normals",
            ("open3d", "model_output"),
            initial_value="open3d",
            hint="Source for normal maps.",
        )
        num_faces = server.gui.add_number("# Faces", initial_value=50_000, min=1)
        texture_resolution = server.gui.add_number("Texture Resolution", min=8, initial_value=2048)
        num_points = server.gui.add_number("# Points", initial_value=1_000_000, min=1, max=None, step=1)
        remove_outliers = server.gui.add_checkbox("Remove outliers", True)

        output_dir = server.gui.add_text("Output Directory", initial_value="exports/mesh/")
        export_button = server.gui.add_button("Export", icon=viser.Icon.FILE_EXPORT)
        download_button = server.gui.add_button("Download Mesh", icon=viser.Icon.DOWNLOAD, disabled=True)
        generate_command = server.gui.add_button("Generate Command", icon=viser.Icon.TERMINAL_2)

        @export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            if config.load_dir is None:
                notif = client.add_notification(
                    title="Export unsuccessful",
                    body="Cannot export before 2000 training steps.",
                    loading=False,
                    with_close_button=True,
                    color="red",
                )
                return

            notif = client.add_notification(
                title="Exporting poisson mesh",
                body="File will be saved under " + str(output_dir.value),
                loading=True,
                with_close_button=False,
            )

            if control_panel.crop_obb is not None and control_panel.crop_viewport:
                obb_center, obb_rotation, obb_scale = get_crop_tuple(
                    control_panel.crop_obb, control_panel.crop_viewport
                )
            else:
                obb_center, obb_rotation, obb_scale = None, None, None

            from nerfstudio.scripts.exporter import ExportPoissonMesh

            export = ExportPoissonMesh(
                load_config=config_path,
                output_dir=Path(output_dir.value),
                target_num_faces=num_faces.value,
                num_pixels_per_side=texture_resolution.value,
                num_points=num_points.value,
                remove_outliers=remove_outliers.value,
                normal_method=normals.value,
                obb_center=obb_center,
                obb_rotation=obb_rotation,
                obb_scale=obb_scale,
            )
            export.main()

            if export._complete:
                notif.title = "Export complete!"
                notif.body = "File saved under " + str(output_dir.value)
                notif.loading = False
                notif.with_close_button = True

                download_button.disabled = False

        @download_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            with open(str(output_dir.value) + "poisson_mesh.ply", "rb") as ply_file:
                ply_bytes = ply_file.read()

            client.send_file_download("poisson_mesh.ply", ply_bytes)

        @generate_command.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            command = " ".join(
                [
                    "ns-export poisson",
                    f"--load-config {config_path}",
                    f"--output-dir {output_dir.value}",
                    f"--target-num-faces {num_faces.value}",
                    f"--num-pixels-per-side {texture_resolution.value}",
                    f"--num-points {num_points.value}",
                    f"--remove-outliers {remove_outliers.value}",
                    f"--normal-method {normals.value}",
                    get_crop_string(control_panel.crop_obb, control_panel.crop_viewport),
                ]
            )
            show_command_modal(event.client, "mesh", command)

    else:
        server.gui.add_markdown("<small>Mesh export is not currently supported with Gaussian Splatting</small>")


def populate_splat_tab(
    server: viser.ViserServer,
    control_panel: ControlPanel,
    config_path: Path,
    viewing_gsplat: bool,
) -> None:
    if viewing_gsplat:
        server.gui.add_markdown("<small>Generate ply export of Gaussian Splat</small>")
        output_dir = server.gui.add_text("Output Directory", initial_value="exports/splat/")
        export_button = server.gui.add_button("Export", icon=viser.Icon.FILE_EXPORT)
        download_button = server.gui.add_button("Download Splat", icon=viser.Icon.DOWNLOAD, disabled=True)
        generate_command = server.gui.add_button("Generate Command", icon=viser.Icon.TERMINAL_2)

        @export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
            if config.load_dir is None:
                notif = client.add_notification(
                    title="Export unsuccessful",
                    body="Cannot export before 2000 training steps.",
                    loading=False,
                    with_close_button=True,
                    color="red",
                )
                return

            notif = client.add_notification(
                title="Exporting gaussian splat",
                body="File will be saved under " + str(output_dir.value),
                loading=True,
                with_close_button=False,
            )

            if control_panel.crop_obb is not None and control_panel.crop_viewport:
                obb_center, obb_rotation, obb_scale = get_crop_tuple(
                    control_panel.crop_obb, control_panel.crop_viewport
                )
            else:
                obb_center, obb_rotation, obb_scale = None, None, None

            from nerfstudio.scripts.exporter import ExportGaussianSplat

            export = ExportGaussianSplat(
                load_config=config_path,
                output_dir=Path(output_dir.value),
                obb_center=obb_center,
                obb_rotation=obb_rotation,
                obb_scale=obb_scale,
            )
            export.main()

            if export._complete:
                notif.title = "Export complete!"
                notif.body = "File saved under " + str(output_dir.value)
                notif.loading = False
                notif.with_close_button = True

                download_button.disabled = False

        @download_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            with open(str(output_dir.value) + "splat.ply", "rb") as ply_file:
                ply_bytes = ply_file.read()

            client.send_file_download("splat.ply", ply_bytes)

        @generate_command.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            command = " ".join(
                [
                    "ns-export gaussian-splat",
                    f"--load-config {config_path}",
                    f"--output-dir {output_dir.value}",
                    get_crop_string(control_panel.crop_obb, control_panel.crop_viewport),
                ]
            )
            show_command_modal(event.client, "splat", command)

    else:
        server.gui.add_markdown("<small>Splat export is only supported with Gaussian Splatting methods</small>")
