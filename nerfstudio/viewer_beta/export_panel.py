from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh.creation
import viser
from typing_extensions import Literal, assert_never


def populate_export_tab(server: viser.ViserServer) -> None:
    crop_options = make_crop_options(server)

    server.add_gui_markdown("")  # Vertical whitespace.

    export_tabs = server.add_gui_tab_group()

    # with server.add_gui_folder("Point Cloud"):
    with export_tabs.add_tab("Points", viser.Icon.GRAIN):
        populate_point_cloud_tab(server, crop_options, Path("./TODO"))
    # with server.add_gui_folder("Mesh (Poisson, recommended)"):
    with export_tabs.add_tab("Mesh", viser.Icon.HEXAGON_LETTER_P):
        populate_mesh_tab(server, crop_options, Path("./TODO"), "poisson")
    # with server.add_gui_folder("Mesh (TSDF)"):
    with export_tabs.add_tab("Mesh (TSDF)", viser.Icon.HEXAGON_LETTER_T):
        populate_mesh_tab(server, crop_options, Path("./TODO"), "tsdf")


def show_command_modal(client: viser.ClientHandle, what: Literal["mesh", "point cloud"], command: str) -> None:
    """Show a modal to each currently connected client.

    In the future, we should only show the modal to the client that pushes the
    generation button.
    """
    with client.add_gui_modal(what.title() + " Export") as modal:
        client.add_gui_markdown(
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
        close_button = client.add_gui_button("Close")

        @close_button.on_click
        def _(_) -> None:
            modal.close()


def populate_point_cloud_tab(
    server: viser.ViserServer,
    crop_options: CropOptionHandles,
    config_path: Path,
) -> None:
    server.add_gui_markdown(
        "<small>Render out depth and normal maps, project to an oriented point cloud, and filter.</small> "
    )
    num_points = server.add_gui_number("# Points", initial_value=1_000_000, min=1, max=None, step=1)
    remove_outliers = server.add_gui_checkbox("Remove outliers", True)
    normals = server.add_gui_dropdown(
        "Normals",
        # TODO: options here could depend on what's available to the model.
        ("open3d", "model_output"),
        initial_value="model_output",
        hint="Normal map source.",
    )
    output_dir = server.add_gui_text("Output Directory", initial_value="exports/pcd/")
    generate_command = server.add_gui_button("Generate Command")

    @generate_command.on_click
    def _(event: viser.GuiEvent) -> None:
        for client in server.get_clients().values():
            command = " ".join(
                [
                    "ns-export pointcloud",
                    f"--load-config {config_path}",
                    f"--output-dir {output_dir.value}",
                    f"--num-points {num_points.value}",
                    f"--remove-outliers {remove_outliers.value}",
                    f"--normal-method {normals.value}",
                    *crop_options.get_bbox_args(),
                ]
            )
            show_command_modal(viser.client, "point cloud", command)


def populate_mesh_tab(
    server: viser.ViserServer,
    crop_options: CropOptionHandles,
    config_path: Path,
    method: Literal["tsdf", "poisson"],
) -> None:
    server.add_gui_markdown(
        "<small>Recommended mesh export method. Renders out depth and normal maps, projects to an oriented point cloud, and runs Poisson surface reconstruction.</small>"
        if method == "poisson"
        else "<small>Mesh export via TSDF fusion. Results are typically lower quality than Poisson-based reconstruction.</small>"
    )

    normals = server.add_gui_dropdown(
        "Normals",
        ("open3d", "model_output"),
        initial_value="model_output",
        hint="Source for normal maps.",
    )
    num_faces = server.add_gui_number("# Faces", initial_value=50_000, min=1)
    texture_resolution = server.add_gui_number("Texture Resolution", min=8, initial_value=2048)
    output_directory = server.add_gui_text("Output Directory", initial_value="exports/mesh/")

    if method == "tsdf":
        generate_command = server.add_gui_button("Generate Command")

        @generate_command.on_click
        def _(event: viser.GuiEvent) -> None:
            command = " ".join(
                [
                    "ns-export tsdf",
                    f"--load-config {config_path}" f"--output-dir {output_directory.value}",
                    f"--target-num-faces {num_faces.value}",
                    f"--num-pixels-per-side {texture_resolution.value}",
                    *crop_options.get_bbox_args(),
                ]
            )
            show_command_modal(event.client, "mesh", command)

    elif method == "poisson":
        num_points = server.add_gui_number("# Points", initial_value=1_000_000, min=1, max=None, step=1)
        remove_outliers = server.add_gui_checkbox("Remove outliers", True)
        normals = server.add_gui_dropdown(
            "Normals",
            # TODO: options here could depend on what's available to the model.
            ("open3d", "model_output"),
            initial_value="model_output",
            hint="Source for normal maps.",
        )

        generate_command = server.add_gui_button("Generate Command")

        @generate_command.on_click
        def _(event: viser.GuiEvent) -> None:
            command = " ".join(
                [
                    "ns-export poisson",
                    f"--load-config {config_path}" f"--output-dir {output_directory.value}",
                    f"--target-num-faces {num_faces.value}",
                    f"--num-pixels-per-side {texture_resolution.value}",
                    f"--num-points {num_points.value}",
                    f"--remove-outliers {remove_outliers.value}",
                    f"--normal-method {normals.value}",
                    *crop_options.get_bbox_args(),
                ]
            )
            show_command_modal(event.client, "mesh", command)


@dataclasses.dataclass
class CropOptionHandles:
    crop_enabled: viser.GuiHandle[bool]
    crop_show: viser.GuiHandle[bool]
    crop_center: viser.GuiHandle[Tuple[float, float, float]]
    crop_scale: viser.GuiHandle[Tuple[float, float, float]]

    def get_bbox_args(self) -> List[str]:
        """Generate bounding box arguments that are shared between point cloud and mesh exports."""
        out = [
            f"--use-bounding-box {self.crop_enabled.value}",
        ]
        if self.crop_enabled.value:
            bbox_min = np.array(self.crop_center.value) - np.array(self.crop_scale.value) / 2.0
            bbox_max = np.array(self.crop_center.value) + np.array(self.crop_scale.value) / 2.0
            out.extend(
                [
                    f"--bounding-box-min {bbox_min}",
                    f"--bounding-box-max {bbox_max}",
                ]
            )
        return out


def make_crop_options(server: viser.ViserServer) -> CropOptionHandles:
    """Make crop option inputs, and return the relevant GUI handles."""
    handles = CropOptionHandles(
        crop_enabled=server.add_gui_checkbox(
            "Crop",
            False,
            hint="Turn cropping on or off. Applies to both point cloud and mesh exports.",
        ),
        crop_show=server.add_gui_checkbox("• Show", True),
        crop_center=server.add_gui_vector3("• Center", initial_value=(0.0, 0.0, 0.0), step=1e-3),
        crop_scale=server.add_gui_vector3("• Scale", initial_value=(2.0, 2.0, 2.0), step=1e-3),
    )
    crop_vis: Optional[viser.SceneNodeHandle] = None

    def update_crop_vis() -> None:
        nonlocal crop_vis

        if not handles.crop_enabled.value or not handles.crop_show.value:
            if crop_vis is not None:
                crop_vis.remove()
                crop_vis = None
            return

        box_mesh = trimesh.creation.box(extents=np.array(handles.crop_scale.value))
        if handles.crop_show.value:
            crop_vis = server.add_mesh_simple(
                "/export/crop_vis",
                box_mesh.vertices + np.array(handles.crop_center.value),
                box_mesh.faces,
                color=(255, 225, 0),
                opacity=0.1,
                side="double",
            )

    @handles.crop_enabled.on_update
    def _(_) -> None:
        handles.crop_show.disabled = not handles.crop_enabled.value
        handles.crop_center.disabled = not handles.crop_enabled.value
        handles.crop_scale.disabled = not handles.crop_enabled.value
        update_crop_vis()

    _(handles.crop_enabled)

    handles.crop_show.on_update(lambda _: update_crop_vis())
    handles.crop_center.on_update(lambda _: update_crop_vis())
    handles.crop_scale.on_update(lambda _: update_crop_vis())

    return handles
