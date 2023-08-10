"""Theming

Viser is adding support for theming. Work-in-progress.
"""

from __future__ import annotations

from datetime import datetime

import viser


def populate_render_tab(server: viser.ViserServer) -> None:
    render_cameras = []
    server.add_gui_markdown("work-in-progress!")

    with server.add_gui_folder("Frames"):
        add_camera_button = server.add_gui_button("Add camera")
        server.add_gui_slider("Smoothness", min=0.0, max=1.0, initial_value=0.0, step=1e-3)
        server.add_gui_checkbox("Loop", False)
        server.add_gui_checkbox("Playing", False)
        server.add_gui_button_group("Playback", ["<", "►", ">"])

    @add_camera_button.on_click
    def _(_) -> None:
        camera = None
        last_update = 0
        for client in server.get_clients().values():
            if client.camera.update_timestamp > last_update:
                camera = client.camera
                last_update = client.camera.update_timestamp

        assert camera is not None
        server.add_camera_frustum(
            f"/render_cameras/{len(render_cameras)}",
            fov=camera.fov,
            aspect=camera.aspect,
            wxyz=camera.wxyz,
            position=camera.position,
            color=(200, 0, 0),
        )
        render_cameras.append(None)  # TODO

    with server.add_gui_folder("Output"):
        export_name = server.add_gui_text("Export name", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        resolution = server.add_gui_vector2("Resolution", initial_value=(1920, 1080), step=1)
        duration = server.add_gui_number("Duration (sec)", initial_value=4.0, min=0.01)
        framerate = server.add_gui_number("FPS", initial_value=24.0, min=1.0)
        camera_type = server.add_gui_dropdown("Camera type", ("Perspective", "Fisheye", "Equirectangular"))
        render_button = server.add_gui_button("Render")
