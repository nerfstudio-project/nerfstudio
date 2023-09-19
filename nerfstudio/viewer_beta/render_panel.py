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
import colorsys
import dataclasses
import threading
import time
from typing import Dict, List, Optional, Tuple
import datetime
from nerfstudio.viewer_beta.control_panel import ControlPanel
import numpy as onp
import splines
import splines.quaternion
import viser
import json
import viser.transforms as tf


@dataclasses.dataclass
class Keyframe:
    position: onp.ndarray
    wxyz: onp.ndarray
    override_fov_enabled: bool
    override_fov_value: float
    aspect: float

    @staticmethod
    def from_camera(camera: viser.CameraHandle, aspect: float) -> Keyframe:
        return Keyframe(
            camera.position,
            camera.wxyz,
            override_fov_enabled=False,
            override_fov_value=camera.fov,
            aspect=aspect,
        )


class CameraPath:
    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._keyframes: Dict[int, Tuple[Keyframe, viser.CameraFrustumHandle]] = {}
        self._keyframe_counter: int = 0
        self._spline: Optional[viser.SceneNodeHandle] = None
        self._camera_edit_panel: Optional[viser.Gui3dContainerHandle] = None

        self._orientation_spline: Optional[splines.quaternion.KochanekBartels] = None
        self._position_spline: Optional[splines.KochanekBartels] = None
        self._fov_spline: Optional[splines.KochanekBartels] = None
        self._keyframes_visible: bool = True

        # These parameters should be overridden externally.
        self.loop: bool = False
        self.smoothness: float = 0.5  # Tension / alpha term.
        self.default_fov: float = 0.0

    def set_keyframes_visible(self, visible: bool) -> None:
        self._keyframes_visible = visible
        for keyframe in self._keyframes.values():
            keyframe[1].visible = visible

    def add_camera(self, keyframe: Keyframe, keyframe_index: Optional[int] = None) -> None:
        """Add a new camera, or replace an old one if `keyframe_index` is passed in."""
        server = self._server

        # Add a keyframe if we aren't replacing an existing one.
        if keyframe_index is None:
            keyframe_index = self._keyframe_counter
            self._keyframe_counter += 1

        frustum_handle = server.add_camera_frustum(
            f"/render_cameras/{keyframe_index}",
            fov=keyframe.override_fov_value if keyframe.override_fov_enabled else self.default_fov,
            aspect=keyframe.aspect,
            scale=0.1,
            color=(127, 127, 127),
            wxyz=keyframe.wxyz,
            position=keyframe.position,
            visible=self._keyframes_visible,
        )

        @frustum_handle.on_click
        def _(_) -> None:
            with server.add_3d_gui_container(
                "/camera_edit_panel",
                wxyz=keyframe.wxyz,
                position=keyframe.position,
            ) as camera_edit_panel:
                self._camera_edit_panel = camera_edit_panel
                override_fov = server.add_gui_checkbox("Override FOV", initial_value=keyframe.override_fov_enabled)
                override_fov_degrees = server.add_gui_slider(
                    "Override FOV (degrees)",
                    5.0,
                    175.0,
                    step=0.1,
                    initial_value=keyframe.override_fov_value * 180.0 / onp.pi,
                    disabled=not keyframe.override_fov_enabled,
                )
                delete_button = server.add_gui_button("Delete", color="red", icon=viser.Icon.TRASH)
                go_to_button = server.add_gui_button("Go to")
                close_button = server.add_gui_button("Close")

                @override_fov.on_update
                def _(_) -> None:
                    keyframe.override_fov_enabled = override_fov.value
                    override_fov_degrees.disabled = not override_fov.value
                    self.add_camera(keyframe, keyframe_index)

                @override_fov_degrees.on_update
                def _(_) -> None:
                    keyframe.override_fov_value = override_fov_degrees.value / 180.0 * onp.pi
                    self.add_camera(keyframe, keyframe_index)

                @delete_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    assert event.client is not None
                    with event.client.add_gui_modal("Confirm") as modal:
                        event.client.add_gui_markdown("Delete keyframe?")
                        confirm_button = event.client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
                        exit_button = event.client.add_gui_button("Cancel")

                        @confirm_button.on_click
                        def _(_) -> None:
                            assert camera_edit_panel is not None

                            keyframe_id = None
                            for i, keyframe_tuple in self._keyframes.items():
                                if keyframe_tuple[1] is frustum_handle:
                                    keyframe_id = i
                                    break
                            assert keyframe_id is not None

                            self._keyframes.pop(keyframe_id)
                            frustum_handle.remove()
                            camera_edit_panel.remove()
                            modal.close()
                            self.update_spline()

                        @exit_button.on_click
                        def _(_) -> None:
                            modal.close()

                @go_to_button.on_click
                def _(event: viser.GuiEvent) -> None:
                    assert event.client is not None
                    client = event.client
                    T_world_current = tf.SE3.from_rotation_and_translation(
                        tf.SO3(client.camera.wxyz), client.camera.position
                    )
                    T_world_target = tf.SE3.from_rotation_and_translation(
                        tf.SO3(keyframe.wxyz), keyframe.position
                    ) @ tf.SE3.from_translation(onp.array([0.0, 0.0, -0.5]))

                    T_current_target = T_world_current.inverse() @ T_world_target

                    for j in range(10):
                        T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / 9.0)

                        # Important bit: we atomically set both the orientation and the position
                        # of the camera.
                        with client.atomic():
                            client.camera.wxyz = T_world_set.rotation().wxyz
                            client.camera.position = T_world_set.translation()
                        time.sleep(1.0 / 30.0)

                @close_button.on_click
                def _(_) -> None:
                    assert camera_edit_panel is not None
                    camera_edit_panel.remove()

        self._keyframes[keyframe_index] = (keyframe, frustum_handle)

    def update_aspect(self, aspect: float) -> None:
        for keyframe_index, frame in self._keyframes.items():
            frame = dataclasses.replace(frame[0], aspect=aspect)
            self.add_camera(frame, keyframe_index=keyframe_index)

    def reset(self) -> None:
        for frame in self._keyframes.values():
            frame[1].remove()
        self._keyframes.clear()
        self.update_spline()

    def interpolate_pose_and_fov(self, normalized_t: float) -> Optional[Tuple[tf.SE3, float]]:
        if len(self._keyframes) < 2:
            return None
        # TODO: this doesn't need to be constantly re-instantiated.
        self._fov_spline = splines.KochanekBartels(
            [
                keyframe[0].override_fov_value if keyframe[0].override_fov_enabled else self.default_fov
                for keyframe in self._keyframes.values()
            ],
            tcb=(self.smoothness, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        assert self._orientation_spline is not None
        assert self._position_spline is not None
        assert self._fov_spline is not None
        max_t = len(self._keyframes) if self.loop else len(self._keyframes) - 1
        t = max_t * normalized_t
        quat = self._orientation_spline.evaluate(t)
        assert isinstance(quat, splines.quaternion.UnitQuaternion)
        return (
            tf.SE3.from_rotation_and_translation(
                tf.SO3(onp.array([quat.scalar, *quat.vector])),
                self._position_spline.evaluate(t),
            ),
            float(self._fov_spline.evaluate(t)),
        )

    def update_spline(self) -> None:
        keyframes = list(self._keyframes.values())
        if len(keyframes) <= 1:
            if self._spline is not None:
                self._spline.remove()
                self._spline = None
            return

        # Update internal splines.
        self._orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(onp.roll(keyframe[0].wxyz, shift=-1))
                for keyframe in keyframes
            ],
            tcb=(self.smoothness, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )
        self._position_spline = splines.KochanekBartels(
            [keyframe[0].position for keyframe in keyframes],
            tcb=(self.smoothness, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        # Update visualized spline.
        num_keyframes = len(keyframes) + 1 if self.loop else len(keyframes)
        points_array = onp.array(
            [self._position_spline.evaluate(t) for t in onp.linspace(0, num_keyframes - 1, num_keyframes * 100)]
        )
        colors_array = onp.array([colorsys.hls_to_rgb(h, 0.5, 1.0) for h in onp.linspace(0.0, 1.0, len(points_array))])
        self._spline = self._server.add_point_cloud(
            "/render_camera_spline",
            points=points_array,
            colors=colors_array,
            point_size=0.035,
        )


def populate_render_tab(
    server: viser.ViserServer, config_path: Path, datapath: Path, control_panel: ControlPanel
) -> None:
    from nerfstudio.viewer_beta.viewer import VISER_NERFSTUDIO_SCALE_RATIO

    fov_degrees = server.add_gui_slider(
        "FOV",
        initial_value=90.0,
        min=0.1,
        max=175.0,
        step=0.01,
        hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
    )

    @fov_degrees.on_update
    def _(_) -> None:
        fov_radians = fov_degrees.value / 180.0 * onp.pi
        for client in server.get_clients().values():
            client.camera.fov = fov_radians
        camera_path.default_fov = fov_radians

        # Updating the aspect ratio will also re-render the camera frustums.
        # Could rethink this.
        camera_path.update_aspect(resolution.value[0] / resolution.value[1])

    resolution = server.add_gui_vector2(
        "Resolution",
        initial_value=(1920, 1080),
        min=(50, 50),
        max=(10_000, 10_000),
        step=1,
        hint="Tension parameter for adjusting smoothness of spline interpolation.",
    )

    @resolution.on_update
    def _(_) -> None:
        """Update the aspect ratio for all cameras when the resolution changes."""
        camera_path.update_aspect(resolution.value[0] / resolution.value[1])

    add_button = server.add_gui_button(
        "Add keyframe",
        icon=viser.Icon.PLUS,
        hint="Add a new keyframe at the current pose.",
    )

    @add_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client_id is not None
        camera = server.get_clients()[event.client_id].camera

        # Add this camera to the path.
        camera_path.add_camera(
            Keyframe.from_camera(camera, aspect=resolution.value[0] / resolution.value[1]),
        )
        camera_path.update_spline()

    reset_up_button = server.add_gui_button(
        "Reset up direction",
        icon=viser.Icon.ARROW_AUTOFIT_UP,
        hint="Reset the orbit up direction.",
    )

    @reset_up_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        event.client.camera.up_direction = tf.SO3(event.client.camera.wxyz) @ onp.array([0.0, -1.0, 0.0])

    clear_keyframes_button = server.add_gui_button(
        "Clear keyframes",
        icon=viser.Icon.TRASH,
        hint="Remove all keyframes from the render path.",
    )

    @clear_keyframes_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client_id is not None
        client = server.get_clients()[event.client_id]
        with client.add_gui_modal("Confirm") as modal:
            client.add_gui_markdown("Clear all keyframes?")
            confirm_button = client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
            exit_button = client.add_gui_button("Cancel")

            @confirm_button.on_click
            def _(_) -> None:
                camera_path.reset()
                modal.close()

                # Clear move handles.
                if len(transform_controls) > 0:
                    for t in transform_controls:
                        t.remove()
                    transform_controls.clear()
                    return

            @exit_button.on_click
            def _(_) -> None:
                modal.close()

    loop = server.add_gui_checkbox("Loop", False)

    @loop.on_update
    def _(_) -> None:
        camera_path.loop = loop.value
        camera_path.update_spline()

    smoothness = server.add_gui_slider(
        "Spline Tension",
        min=0.0,
        max=1.0,
        initial_value=0.0,
        step=0.01,
        hint="Tension parameter for adjusting smoothness of spline interpolation.",
    )

    @smoothness.on_update
    def _(_) -> None:
        camera_path.smoothness = smoothness.value
        camera_path.update_spline()

    move_checkbox = server.add_gui_checkbox(
        "Move keyframes",
        initial_value=False,
        hint="Toggle move handles for keyframes in the scene.",
    )

    @move_checkbox.on_update
    def _(event: viser.GuiEvent) -> None:
        # Clear move handles when toggled off.
        if move_checkbox.value is False:
            for t in transform_controls:
                t.remove()
            transform_controls.clear()
            return

        def _make_transform_controls_callback(
            keyframe: Tuple[Keyframe, viser.SceneNodeHandle],
            controls: viser.TransformControlsHandle,
        ) -> None:
            @controls.on_update
            def _(_) -> None:
                keyframe[0].wxyz = controls.wxyz
                keyframe[0].position = controls.position

                keyframe[1].wxyz = controls.wxyz
                keyframe[1].position = controls.position

                camera_path.update_spline()

        # Show move handles.
        assert event.client is not None
        for keyframe_index, keyframe in camera_path._keyframes.items():
            controls = event.client.add_transform_controls(
                f"/keyframe_move/{keyframe_index}",
                scale=0.4,
                wxyz=keyframe[0].wxyz,
                position=keyframe[0].position,
            )
            transform_controls.append(controls)
            _make_transform_controls_callback(keyframe, controls)

    playback_folder = server.add_gui_folder("Playback")
    with playback_folder:
        duration_number = server.add_gui_number("Duration (sec)", min=0.0, max=1e8, step=0.0001, initial_value=4.0)
        framerate_slider = server.add_gui_slider("FPS", min=1.0, max=240.0, step=1e-8, initial_value=30.0)
        framerate_buttons = server.add_gui_button_group("", ("24", "30", "60"))

        @framerate_buttons.on_click
        def _(_) -> None:
            framerate_slider.value = float(framerate_buttons.value)

        play_button = server.add_gui_button("Play", icon=viser.Icon.PLAYER_PLAY)
        pause_button = server.add_gui_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
        attach_viewport_checkbox = server.add_gui_checkbox("Attach viewport", initial_value=False)
        show_checkbox = server.add_gui_checkbox(
            "Show keyframes",
            initial_value=True,
            hint="Show keyframes in the scene.",
        )

    @show_checkbox.on_update
    def _(_: viser.GuiEvent) -> None:
        camera_path.set_keyframes_visible(show_checkbox.value)

    def add_preview_frame_slider() -> Optional[viser.GuiInputHandle[int]]:
        """Helper for creating the current frame # slider. This is removed and
        re-added anytime the `max` value changes."""
        max_frame_index = int(framerate_slider.value * duration_number.value) - 1

        if max_frame_index <= 0:
            return None
        with playback_folder:
            preview_frame_slider = server.add_gui_slider(
                "Preview frame",
                min=0,
                max=max_frame_index,
                step=1,
                initial_value=0,
                # Place right after the pause button.
                order=pause_button.order + 0.01,
            )

        @preview_frame_slider.on_update
        def _(_) -> None:
            max_frame_index = int(framerate_slider.value * duration_number.value) - 1
            maybe_pose_and_fov = camera_path.interpolate_pose_and_fov(
                preview_frame_slider.value / max_frame_index if max_frame_index > 0 else 0
            )
            if maybe_pose_and_fov is None:
                return
            pose, fov = maybe_pose_and_fov
            server.add_camera_frustum(
                "/preview_camera",
                fov=fov,
                aspect=resolution.value[0] / resolution.value[1],
                scale=0.35,
                wxyz=pose.rotation().wxyz,
                position=pose.translation(),
                color=(10, 200, 30),
                # Hack: hide green frustum if the viewport is attached.
                # This is a waste of bandwidth, but will ensure that any old
                # frustums are removed/aren't rendered.
                #
                # Easy to fix with a global variable.
                visible=not attach_viewport_checkbox.value,
            )
            if attach_viewport_checkbox.value:
                for client in server.get_clients().values():
                    client.camera.wxyz = pose.rotation().wxyz
                    client.camera.position = pose.translation()
                    client.camera.fov = fov

        return preview_frame_slider

    @attach_viewport_checkbox.on_update
    def _(_) -> None:
        if not attach_viewport_checkbox.value:
            for client in server.get_clients().values():
                client.camera.fov = fov_degrees.value

    preview_frame_slider = add_preview_frame_slider()

    @duration_number.on_update
    @framerate_slider.on_update
    def _(_) -> None:
        nonlocal preview_frame_slider
        old = preview_frame_slider
        assert old is not None

        preview_frame_slider = add_preview_frame_slider()
        if preview_frame_slider is not None:
            old.remove()
        else:
            preview_frame_slider = old

    # Play the camera trajectory when the play button is pressed.
    @play_button.on_click
    def _(_) -> None:
        play_button.visible = False
        pause_button.visible = True

        def play() -> None:
            while not play_button.visible:
                max_frame = int(framerate_slider.value * duration_number.value)
                if max_frame > 0:
                    assert preview_frame_slider is not None
                    preview_frame_slider.value = (preview_frame_slider.value + 1) % max_frame
                time.sleep(1.0 / framerate_slider.value)

        threading.Thread(target=play).start()

    # Play the camera trajectory when the play button is pressed.
    @pause_button.on_click
    def _(_) -> None:
        play_button.visible = True
        pause_button.visible = False

    # set the initial value to the current date-time string
    now = datetime.datetime.now()
    render_name_text = server.add_gui_text(
        "Render Name", initial_value=now.strftime("%Y-%m-%d-%H-%M-%S"), hint="Name of the render"
    )
    render_button = server.add_gui_button(
        "Generate Command",
        color="green",
        icon=viser.Icon.FILE_EXPORT,
        hint="Generate the ns-render command for rendering the camera path.",
    )

    @render_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        num_frames = int(framerate_slider.value * duration_number.value)
        json_data = {}
        # json data has the properties:
        # keyframes: list of keyframes with
        # matrix : flattened 4x4 matrix
        # fov: float in degrees
        # aspect: float
        # camera_type: string of camera type
        # render_height: int
        # render_width: int
        # fps: int
        # seconds: float
        # is_cycle: bool
        # smoothness_value: float
        # camera_path: list of frames with properties
        # camera_to_world: flattened 4x4 matrix
        # fov: float in degrees
        # aspect: float
        # first populate the keyframes:
        keyframes = []
        for keyframe, dummy in camera_path._keyframes.values():
            pose = tf.SE3.from_rotation_and_translation(
                tf.SO3(keyframe.wxyz) @ tf.SO3.from_x_radians(onp.pi),
                keyframe.position / VISER_NERFSTUDIO_SCALE_RATIO,
            )
            keyframes.append(
                {
                    "matrix": pose.as_matrix().flatten().tolist(),
                    "fov": onp.rad2deg(keyframe.override_fov_value)
                    if keyframe.override_fov_enabled
                    else fov_degrees.value,
                    "aspect": keyframe.aspect,
                }
            )
        json_data["keyframes"] = keyframes
        json_data["camera_type"] = "perspective"
        json_data["render_height"] = resolution.value[1]
        json_data["render_width"] = resolution.value[0]
        json_data["fps"] = framerate_slider.value
        json_data["seconds"] = duration_number.value
        json_data["is_cycle"] = loop.value
        json_data["smoothness_value"] = smoothness.value
        # now populate the camera path:
        camera_path_list = []
        for i in range(num_frames):
            maybe_pose_and_fov = camera_path.interpolate_pose_and_fov(i / num_frames)
            if maybe_pose_and_fov is None:
                return
            pose, fov = maybe_pose_and_fov
            # rotate the axis of the camera 180 about x axis
            pose = tf.SE3.from_rotation_and_translation(
                pose.rotation() @ tf.SO3.from_x_radians(onp.pi),
                pose.translation() / VISER_NERFSTUDIO_SCALE_RATIO,
            )
            camera_path_list.append(
                {
                    "camera_to_world": pose.as_matrix().flatten().tolist(),
                    "fov": onp.rad2deg(fov),
                    "aspect": resolution.value[0] / resolution.value[1],
                }
            )
        json_data["camera_path"] = camera_path_list
        # finally add crop data if crop is enabled
        if control_panel.crop_viewport:
            obb = control_panel.crop_obb
            rpy = tf.SO3.from_matrix(obb.R.numpy()).as_rpy_radians()
            color = control_panel.background_color
            json_data["crop"] = {
                "crop_center": obb.T.tolist(),
                "crop_scale": obb.S.tolist(),
                "crop_rot": [rpy.roll, rpy.pitch, rpy.yaw],
                "crop_bg_color": {"r": color[0], "g": color[1], "b": color[2]},
            }

        # now write the json file
        json_outfile = datapath / "camera_paths" / f"{render_name_text.value}.json"
        with open(json_outfile.absolute(), "w") as outfile:
            json.dump(json_data, outfile)
        # now show the command
        with event.client.add_gui_modal("Render Command") as modal:
            dataname = datapath.name
            command = " ".join(
                [
                    "ns-render camera-path",
                    f"--load-config {config_path}",
                    f"--camera-path-filename {json_outfile.absolute()}",
                    f"--output-path renders/{dataname}/{render_name_text.value}.mp4",
                ]
            )
            event.client.add_gui_markdown(
                "\n".join(
                    [
                        "To render the trajectory, run the following from the command line:",
                        "",
                        "```",
                        command,
                        "```",
                    ]
                )
            )
            close_button = event.client.add_gui_button("Close")

            @close_button.on_click
            def _(_) -> None:
                modal.close()

    camera_path = CameraPath(server)
    camera_path.default_fov = fov_degrees.value / 180.0 * onp.pi

    transform_controls: List[viser.SceneNodeHandle] = []
