# type: ignore

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

"""
nerfstudio_blender.py
"""

bl_info = {
    "name": "Nerfstudio Add-On",
    "description": "Create a Nerfstudio JSON camera path from the Blender camera path \
    or import a Nerfstudio camera path as a Blender camera to composite Blender renders \
    over a NeRF background render for VFX",
    "author": "Cyrus Vachha",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "category": "Nerfstudio",
}


import json  # noqa: E402
from math import atan, degrees, radians, tan  # noqa: E402

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402


class CreateJSONCameraPath(bpy.types.Operator):
    """Create a JSON camera path from the Blender camera animation."""

    bl_idname = "opr.create_json_camera_path"
    bl_label = "Nerfstudio Camera Path Generator"

    cam_obj = None  # the render camera is the active camera
    nerf_bg_mesh = None  # the background NeRF as a mesh

    fov_list = []  # list of FOV at each frame
    transformed_camera_path_mat = []  # final transformed world matrix of the camera at each frame

    complete_json_obj = {}  # full Nerfstudio input json object

    file_path_json = ""  # file path input

    def get_camera_coordinates(self):
        """Create a list of transformed Blender camera coordinates and converted FOV."""

        org_camera_path_mat = []  # list of world matrix of the active camera at each frame
        nerf_mesh_mat_list = []  # list of world matrix of the NeRF mesh at each frame

        curr_frame = bpy.context.scene.frame_start

        while curr_frame <= bpy.context.scene.frame_end:
            bpy.context.scene.frame_set(curr_frame)
            org_camera_path_mat += [self.cam_obj.matrix_world.copy()]

            if bpy.context.scene.render.resolution_y >= bpy.context.scene.render.resolution_x:
                # portrait orientation

                if self.cam_obj.data.sensor_fit == "HORIZONTAL":
                    # convert horizontal fov to vertical fov with aspect ratio
                    cam_aspect_ratio = bpy.context.scene.render.resolution_y / bpy.context.scene.render.resolution_x
                    nerfstudio_fov = 2 * atan(tan(self.cam_obj.data.angle / 2.0) * cam_aspect_ratio)
                else:
                    # sensor fit is either vertical or auto
                    nerfstudio_fov = self.cam_obj.data.angle

            else:
                # landscape orientation

                if self.cam_obj.data.sensor_fit == "VERTICAL":
                    nerfstudio_fov = self.cam_obj.data.angle
                else:
                    # sensor fit is either horizontal or auto
                    # convert horizontal fov to vertical fov with aspect ratio
                    cam_aspect_ratio = bpy.context.scene.render.resolution_y / bpy.context.scene.render.resolution_x
                    nerfstudio_fov = 2 * atan(tan(self.cam_obj.data.angle / 2.0) * cam_aspect_ratio)

            self.fov_list += [degrees(nerfstudio_fov)]
            curr_frame += bpy.context.scene.frame_step
            nerf_mesh_mat_list += [self.nerf_bg_mesh.matrix_world.copy()]

            # case when step size is 0 there is only one frame
            if bpy.context.scene.frame_step == 0:
                break

        # transform the camera world matrix based on the NeRF mesh transformation
        for i, org_cam_path_mat_val in enumerate(org_camera_path_mat):
            self.transformed_camera_path_mat += [nerf_mesh_mat_list[i].inverted() @ org_cam_path_mat_val]

    def get_list_from_matrix_path(self, input_mat):
        """Flatten matrix to list for camera path."""
        full_arr = list(input_mat.row[0]) + list(input_mat.row[1]) + list(input_mat.row[2]) + list(input_mat.row[3])
        return full_arr

    def get_list_from_matrix_keyframe(self, input_mat):
        """Flatten matrix to list for keyframes."""
        full_arr = list(input_mat.col[0]) + list(input_mat.col[1]) + list(input_mat.col[2]) + list(input_mat.col[3])
        return full_arr

    def construct_json_obj(self):
        """Get fields for JSON camera path."""
        cam_type = self.cam_obj.data.type
        if cam_type == "PERSP":
            cam_type = "perspective"
        elif cam_type == "PANO" and self.cam_obj.data.cycles.panorama_type == "EQUIRECTANGULAR":
            cam_type = "equirectangular"
        else:
            self.report(
                {"WARNING"}, "Nerfstudio Add-on Warning: Only perspective and equirectangular cameras are supported"
            )
            cam_type = "perspective"

        render_height = int(
            bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage * 0.01)
        )
        render_width = int(
            bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage * 0.01)
        )
        render_fps = bpy.context.scene.render.fps

        # case when step size is 0 there is only one frame
        if bpy.context.scene.frame_step == 0:
            render_seconds = 1 / render_fps
        else:
            render_seconds = (
                ((bpy.context.scene.frame_end - bpy.context.scene.frame_start) // (bpy.context.scene.frame_step) + 1)
            ) / render_fps

        smoothness_value = 0
        is_cycle = False

        # construct camera path
        final_camera_path = []

        for i, transformed_camera_path_mat_val in enumerate(self.transformed_camera_path_mat):
            camera_path_elem = {
                "camera_to_world": self.get_list_from_matrix_path(transformed_camera_path_mat_val),
                "fov": self.fov_list[i],
                "aspect": 1,
            }
            final_camera_path += [camera_path_elem]

        # construct keyframes
        keyframe_list = []

        for i, transformed_camera_path_mat_val in enumerate(self.transformed_camera_path_mat):
            curr_properties = (
                '[["FOV",'
                + str(self.fov_list[i])
                + '],["NAME","Camera '
                + str(i)
                + '"],["TIME",'
                + str(i / render_fps)
                + "]]"
            )

            keyframe_elem = {
                "matrix": str(self.get_list_from_matrix_keyframe(self.transformed_camera_path_mat[i])),
                "fov": self.fov_list[i],
                "aspect": 1,
                "properties": curr_properties,
            }
            keyframe_list += [keyframe_elem]

        overall_json = {
            "keyframes": keyframe_list,
            "camera_type": cam_type,
            "render_height": render_height,
            "render_width": render_width,
            "camera_path": final_camera_path,
            "fps": render_fps,
            "seconds": render_seconds,
            "smoothness_value": smoothness_value,
            "is_cycle": is_cycle,
        }

        self.complete_json_obj = json.dumps(overall_json, indent=2)

    def write_json_to_file(self):
        """Write the JSON object to a new file."""

        full_abs_file_path = bpy.path.abspath(self.file_path_json + "camera_path_blender.json")
        with open(full_abs_file_path, "w", encoding="utf8") as output_json_camera_path:
            output_json_camera_path.truncate(0)
            output_json_camera_path.write(self.complete_json_obj)

        self.complete_json_obj = {}
        print("\nFinished creating camera path json file at " + full_abs_file_path + "\n")

    def execute(self, context):
        """Execute the camera path creation process."""
        # get user specified values from UI

        self.cam_obj = bpy.context.scene.camera
        self.nerf_bg_mesh = context.scene.NeRF
        self.file_path_json = context.scene.JSONInputFilePath

        # check input
        if self.nerf_bg_mesh is None:
            self.report(
                {"ERROR"}, "Nerfstudio add-on Error! - Please input NeRF representation (as mesh or point cloud)"
            )
            return {"FINISHED"}

        if self.file_path_json == "":
            self.report({"ERROR"}, "Nerfstudio add-on Error! - Please input a file path for the output JSON")
            return {"FINISHED"}

        # reset lists before running
        self.fov_list = []
        self.transformed_camera_path_mat = []
        self.complete_json_obj = {}

        # create the path
        self.get_camera_coordinates()
        self.construct_json_obj()
        self.write_json_to_file()

        return {"FINISHED"}


class ReadJSONinputCameraPath(bpy.types.Operator):
    """Create a camera with an animation path based on an input Nerfstudio JSON."""

    bl_idname = "opr.read_json_camera_path"
    bl_label = "Blender Camera Generator from JSON"

    # cam_obj = None # the render camera is the active camera
    nerf_bg_mesh = None  # the background NeRF as a mesh

    fov_list = []  # list of FOV at each frame
    transformed_camera_path_mat = []  # final transformed world matrix of the camera at each frame
    input_json = None

    def read_camera_coordinates(self):
        """Read the camera coordinates (world matrix and fov) from the json camera path."""

        json_cam_path = self.input_json["camera_path"]
        self.fov_list = []
        self.transformed_camera_path_mat = []

        keyframe_counter = 0
        for cam_keyframe in json_cam_path:
            cam_to_world = cam_keyframe["camera_to_world"]

            # convert cam_to_world to 4x4 matrix
            orig_cam_mat = Matrix([cam_to_world[0:4], cam_to_world[4:8], cam_to_world[8:12], cam_to_world[12:]])

            # matrix transformation based on the nerf mesh to find relative camera positions
            self.transformed_camera_path_mat += [self.nerf_bg_mesh.matrix_world.copy() @ orig_cam_mat]

            # record fov
            self.fov_list += [cam_keyframe["fov"]]

            keyframe_counter += 1

    def generate_camera(self):
        """Create a new camera with the animation (position and fov) and the corresponding type."""

        json_cam_path = self.input_json["camera_path"]

        camera_data = bpy.data.cameras.new(name="NerfstudioCamera")
        camera_data = bpy.data.cameras.new(name="NerfstudioCamera")
        nerfstudio_camera_object = bpy.data.objects.new("NerfstudioCamera", camera_data)
        bpy.context.scene.collection.objects.link(nerfstudio_camera_object)

        curr_frame = 0
        while curr_frame < len(json_cam_path):
            actual_frame = curr_frame + 1
            # animate camera transform
            nerfstudio_camera_object.matrix_world = self.transformed_camera_path_mat[curr_frame]
            nerfstudio_camera_object.keyframe_insert("location", frame=actual_frame)
            nerfstudio_camera_object.keyframe_insert("rotation_euler", frame=actual_frame)

            # set scale to 1,1,1 (scale is not keyframed)
            nerfstudio_camera_object.scale = (1, 1, 1)

            # animate fov
            nerfstudio_camera_object.data.sensor_fit = "VERTICAL"
            nerfstudio_camera_object.data.lens_unit = "FOV"
            nerfstudio_camera_object.data.angle = radians(self.fov_list[curr_frame])

            # set keyframe for focal length
            nerfstudio_camera_object.data.keyframe_insert(data_path="lens", frame=actual_frame)

            curr_frame += 1

        # set camera attributes
        input_cam_type = self.input_json["camera_type"]
        if input_cam_type == "perspective":
            nerfstudio_camera_object.data.type = "PERSP"
        if input_cam_type == "equirectangular":
            nerfstudio_camera_object.data.type = "PANO"
            bpy.context.scene.render.engine = "CYCLES"
            nerfstudio_camera_object.data.cycles.panorama_type = "EQUIRECTANGULAR"
        if input_cam_type == "fisheye":
            nerfstudio_camera_object.data.type = "PERSP"
            self.report({"WARNING"}, "Nerfstudio Add-on Warning: Fisheye cameras are not supported")

    def execute(self, context):
        """Execute Blender camera creation process."""

        # initializat variables
        self.nerf_bg_mesh = context.scene.NeRF
        file_path_ns_json = context.scene.NS_input_jsonFilePath  # input file path for the input json file

        # check input
        if self.nerf_bg_mesh is None:
            self.report(
                {"ERROR"}, "Nerfstudio add-on Error! - Please input NeRF representation (as mesh or point cloud)"
            )
            return {"FINISHED"}

        if file_path_ns_json == "":
            self.report({"ERROR"}, "Nerfstudio add-on Error! - Please input a Nerfstudio JSON camera path")
            return {"FINISHED"}

        # open the json file
        full_abs_file_path = bpy.path.abspath(file_path_ns_json)
        with open(full_abs_file_path, encoding="utf8") as json_ns_file:
            self.input_json = json.load(json_ns_file)

        # call methods to read cam path and create camera
        self.read_camera_coordinates()
        self.generate_camera()

        return {"FINISHED"}


# --- Blender UI Panel --- #


class NerfstudioMainPanel(bpy.types.Panel):
    """Blender UI main panel for the add-on."""

    bl_idname = "NERFSTUDIO_PT_NerfstudioMainPanel"
    bl_label = "Nerfstudio Add-on"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        """Main panel UI components."""
        # NeRF representation object input box
        self.layout.label(text="NeRF Representation (mesh or point cloud)")
        self.layout.prop_search(context.scene, "NeRF", context.scene, "objects")
        _ = self.layout.column()


class NerfstudioBgPanel(bpy.types.Panel):
    """Blender UI sub-panel for the camera path creation."""

    bl_idname = "NERFSTUDIO_PT_NerfstudioBgPanel"
    bl_label = "Nerfstudio Path Generator"
    bl_parent_id = "NERFSTUDIO_PT_NerfstudioMainPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        """Sub-panel UI components."""

        self.layout.label(text="Camera path for Nerfstudio")

        col = self.layout.column()
        for prop_name, _ in INPUT_PROPERTIES:
            row = col.row()
            row.prop(context.scene, prop_name)

        col.operator("opr.create_json_camera_path", text="Generate JSON File")


class NerfstudioInputPanel(bpy.types.Panel):
    """Blender UI sub-panel for the Blender camera creation."""

    bl_idname = "NERFSTUDIO_PT_NerfstudioInputPanel"
    bl_label = "Nerfstudio Camera Generator"
    bl_parent_id = "NERFSTUDIO_PT_NerfstudioMainPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):
        """Sub-panel UI components."""

        col = self.layout.column()
        self.layout.label(text="Create Blender Camera From Nerfstudio JSON")
        col = self.layout.column()

        for prop_name, _ in INPUT_PROPERTIES_NS_CAMERA:
            row = col.row()
            row.prop(context.scene, prop_name)

        col.operator("opr.read_json_camera_path", text="Create Camera from JSON")


CLASSES = [
    NerfstudioMainPanel,
    NerfstudioBgPanel,
    NerfstudioInputPanel,
    CreateJSONCameraPath,
    ReadJSONinputCameraPath,
]

INPUT_PROPERTIES = [
    (
        "JSONInputFilePath",
        bpy.props.StringProperty(name="JSON File Path", default="//", description="Path for JSON", subtype="DIR_PATH"),
    )
]

INPUT_PROPERTIES_NS_CAMERA = [
    (
        "NS_input_jsonFilePath",
        bpy.props.StringProperty(
            name="JSON Nerfstudio File",
            default="",
            description="Path for JSON from Nerfstudio editor",
            subtype="FILE_PATH",
        ),
    )
]

OBJ_PROPERTIES = ["NeRF", "RenderCamera"]


def register():
    """Register classes for UI panel."""

    for prop_name, prop_value in INPUT_PROPERTIES:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for prop_name, prop_value in INPUT_PROPERTIES_NS_CAMERA:
        setattr(bpy.types.Scene, prop_name, prop_value)

    bpy.types.Scene.NeRF = bpy.props.PointerProperty(type=bpy.types.Object)

    for curr_class in CLASSES:
        bpy.utils.register_class(curr_class)


def unregister():
    """Unregister classes for UI panel."""

    for prop_name, _ in INPUT_PROPERTIES:
        delattr(bpy.types.Scene, prop_name)

    for prop_name, _ in INPUT_PROPERTIES_NS_CAMERA:
        delattr(bpy.types.Scene, prop_name)

    del bpy.types.Scene.NeRF

    for curr_class in CLASSES:
        bpy.utils.unregister_class(curr_class)


if __name__ == "__main__":
    register()
