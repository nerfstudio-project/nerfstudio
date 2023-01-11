bl_info = {
    "name": "Nerfstudio Add-On",
    "description": "Create a Nerfstudio JSON camera path from the Blender camera path or import a Nerfstudio camera path as a Blender camera to composite Blender renders over a NeRF background render",
    "author": "Cyrus Vachha",
    "version": (1, 0),
    "blender": (3, 0, 1),
    "category": "Nerfstudio",
}

import json
from math import atan, degrees, radians, tan

import bpy
from mathutils import Matrix


# create a JSON camera path from the Blender camera animation
class CreateJSONCameraPath(bpy.types.Operator):
    bl_idname = "opr.create_json_camera_path"
    bl_label = "Nerfstudio Camera Path Generator"

    cam_obj = None  # the render camera is the active camera
    nerf_bg_mesh = None  # the background NeRF as a mesh

    fov_list = []  # list of FOV at each frame
    transformed_cameraPath_mat = []  # final transformed world matrix of the camera at each frame

    complete_json_obj = {}  # full Nerfstudio input json object

    file_path_json = ""  # file path input

    def getCameraCoodinates(self):
        org_cameraPath_mat = []  # list of world matrix of the active camera at each frame
        nerf_mesh_mat_list = []  # list of world matrix of the NeRF mesh at each frame

        curr_frame = bpy.context.scene.frame_start

        while curr_frame <= bpy.context.scene.frame_end:
            bpy.context.scene.frame_set(curr_frame)
            org_cameraPath_mat += [self.cam_obj.matrix_world.copy()]

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

        # transform the camera world matrix based on the NeRF mesh transformation
        for i in range(len(org_cameraPath_mat)):
            self.transformed_cameraPath_mat += [nerf_mesh_mat_list[i].inverted() @ org_cameraPath_mat[i]]

    def getListFromMatrixPath(self, inputMat):
        # flatten matrix to list for camera path
        fullArr = list(inputMat.row[0]) + list(inputMat.row[1]) + list(inputMat.row[2]) + list(inputMat.row[3])
        return fullArr

    def getListFromMatrixKeyframe(self, inputMat):
        # flatten matrix to list for keyframes
        fullArr = list(inputMat.col[0]) + list(inputMat.col[1]) + list(inputMat.col[2]) + list(inputMat.col[3])
        return fullArr

    def constructJsonObj(self):
        # get camera parameters
        cam_type = self.cam_obj.data.type
        if cam_type == "PERSP":
            cam_type = "perspective"
        elif cam_type == "PANO" and self.cam_obj.data.cycles.panorama_type == "EQUIRECTANGULAR":
            cam_type = "equirectangular"
        else:
            cam_type = "perspective"

        render_height = int(
            bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage * 0.01)
        )
        render_width = int(
            bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage * 0.01)
        )
        render_fps = 1
        render_seconds = len(self.transformed_cameraPath_mat)
        smoothness_value = 0
        is_cycle = False

        # construct camera path
        final_camera_path = []

        for i in range(len(self.transformed_cameraPath_mat)):
            camera_path_elem = {
                "camera_to_world": self.getListFromMatrixPath(self.transformed_cameraPath_mat[i]),
                "fov": self.fov_list[i],
                "aspect": 1,
            }
            final_camera_path += [camera_path_elem]

        # construct keyframes
        keyframe_list = []
        for i in range(len(self.transformed_cameraPath_mat)):
            curr_properties = (
                '[["FOV",' + str(self.fov_list[i]) + '],["NAME","Camera ' + str(i) + '"],["TIME",' + str(i) + "]]"
            )

            keyframe_elem = {
                "matrix": str(self.getListFromMatrixKeyframe(self.transformed_cameraPath_mat[i])),
                "fov": self.fov_list[i],
                "aspect": 1,
                "properties": curr_properties,
            }
            keyframe_list += [keyframe_elem]

        overallJSON = {
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

        self.complete_json_obj = json.dumps(overallJSON, indent=2)

    def writeJSONToFile(self):
        full_abs_file_path = bpy.path.abspath(self.file_path_json + "camera_path_blender.json")
        with open(full_abs_file_path, "w") as outputJSONCameraPath:
            outputJSONCameraPath.truncate(0)
            outputJSONCameraPath.write(self.complete_json_obj)

        self.complete_json_obj = {}
        print("\nFinished creating camera path json file at " + full_abs_file_path + "\n")

    def execute(self, context):
        # get user specified values from UI
        params = (context.scene.NeRF, context.scene.JSONInputFilePath)

        self.cam_obj = bpy.context.scene.camera
        self.nerf_bg_mesh = context.scene.NeRF
        self.file_path_json = context.scene.JSONInputFilePath

        # check input
        if self.nerf_bg_mesh is None:
            print("Nerfstudio add-on Error! - Please input NeRF representation (as mesh or point cloud)")
            return {"FINISHED"}

        # reset lists before running
        self.fov_list = []
        self.transformed_cameraPath_mat = []
        self.complete_json_obj = {}

        # create the path
        self.getCameraCoodinates()
        self.constructJsonObj()
        self.writeJSONToFile()

        return {"FINISHED"}


# create a camera with an animation path based on an input Nerfstudio JSON
class ReadJSONinputCameraPath(bpy.types.Operator):
    bl_idname = "opr.read_json_camera_path"
    bl_label = "Blender Camera Generator from JSON"

    # cam_obj = None # the render camera is the active camera
    nerf_bg_mesh = None  # the background NeRF as a mesh

    fov_list = []  # list of FOV at each frame
    transformed_cameraPath_mat = []  # final transformed world matrix of the camera at each frame
    inputJson = None

    def readCameraCoodinates(self):
        # read the camera coordinates (world matrix and fov) from the json camera path

        json_cam_path = self.inputJson["camera_path"]
        nerf_mesh_mat_list = []
        self.fov_list = []
        self.transformed_cameraPath_mat = []

        keyframe_counter = 0
        for cam_keyframe in json_cam_path:
            cam_to_world = cam_keyframe["camera_to_world"]

            # convert cam_to_world to 4x4 matrix
            orig_cam_mat = Matrix([cam_to_world[0:4], cam_to_world[4:8], cam_to_world[8:12], cam_to_world[12:]])

            # matrix transformation based on the nerf mesh to find relative camera positions
            self.transformed_cameraPath_mat += [self.nerf_bg_mesh.matrix_world.copy() @ orig_cam_mat]

            # record fov
            self.fov_list += [cam_keyframe["fov"]]

            keyframe_counter += 1

    def generateCamera(self):
        # create a new camera with the animation (position and fov) and the corresponding type

        json_cam_path = self.inputJson["camera_path"]

        newCamera = camera_data = bpy.data.cameras.new(name="NerfstudioCamera")
        camera_data = bpy.data.cameras.new(name="NerfstudioCamera")
        nerfstudio_camera_object = bpy.data.objects.new("NerfstudioCamera", camera_data)
        bpy.context.scene.collection.objects.link(nerfstudio_camera_object)

        curr_frame = 0
        while curr_frame < len(json_cam_path):
            actual_frame = curr_frame + 1
            # animate camera transform
            nerfstudio_camera_object.matrix_world = self.transformed_cameraPath_mat[curr_frame]
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
        inputCamType = self.inputJson["camera_type"]
        if inputCamType == "perspective":
            nerfstudio_camera_object.data.type = "PERSP"
        if inputCamType == "equirectangular":
            nerfstudio_camera_object.data.type = "PANO"
            bpy.context.scene.render.engine = "CYCLES"
            nerfstudio_camera_object.data.cycles.panorama_type = "EQUIRECTANGULAR"
        if inputCamType == "fisheye":
            nerfstudio_camera_object.data.type = "PERSP"
            print("Nerfstudio Add-on Warning: Fisheye cameras are not supported")

    def execute(self, context):

        # initializat variables
        params = (context.scene.NeRF, context.scene.NS_InputJSONFilePath)

        self.nerf_bg_mesh = context.scene.NeRF
        file_path_ns_json = context.scene.NS_InputJSONFilePath  # input file path for the input json file

        # check input
        if self.nerf_bg_mesh is None:
            print("Nerfstudio add-on Error! - Please input NeRF representation (as mesh or point cloud)")
            return {"FINISHED"}

        if file_path_ns_json == "":
            print("Nerfstudio add-on Error! - Please input a Nerfstudio JSON camera path")
            return {"FINISHED"}

        # open the json file
        full_abs_file_path = bpy.path.abspath(file_path_ns_json)
        json_ns_file = open(full_abs_file_path)
        self.inputJson = json.load(json_ns_file)

        # call methods to read cam path and create camera
        self.readCameraCoodinates()
        self.generateCamera()

        return {"FINISHED"}


# --- Blender UI Panel --- #


class NerfstudioMainPanel(bpy.types.Panel):

    bl_idname = "NERFSTUDIO_PT_NerfstudioMainPanel"
    bl_label = "Nerfstudio Add-on"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):

        # NeRF representation object input box
        self.layout.label(text="NeRF Representation (mesh or point cloud)")
        self.layout.prop_search(context.scene, "NeRF", context.scene, "objects")
        col = self.layout.column()


class NerfstudioBgPanel(bpy.types.Panel):

    bl_idname = "NERFSTUDIO_PT_NerfstudioBgPanel"
    bl_label = "Nerfstudio Path Generator"
    bl_parent_id = "NERFSTUDIO_PT_NerfstudioMainPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):

        self.layout.label(text="Camera path for Nerfstudio")

        col = self.layout.column()
        for (prop_name, _) in INPUT_PROPERTIES:
            row = col.row()
            row.prop(context.scene, prop_name)

        col.operator("opr.create_json_camera_path", text="Generate JSON File")


class NerfstudioInputPanel(bpy.types.Panel):

    bl_idname = "NERFSTUDIO_PT_NerfstudioInputPanel"
    bl_label = "Nerfstudio Camera Generator"
    bl_parent_id = "NERFSTUDIO_PT_NerfstudioMainPanel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    def draw(self, context):

        col = self.layout.column()
        self.layout.label(text="Create Blender Camera From Nerfstudio JSON")
        col = self.layout.column()

        for (prop_name, _) in INPUT_PROPERTIES_NS_CAMERA:
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
        "NS_InputJSONFilePath",
        bpy.props.StringProperty(
            name="JSON Nerfstudio path",
            default="//",
            description="Path for JSON from Nerfstudio editor",
            subtype="FILE_PATH",
        ),
    )
]

OBJ_PROPERTIES = ["NeRF", "RenderCamera"]


def register():
    for (prop_name, prop_value) in INPUT_PROPERTIES:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for (prop_name, prop_value) in INPUT_PROPERTIES_NS_CAMERA:
        setattr(bpy.types.Scene, prop_name, prop_value)

    bpy.types.Scene.NeRF = bpy.props.PointerProperty(type=bpy.types.Object)

    for curr_class in CLASSES:
        bpy.utils.register_class(curr_class)


def unregister():
    for (prop_name, _) in INPUT_PROPERTIES:
        delattr(bpy.types.Scene, prop_name)

    for (prop_name, _) in INPUT_PROPERTIES_NS_CAMERA:
        delattr(bpy.types.Scene, prop_name)

    del bpy.types.Scene.NeRF

    for curr_class in CLASSES:
        bpy.utils.unregister_class(curr_class)


if __name__ == "__main__":
    register()
