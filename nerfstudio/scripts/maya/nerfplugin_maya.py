# type: ignore

# Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import maya.api.OpenMaya as om
import maya.cmds as mc
import maya.mel as mel
import sys
import os
import math
import json 


"""
nerfplugin.py
"""

class CreateJSONCameraPath():
    """Create a JSON camera path from the Maya camera animation."""
    
    def __init__(self, nerf, camera, jsonfp):
        self.NeRF = mc.textField(nerf, query=True, text=True)
        self.cam_obj = mc.textField(camera, query=True, text=True)
        self.framestep = 1
        self.frame_start = mc.playbackOptions(query=True, min=True)
        self.frame_end = mc.playbackOptions(query=True, max=True)
        self.fov_list = [] # list of FOV at every frame
        self.transformed_camera_path_mat = [] # final transformed world matrix of the camera at each frame
        
        self.complete_json_obj = {}  # full Nerfstudio input json object
        self.file_path_json = mc.textField(jsonfp, query=True, text=True) # file path input

    
    def get_camera_coordinates(self):
        """Create a list of transformed camera coordinates and converted FOV.

            Note that Maya transformation matrices are stored transposed, and use
            the right-hand y-up coordinate system while Nerfstudio uses the right-hand
            z-up coordinate system.
        """

        org_camera_path_mat = []  # list of world matrix of the active camera at each frame
        nerf_mesh_mat_list = []  # list of world matrix of the NeRF mesh at each frame

        curr_frame = self.frame_start
        transformation_mat = om.MMatrix([1,0,0,0,
                                  0,0,1,0,
                                  0,-1,0,0,
                                  0,0,0,1]) # change of basis from Maya coordinates to Nerfstudio coordinates
        
        while curr_frame <= self.frame_end:
            mc.currentTime(curr_frame, edit=True)
            cam_mat_list = mc.xform(self.cam_obj, query=True, matrix=True, ws=True)
            
            org_camera_path_mat += [ transformation_mat * (om.MMatrix(cam_mat_list)).transpose()]
            
            self.fov_list += [self.getFOV()]

            curr_frame += self.framestep

            nerf_mat_list = mc.xform(self.NeRF, query=True, matrix=True, ws=True)
            nerf_mesh_mat_list += [transformation_mat * om.MMatrix(nerf_mat_list).transpose()]
          
            if self.framestep == 0: # only one frame
                break
        
        # transform the camera world matrix based on the NeRF mesh transformation
        for i, org_cam_path_mat_val in enumerate(org_camera_path_mat):
            self.transformed_camera_path_mat += [nerf_mesh_mat_list[i].inverse() * org_cam_path_mat_val]

    def getFOV(self):
        """ Get the vertical field of view depending on the fit resolution gate."""

        renderratio = mc.getAttr("defaultResolution.width")/mc.getAttr("defaultResolution.height")
        fitresgate = mc.getAttr(self.cam_obj + ".filmFit")

        f = mc.camera(self.cam_obj, query=True, focalLength=True) # in mm

        if fitresgate == 0:
            # full
            if renderratio > 1:
                # horizontal
                fitresgate = 1
            else:
                # vertical
                fitresgate = 2
        if fitresgate == 1  or fitresgate == 3:
            # horizontal or overscan
            hfa = mc.camera(self.cam_obj, query=True, horizontalFilmAperture=True) * 25.4 # in mm
            fov = math.degrees(2 * math.atan(hfa  / (2 * f * renderratio)))
            
        else:
            # vertical
            vfa = mc.camera(self.cam_obj, query=True, verticalFilmAperture=True) * 25.4 # in mm
            fov = math.degrees(2 * math.atan( vfa / (2 * f)))

        return fov
    
    def get_list_from_matrix_path(self, input_mat):
        """Flatten matrix to list for camera path."""
        return list(input_mat)
    
    def get_list_from_matrix_keyframe(self, input_mat):
        """Flatten matrix to list for keyframes."""
        return list(input_mat)
    
    def write_json_to_file(self):
        """Write the JSON object to a new file."""

        full_abs_file_path = os.path.abspath(self.file_path_json + "/camera_path.json")
        with open(full_abs_file_path, "w", encoding="utf8") as output_json_camera_path:
            output_json_camera_path.truncate(0)
            output_json_camera_path.write(self.complete_json_obj)

        self.complete_json_obj = {}
        print("\nFinished creating camera path json file at " + full_abs_file_path + "\n")

    def construct_json_obj(self):
        """Get fields for JSON camera path. Supports only perspective cameras for now.
        Maybe other cameras later"""

        # Only support perspective cameras
        cam_type = mc.camera(self.cam_obj, query=True, orthographic=True)
        if cam_type == False:
            cam_type = "perspective"
        else:
            self.report(
                {"WARNING"}, "Nerfstudio Add-on Warning: Only perspective cameras are supported"
            )
            cam_type = "perspective"

        # get FPS of scene
        time_unit = mc.currentUnit(query=1, t=1)
        index = mel.eval(f'getIndexFromCurrentUnitCmdValue("{time_unit}")') - 1
        fps_name = mel.eval(f'getTimeUnitDisplayString({index});')
        render_fps = float(fps_name.split(' ')[0])


        # case when step size is 0 there is only one frame
        if self.framestep == 0:
            render_seconds = 1 / render_fps
        else:
            render_seconds = (
                (self.frame_end - self.frame_start) // (self.framestep) + 1
            ) / render_fps

        smoothness_value = 0
        is_cycle = False

        # construct camera path
        final_camera_path = []

        for i, transformed_camera_path_mat_val in enumerate(self.transformed_camera_path_mat):
            camera_path_elem = {
                "camera_to_world": self.get_list_from_matrix_path(transformed_camera_path_mat_val),
                "fov": self.fov_list[i],
                "aspect": mc.camera(self.cam_obj, query=True, aspectRatio=True),
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
                "aspect": mc.camera(self.cam_obj, query=True, aspectRatio=True),
                "properties": curr_properties,
            }
            keyframe_list += [keyframe_elem]
        
        render_height = int(mc.getAttr("defaultResolution.height"))
        render_width = int(mc.getAttr("defaultResolution.width"))

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
        
    def execute(self):

        if self.NeRF==None or self.cam_obj==None:
            sys.stderr.write(" Please input NeRF representation or camera object")
        
        if self.file_path_json == "":
            sys.stderr.write("Please input a file path for the output JSON")
        
        self.get_camera_coordinates()
        self.construct_json_obj()
        self.write_json_to_file()

        return 0

class  ReadJSONinputCameraPath():
    """Create a camera with an animation path based on an input Nerfstudio JSON."""
    
    def __init__(self, nerf, fptxt):
        self.NeRF = mc.textField(nerf, query=True, text=True)
        self.cam_obj = None
  
        self.file_path_json = mc.textField(fptxt, query=True, text=True)

        self.fov_list = []  # list of FOV at each frame
        self.transformed_camera_path_mat = []  # final transformed world matrix of the camera at each frame
        self.input_json = None

    def read_camera_coordinates(self):
        """Read the camera coordinates (world matrix and fov) from the json camera path."""

        json_cam_path = self.input_json["camera_path"]

        keyframe_counter = 0

        for cam_keyframe in json_cam_path:
            cam_to_world = cam_keyframe["camera_to_world"]

            orig_cam_mat = om.MMatrix(cam_to_world)
            self.transformed_camera_path_mat += [ orig_cam_mat.transpose() * om.MMatrix(mc.xform(self.NeRF, query=True, matrix=True, ws=True))]
            
            self.fov_list += [cam_keyframe["fov"]]

            keyframe_counter += 1

    def generate_camera(self):
        """Create a new camera with the animation (position and fov) and the corresponding type."""

        json_cam_path = self.input_json["camera_path"]
        self.cam_obj = mc.camera(name="nerfstudioCam")
        curr_frame = 0
        aspect = mc.camera(self.cam_obj[0], query=True, aspectRatio=True)
        
        while curr_frame < len(json_cam_path):

            # animate camera transform
            mc.currentTime(curr_frame, edit=True)
            world_matrix_list = [self.transformed_camera_path_mat[curr_frame].getElement(i, j) for i in range(4) for j in range(4)]
            
            mc.xform(self.cam_obj[0], matrix=world_matrix_list, worldSpace=True, scale=[1,1,1])
            mc.setKeyframe(self.cam_obj[0], attribute="translate")
            mc.setKeyframe(self.cam_obj[0], attribute="rotate")

            # animate fov 
            mc.camera(self.cam_obj[0], edit=True, vfv = self.fov_list[curr_frame]) #set fov
            mc.setKeyframe(self.cam_obj[0], attribute="verticalFieldOfView", t=curr_frame)
            
            mc.setAttr(f"{self.cam_obj[0]}.filmFit", 2) # set film gate to vertical
            mc.setKeyframe(self.cam_obj[0], attribute="filmFit", t=curr_frame)
           
            #focal length
            hfa = mc.camera(self.cam_obj[0], query=True, horizontalFilmAperture=True) * 25.4 # mm
            mc.camera(self.cam_obj[0], edit=True, focalLength = hfa / (2 * aspect * math.tan( math.radians(self.fov_list[curr_frame] / 2))))
            mc.setKeyframe(self.cam_obj[0], attribute="focalLength", t=curr_frame)

            curr_frame += 1

    def execute(self):
        """Execute camera creation process."""
        # check input
        if self.NeRF==None or self.cam_obj==None:
            sys.stderr.write(" Please input NeRF representation or camera object")
        
        
        if self.file_path_json == "":
            sys.stderr.write("Please input a file path for the output JSON")

        full_abs_file_path = self.file_path_json
        with open(full_abs_file_path, encoding="utf8") as json_ns_file:
            self.input_json = json.load(json_ns_file)

        self.read_camera_coordinates()
        self.generate_camera()

# --- MAYA UI PANEL --- #

def maya_useNewAPI():
    pass

commandName = "nerfCommand"

class pluginCommand(om.MPxCommand):

    def __init__(self):
        om.MPxCommand.__init__(self)

    def doIt(self, args):
        window = Window()
        window.createWindow()
        return 0

    @staticmethod
    def cmdCreator():
        return pluginCommand()

# initialize and uninitialize plugin
def initializePlugin(mobject):
    
    mplugin = om.MFnPlugin(mobject)
    
    try:
        mplugin.registerCommand(commandName, pluginCommand.cmdCreator)
        
    except:
        sys.stderr.write("Failed to register command:" + commandName)
    addNewShelf()

def uninitializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(commandName)
    except:
        sys.stderr.write("Failed to deregister command:" + commandName)

def addNewShelf(): # initializing a new shelf for Nerfstudio stuff
    def replacebuttons(shelf_name):
        shelf_buttons = mc.shelfLayout(shelf_name, query=True, childArray=True) or []
        
        for button in shelf_buttons:
            mc.deleteUI(button)
            
        mc.shelfButton(
            label='Nerfstudio Camera Path',
            command=lambda *args: mc.nerfCommand(),
            annotation='Camera path json generator for nerfstudio viewer, or load camera path json into maya scene',
            image1='pythonFamily.png',
            parent='Nerfstudio'
        )
    
    if not mc.shelfLayout('Nerfstudio', exists=True):
        mc.shelfLayout('Nerfstudio', parent='ShelfLayout')
   
    replacebuttons('Nerfstudio')

# sets up the window UI
class Window():
    def __init__(self):
        self.title = "NeRF VFX Settings"
        self.size = (350, 300)
    
    def createWindow(self):
        if mc.window("NeRF VFX Settings", exists=True):
            mc.deleteUI("NeRF VFX Settings", window=True)

        self.window = mc.window("NeRF VFX Settings", title=self.title, widthHeight= self.size, sizeable = True)
        mc.columnLayout(adjustableColumn = True, columnAttach=('both', 5), rowSpacing=5)
        
        mc.separator(h=10) # Below: Select nerf button

        mc.text("NeRF Representation (mesh)")
        mc.rowLayout(adjustableColumn=1, numberOfColumns=2)
        meshtxtfield = mc.textField(editable=False)
        mc.button(label='Store', command=lambda *args: self.store_nerf(meshtxtfield))
        mc.setParent('..')

        mc.separator(h=10) # Below: Select camera to render button
        
        mc.text("Camera object")
        mc.rowLayout(adjustableColumn=1, numberOfColumns=2)
        camtxtfield = mc.textField(editable=False)
        mc.button(label='Store', command=lambda *args: self.store_cam(camtxtfield))
        mc.setParent('..')

        mc.separator(h=10) #Below: Create a Nerfstudio JSON

        mc.text("Camera path for NerfStudio")
        mc.rowLayout(numberOfColumns=2, adjustableColumn=1, columnAlign=(1, 'left'), columnAttach=(1, 'both', 5))
        file_path_field = mc.textField(editable=False)
        mc.iconTextButton(style='iconOnly', image1='fileOpen.png', command=lambda *args: self.open_file_browser(file_path_field))
        mc.setParent('..')
        mc.button(label="Generate JSON File", command= lambda *args: CreateJSONCameraPath(meshtxtfield, camtxtfield, file_path_field).execute())

        mc.separator(h=10) # Below: Upload JSON file and import into Maya

        mc.text("Create camera from NerfStudio JSON")
        mc.rowLayout(numberOfColumns=2, adjustableColumn=1, columnAlign=(1, 'left'), columnAttach=(1, 'both', 5))
        file_path_field2 = mc.textField(editable=False)
        mc.iconTextButton(style='iconOnly', image1='fileOpen.png', command=lambda *args: self.open_file_browser_json(file_path_field2))
        mc.setParent('..')
        mc.button(label="Create Camera From JSON", command= lambda *args: ReadJSONinputCameraPath(meshtxtfield, file_path_field2).execute())

        mc.separator(h=10)
        
        mc.button(label="Close", command= lambda *args: mc.deleteUI(self.window, window=True))
        
        mc.showWindow(self.window)
    
    ### helpers
    def store_nerf(self, textfield):
            try:
                sel = mc.ls(sl=True)[0]
            except:
                sys.stderr.write("No object is selected! \n")

            shape = mc.listRelatives(sel, shapes=True)
            if mc.objectType(shape, isType ='mesh'):
                mc.textField(textfield, edit=True, text=sel)
            else:
                sys.stderr.write("Object selected is not a mesh \n")
            
    
    def store_cam(self, textfield):
            try:
                sel = mc.ls(sl=True)[0]
            except:
                sys.stderr.write("No object is selected!")

            shape = mc.listRelatives(sel, shapes=True)
            if mc.objectType(shape, isType ='camera'):
                mc.textField(textfield, edit=True, text=sel)
            else:
                sys.stderr.write("Object selected is not a camera \n")
        
            
    def open_file_browser(self, file_path_field):
            file_path = mc.fileDialog2(fileMode=3, caption="Select a Folder")
            if file_path:
                mc.textField(file_path_field, edit=True, text=file_path[0])

    def open_file_browser_json(self, file_path_field):
        file_path = mc.fileDialog2(fileMode=1, caption="Select a JSON File", fileFilter="JSON Files (*.json)")
        if file_path:
            mc.textField(file_path_field, edit=True, text=file_path[0])
