# Maya Plug-in

## Overview

This extension allows users to combine animated scenes and VFX in Autodesk Maya with NeRF renders in Nerfacto. The extension creates a camera path JSON file that can be read by both Maya and Nerfstudio, then the user composites together an exported mp4 render from Nerfstudio and animation frames from Maya.

This plugin only supports perspective cameras as of now.

This plugin is available in Blender as well. See [Blender VFX add-on](https://docs.nerf.studio/extensions/blender_addon.html)

## Loading the plugin

1. Download the plug-in titled nerfplugin_maya.py, and save it to a safe place on your computer.

2. On the top bar, Windows>Settings/Preferences>Plugin Manager>Browse, and locate nerfplugin_maya.py then click Open


3. You will find that there is a new shelf item called “Nerfstudio”. Click into it, and click on the only icon in it. 

4. A window will pop up that looks like the image below. Let’s go over what each option in the window means:
   - NeRF Representation (mesh): Expects a nerf scene in the form of a mesh object exported from Nerfstudio. Select the desired mesh, then click the Store button. Note that it only accepts one mesh object, and not multiple.
   - Camera object: Expects a camera object. Select the camera, then click the Store button. Note that it only takes one camera, and not multiple.
   - Camera path for Nerfstudio: Expects a file path to a folder so it can generate a json file and store it in that folder
   - Create camera from Nerfstudio JSON: Expects a camera json file so it can read and load its camera path into the scene


## Setting up the scene

1. Export the mesh of the NeRF from Nerfstudio, This will be used as reference for the actual NeRF in Maya. It usually helps to crop the viewport in order to get rid of any artifacts that might affect the output mesh’s geometry. Under Export, select Use Crop and Remove Outliers. Then, generate the command to export the nerf mesh. This exports the mesh with its associated material and texture map.

2. Import the mesh of the NeRF into the Maya scene. You will need to resize, position, and rotate the NeRF representation to fit your scene, since Nerfstudio and Autodesk Maya use different default coordinate systems. The NeRF representation is used as a reference for any animation or VFX done in Maya, and won’t be visible in the final render. Do not delete the history of your NeRF mesh representation at any point!

3. The hotkey ‘6’ toggles the shaded display if you’d like to view the corresponding texture map onto the mesh.

4. Feel free to import as many nerf representations into one scene as you would like, if you want to composite different nerfs together.

5. Start animating and keying camera movements!

## Exporting a camera path from Maya into Nerfstudio

### Generating the JSON File

1. After you finished animating, open up the Nerf VFX Settings window
2. Select the NeRF mesh representation. Then under NeRF Representation (mesh) in the window, click Store
3. Select the camera you’d like to render from. Then under Camera object in the window, click Store
4. Under Camera path for NeRFStudio, click on the folder icon and locate a folder in which you’d like to save your camera_path.json file. Click Generate JSON File. A camera_path.json file will be generated in this folder. This json file will be used by Nerfstudio to generate a camera in the Nerfstudio viewer.

Note that only perspective cameras are supported as of now.

### Rendering in Nerfstudio

1. Run the following command in your terminal to export an mp4 file of the nerf scene:
   <code>ns-render camera-path --load-config {path/to/config.yml} --camera-path-filename  {path/to/camera_path.json} --output-path  {path/to/nerf_maya_render.mp4}</code>
   OR
   Put your camera_path.json file in {path/to/nerfdata/camera_paths/camera_path.json}. You can open up the Nerfstudio viewer, and under “Load path,” select the camera_path.json file. Then, click “Generate Command” and copy and paste the command in the terminal to render out an mp4 file.

## Exporting a camera path from Nerfstudio into Maya

1. Create a camera path in Nerfstudio. Export the nerf as a mesh if you haven’t done so already. Adjust the FOV and resolution as needed. Note that only perspective cameras are supported as of now. Hit “Generate Command” and run the command in the terminal. This will export a .mp4 file of the nerf scene in your renders folder, and a camera path json file in the same folder where your nerf data is.

2. Open up Autodesk Maya. Under NeRF Representation (mesh), select your Nerf mesh representation and click Store. Under Create camera from Nerfstudio JSON, click on the folder icon and locate your camera path json file that you just exported from Nerfstudio. Click on “Create Camera from JSON.” This will generate a new camera in Maya that has the same camera path from Nerfstudio.

## Bringing it together! 

1. Render your animation in Autodesk Maya with your render engine of choice. Be sure to hide your NeRF mesh representation before doing so. Maya uses Arnold by default, but it should not matter whether you use a different one. 
   - [Click here to learn how to render with Arnold](https://www.ucbugg.com/labs/Post_Production/rendering_arnold#heading-2)
   - [Click here to learn how to render out shadows separately](https://www.youtube.com/watch?v=2G9wYcI3fSg)
2. Composite your NeRF mp4 file from Nerfstudio with the animation frames and optional shadow frames rendered from Maya. The NeRF mp4 file runs in 24 fps, so be sure to adjust accordingly when importing your rendered Maya frames. 
   - You can use any compositing software of choice; Adobe AfterEffects tends to a popular choice. [Click here to learn the basics](https://www.ucbugg.com/labs/Post_Production/compositing_layer_management#heading-6)
