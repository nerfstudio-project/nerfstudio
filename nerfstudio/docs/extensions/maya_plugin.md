# Autodesk Maya Plug-in

<p align="center">
    <img width="500" alt="image" src="https://github.com/user-attachments/assets/8fde7cae-d320-4de7-a5af-cb0204dd7959">
</p>

## Overview

**[Link to the plug-in](https://github.com/nerfstudio-project/nerfstudio/tree/main/nerfstudio/scripts/maya)**

This extension allows users to combine animated scenes and VFX in Autodesk Maya with NeRF renders in Nerfacto. The extension creates a camera path JSON file that can be read by both Maya and Nerfstudio, then the user composites together an exported mp4 render from Nerfstudio and animation frames from Maya.

<center>
<img width="800" alt="1" src="https://github.com/user-attachments/assets/8b85ee8a-f365-4de1-a47d-ef4db5a6184d" />
</center>

This plugin only supports perspective cameras as of now, and can support any renderer (e.g. Arnold, Redshift).

This plugin is available in Blender as well. See [Blender VFX add-on](https://docs.nerf.studio/extensions/blender_addon.html)

## Loading the plugin

1. Download the plug-in titled nerfplugin_maya.py, and save it to a safe place on your computer.

2. On the top bar, Windows>Settings/Preferences>Plugin Manager>Browse, and locate nerfplugin_maya.py then click Open

<center>
<img width="800" alt="1" src="https://github.com/user-attachments/assets/bf0b9408-ffda-466f-b322-0858367218c2" />
</center>

<center>
<img width="800" alt="2" src="https://github.com/user-attachments/assets/4cdd93f0-bea4-4e03-bcb0-aca34f5e9a8c" />
</center>

3. You will find that there is a new shelf item called “Nerfstudio”. Click into it, and click on the only icon in it.

<center>
<img width="800" alt="3" src="https://github.com/user-attachments/assets/db7a8e29-5e88-42a4-93fc-d3a49b3e50ab" />
</center>

4. A window will pop up that looks like the image below. Let’s go over what each option in the window means:

   - NeRF Representation (mesh): Expects a nerf scene in the form of a mesh object exported from Nerfstudio. Select the desired mesh, then click the Store button. Note that it only accepts one mesh object, and not multiple.

   - Camera object: Expects a camera object. Select the camera, then click the Store button. Note that it only takes one camera, and not multiple.

   - Camera path for Nerfstudio: Expects a file path to a folder so it can generate a json file and store it in that folder

   - Create camera from Nerfstudio JSON: Expects a camera json file so it can read and load its camera path into the scene

<center>
<img width="400" alt="4" src="https://github.com/user-attachments/assets/62ec35d0-afe2-42a4-a2ca-9315d391aeb3" />
</center>

## Setting up the scene

1. Export the mesh of the NeRF from Nerfstudio, This will be used as reference for the actual NeRF in Maya. It usually helps to crop the viewport in order to get rid of any artifacts that might affect the output mesh’s geometry. Under Export, select Use Crop and Remove Outliers. Then, generate the command to export the nerf mesh. This exports the mesh with its associated material and texture map.

2. Import the mesh of the NeRF into the Maya scene. You can do this by dragging and dropping the exported .obj file into the scene.
   - You will need to resize, position, and rotate the NeRF representation to fit your scene, since Nerfstudio and Autodesk Maya use different default coordinate systems. The NeRF representation is used as a reference for any animation or VFX done in Maya, and won’t be visible in the final render. Do not delete the history of your NeRF mesh representation at any point!

3. The hotkey ‘6’ toggles the shaded display if you’d like to view the corresponding texture map onto the mesh.

4. Feel free to import as many nerf representations into one scene as you would like, if you want to composite different nerfs together.

5. Start animating and keying camera movements!

## Exporting a camera path from Maya into Nerfstudio

### Generating the JSON File

1. After you finished animating, open up the Nerf VFX Settings window

2. Select the NeRF mesh representation. Then under NeRF Representation (mesh) in the window, click Store

3. Select the camera you’d like to render from. Then under Camera object in the window, click Store

4. Under Camera path for NeRFStudio, click on the folder icon and locate a folder in which you’d like to save your camera_path.json file. Click Generate JSON File. A camera_path.json file will be generated in this folder. This json file will be used by Nerfstudio to generate a camera in the Nerfstudio viewer.

<center>
<img width="800" alt="image" src="https://github.com/user-attachments/assets/311e1ef4-a63b-4d31-a50f-a2a4b20653e6" />
</center>

Note that only perspective cameras are supported as of now.

### Rendering in Nerfstudio

1. Run the following command in your terminal to export an mp4 file of the nerf scene:
   <code>ns-render camera-path --load-config {path/to/config.yml} --camera-path-filename  {path/to/camera_path.json} --output-path  {path/to/nerf_maya_render.mp4}</code>
   OR
   Put your camera_path.json file in {path/to/nerfdata/camera_paths/camera_path.json}. You can open up the Nerfstudio viewer, click on the Render tab, and under “Load path,” select the camera_path.json file. Then, click “Generate Command” and copy and paste the command in the terminal to render out an mp4 file.

   <center>
   <img width="800" alt="image" src="https://github.com/user-attachments/assets/1a775099-6201-494f-bb99-dfbba72289d2" />
   </center>

## Exporting a camera path from Nerfstudio into Maya

1. Create a camera path in Nerfstudio. Export the nerf as a mesh if you haven’t done so already. Adjust the FOV and resolution as needed. Note that only perspective cameras are supported as of now. Hit “Generate Command” and run the command in the terminal. This will export a .mp4 file of the nerf scene in your renders folder, and a camera path json file in the same folder as your nerf data.

 <center>
   <img width="800" alt="image" src="https://github.com/user-attachments/assets/2dc92418-787e-4fc8-8218-8f33ee0a2f88" />
</center>

2. Open up Autodesk Maya. Under NeRF Representation (mesh), select your Nerf mesh representation and click Store. Under Create camera from Nerfstudio JSON, click on the folder icon and locate your camera path json file that you just exported from Nerfstudio. Click on “Create Camera from JSON.” This will generate a new camera in Maya that has the same camera path from Nerfstudio.

<center>
   <img width="800" alt="image" src="https://github.com/user-attachments/assets/37608bc8-0b40-42b5-bf51-d821b0beae35" />
</center>

## Bringing it together!

1. Render your animation in Autodesk Maya with your render engine of choice. Be sure to hide your NeRF mesh representation before doing so. Maya uses Arnold by default, but it should not matter whether you use a different one.

   - [Click here to learn how to render with Arnold](https://www.ucbugg.com/labs/Post_Production/rendering_arnold#heading-2)

   - [Click here to learn how to render out shadows separately](https://www.youtube.com/watch?v=2G9wYcI3fSg)

2. Composite your NeRF mp4 file from Nerfstudio with the animation frames and optional shadow frames rendered from Maya. The NeRF mp4 file runs in 24 fps, so be sure to adjust accordingly when importing your rendered Maya frames.

   - You can use any compositing software of choice; Adobe AfterEffects tends to a popular choice. [Click here to learn the basics](https://www.ucbugg.com/labs/Post_Production/compositing_layer_management#heading-6)

<center>
   <img width="1481" alt="9" src="https://github.com/user-attachments/assets/678027ac-37d5-4fca-81ce-a7190f83fda4" />
</center>

## Implementation
This plugin is designed to create and process JSON files that Nerfstudio exports as a camera path. When creating a camera path from a selected camera and meshed Nerf in Maya, we calculate three things for the camera: the transformation matrix relative to the meshed Nerf, the field of view, and the aspect ratio. We also set the frame rate to default at 24 frames per second. Because Autodesk Maya uses a Y-up coordinate system by default, whereas Nerfstudio uses a Z-up coordinate system, we must apply the transformation matrix along with this conversion. After querying this information per frame, we write into a JSON file that can be parsed by Nerfstudio and replicate the camera motion that was in Maya. 

When importing a camera path into Maya, the plugin reads all the information from the exported camera path JSON file and sets a keyframe at every frame in the animation. It calculates where the position of the camera would be relative to the meshed Nerf, as well as its focal length and aspect ratio, and creates a camera within Maya based off of these attributes.

Happy rendering!

## Credits
Models and rigs taken from [Agora](https://agora.community).
- [Link to Pizza rig](https://agora.community/content/pizza-maya)
- [Link to Seagull rig](https://agora.community/content/seagull)

Datasets from Cyrus Vachha
