# Blender VFX add-on

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211442247-99d1ebc7-3ef9-46f7-9bcc-0e18553f19b7.PNG">
</p>


## Overview

This Blender add-on allows for compositing with a Nerfstudio render as a background layer by generating a camera path JSON file from the Blender camera path, as well as a way to import Nerfstudio JSON files as a Blender camera baked with the Nerfstudio camera path. This is achieved by importing a mesh or point-cloud representation of the NeRF scene from Nerfstudio into Blender and getting the camera coordinates relative to the transformations of the NeRF representation. Dynamic FOV from the Blender camera is supported and will match the Nerfstudio render. Perspective and equirectangular cameras are also supported.

## Prerequisites

[Blender](https://www.blender.org/) on Mac or Windows. The python file add-on can be downloaded from the main repo in the `scripts/blender` folder. Before starting, export the mesh or point cloud representation of the NeRF from Nerfstudio, which will be used as reference for the actual NeRF in the Blender scene. Mesh export at a good quality is preferred, however, if the export is not clear or the NeRF is very large, a detailed point cloud export will also work.

## Add-on Setup Instructions

1. Open a new or existing scene to add the NeRF background in.


2. Install and enable the add-on in Blender in Edit → Preferences → Add-Ons. The add-on will be visible in the Render Properties tab on the right panel.
<p align="center">
    <img width="400" alt="image" src="https://user-images.githubusercontent.com/9502341/211244462-aec10c53-cb33-4406-ae69-9728c5551f24.png">
    <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211244590-0c415e19-9b6f-4023-8ce0-5480a3cc376c.png">
</p>

3. Import the mesh or point cloud representation of the NeRF into the scene. It is preferred that a mesh is used at a good enough fidelity. You may need to crop the mesh further. Since it is used as a reference and won't be visible in the final render, only the parts that the blender animation will interact with may be necessary to import.

4. Resize, position, or rotate the NeRF representation to fit your scene.


## Generate Nerfstudio JSON Camera Path from Blender Camera

1. There are a few ways to hide the reference mesh for the Blender render
    - In object properties, select "Shadow Catcher". This makes the representation invisible in the render, but all shadows cast on it will render. You may have to switch to the cycles renderer to see the shadow catcher option.    
    <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211244787-859ca9b5-6ba2-4056-aaf2-c89fc6370c2a.png">
    
    - Note: This may not give ideal results if the mesh is not very clear or occludes other objects in the scene. If this is the case, you can hide the mesh from the render instead by clicking the camera button in the Outliner next to its name.   
    <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211244858-54091d36-086d-4211-a7d9-75dcfdcb1436.png">


2. Verify that the animation plays and the NeRF representation does not occlude the camera.
3. Go to the Nerfstudio Add-on in Render Properties and expand the "Nerfstudio Path Generator" tab in the panel. Use the object selector to select the NeRF representation. Then, select the file path for the output JSON camera path file. 
4. Click "Generate JSON file". The output JSON file is named `camera_path_blender.json`.    
<img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211442361-999fe040-a1ed-43f0-b079-0c659d70862f.png">


5. Render the NeRF with the generated camera path using Nerfstudio in the command prompt or terminal.

6. Before rendering the Blender animation, go to the Render Properties and in the Film settings select "Transparent" so that the render will be rendered with a clear background to allow it to be composited over the NeRF render.     
<img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211244801-c555c3b5-ab3f-4d84-9f64-68559b03ff37.png">


7. Now the scene can be rendered and composited over the camera aligned Nerfstudio render.  
<img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211245025-2ef5adbe-9306-4eab-b761-c78e3ec187bd.png">

### Examples

<p align="center">
    <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/212463397-ba34d60f-a744-47a1-95ce-da6945a7fc00.gif">
    <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/212461881-e2096710-4732-4f20-9ba7-1aeb0d0d653b.gif">
</p>

### Additional details

- You can also apply an environment texture to the Blender scene by using Nerfstudio to render an equirectangular image (360 image) of the NeRF from a place in the scene. 
- The settings for the Blender equirectangular camera are: "Panoramic" camera type and panorama type of "Equirectangular".  
<img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211245895-76dfb65d-ed81-4c36-984a-4c683fc0e1b4.png">

- The generated JSON camera path follows the user specified frame start, end, and step fields in the Output Properties in Blender. The JSON file specifies the user specified x and y render resolutions at the given % in the Output Properties.
- The add-on computes the camera coordinates based on the active camera in the scene.
- FOV animated changes of the camera will be matched with the NeRF render.
- Perspective and equirectangular cameras are supported and can be selected within Blender.
- The generated JSON file can be imported into Nerfstudio. Each keyframe of the camera transform in the frame sequence in Blender will be a keyframe in Nerfstudio. The exported JSON camera path is baked, where each frame in the sequence is a keyframe. This is to ensure that frame interpolation across Nerfstudio and Blender do not differ.
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211245108-587a97f7-d48d-4515-b09a-5644fd966298.png">
- It is recommended to run the camera path in the Nerfstudio web interface with the NeRF to ensure that the NeRF is visible throughout the camera path.
- The fps and keyframe timestamps are based on the "Frame Rate" setting in the Output Properties in Blender.
- The NeRF representation can also be transformed (position, rotation, and scale) as well as animated.
- It is recommended to export a high fidelity mesh as the NeRF representation from Nerfstudio. However if that is not possible or the scene is too large, a point cloud representation also works.
- For compositing, it is recommended to convert the video render into image frames to ensure the Blender and NeRF renders are in sync.
- Currently, dynamic camera focus is not supported.
- Fisheye and orthographic cameras are not supported.

## Create Blender Camera from Nerfstudio JSON Camera Path

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211246016-88bc6031-01d4-418f-8230-fb8c29212200.png">
</p>


1. Expand the "Nerfstudio Camera Generator" tab in the panel. After inputting the NeRF representation, select the JSON Nerfstudio file and click "Create Camera from JSON"

2. A new camera named "NerfstudioCamera" should appear in the viewport with the camera path and FOV of the input file. This camera's type will match the Nerfstudio input file, except fisheye cameras.

### Additional details

- Since only the camera path, camera type, and FOV are applied on the created Blender camera, the timing and duration of the animation may need to be adjusted.
- Fisheye cameras imported from Nerfstudio are not supported and will default to perspective cameras.
- Animated NeRF representations will not be reflected in the imported camera path animation.
- The newly created camera is not active by default, so you may need to right click and select "Set Active Camera".
- The newly created Blender camera animation is baked, where each frame in the sequence is a keyframe. This is to ensure that frame interpolation across Nerfstudio and Blender do not differ.
- The resolution settings from the input JSON file do not affect the Blender render settings
- Scale of the camera is not keyframed from the Nerfstudio camera path.
- This newly created camera has a sensor fit of "Vertical"


## Implementation Details

For generating the JSON camera path, we iterate over the scene frame sequence (from the start to the end with step intervals) and get the camera 4x4 world matrix at each frame. The world transformation matrix gives the position, rotation, and scale of the camera. We then obtain the world matrix of the NeRF representation at each frame and transform the camera coordinates with this to get the final camera world matrix. This allows us to re-position, rotate, and scale the NeRF representation in Blender and generate the right camera path to render the NeRF accordingly in Nerfstudio. Additionally, we calculate the FOV of the camera at each frame based on the sensor fit (horizontal or vertical), angle of view, and aspect ratio. 
Next, we construct the list of keyframes which is very similar to the world matrices of the transformed camera matrix.
Camera properties in the JSON file are based on user specified fields such as resolution (user specified in Output Properties in Blender), camera type (Perspective or Equirectangular). In the JSON file, `aspect` is specified as 1.0, `smoothness_value` is set to 0, and `is_cycle` is set to false. The Nerfstudio render is the fps specified in Blender where the duration is the total number of frames divided by the fps.
Finally, we construct the full JSON object and write it to the file path specified by the user.

For generating the camera from the JSON file, we create a new Blender camera based on the input file and iterate through the `camera_path` field in the JSON to get the world matrix of the object from the `matrix_to_world` and similarly get the FOV from the `fov` fields. At each iteration, we set the camera to these parameters and insert a keyframe based on the position, rotation, and scale of the camera as well as the focal length of the camera based on the vertical FOV input. 

