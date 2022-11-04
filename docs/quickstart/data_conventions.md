# Data conventions

## Coordinate conventions

Here we explain the coordinate conventions for using our repo.

### Camera/view space

We use the OpenGL/Blender (and original NeRF) coordinate convention for cameras. +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. Other codebases may use the COLMAP/OpenCV convention, where the Y and Z axes are flipped from ours but the +X axis remains the same.

### World space

Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane. In the viewer, you'll notice that red, green, and blue vectors correspond to X, Y, and Z respectively.

<hr>

## Dataset format

Our explanation here is for the nerfstudio data format. The `transforms.json` has a similar format to [Instant NGP](https://github.com/NVlabs/instant-ngp).

**Camera intrinsics**

At the top of the file, we specify the camera intrinsics. We assume that all the intrinsics parameters are the same for every camera in the dataset. The following is an example

```json
{
  "fl_x": 1072.281897246229, // focal length x
  "fl_y": 1068.6906965388932, // focal length y
  "cx": 1504.0, // principal point x
  "cy": 1000.0, // principal point y
  "w": 3008, // image width
  "h": 2000, // image height
  "camera_model": "OPENCV_FISHEYE", // camera model type
  "k1": 0.03126218448029553, // first radial distorial parameter
  "k2": 0.005177020067511987, // second radial distorial parameter
  "k3": 0.0006640977794272005, // third radial distorial parameter
  "k4": 0.00010067035656515042, // fourth radial distorial parameter
  "p1": -6.472477652140879e-5, // first tangential distortion parameter
  "p2": -1.374647851912992e-7, // second tangential distortion parameter
  "frames": // ... extrinsics parameters explained below
}
```

The valid `camera_model` strings are currently "OPENCV" and "OPENCV_FISHEYE". "OPENCV" (i.e., perspective) uses k1-2 and p1-2. "OPENCV_FISHEYE" uses k1-4.

**Camera extrinsics**

For a transform matrix, the first 3 columns are the +X, +Y, and +Z defining the camera orientation, and the X, Y, Z values define the origin. The last row is to be compatible with homogeneous coordinates.

```json
{
  // ... intrinsics parameters
  "frames": [
    {
      "file_path": "images/frame_00001.jpeg",
      "transform_matrix": [
        // [+X0 +Y0 +Z0 X]
        // [+X1 +Y1 +Z1 Y]
        // [+X2 +Y2 +Z2 Z]
        // [0.0 0.0 0.0 1]
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```
