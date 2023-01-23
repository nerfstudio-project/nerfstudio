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

### Camera intrinsics

If all of the images share the same camera intrinsics, the values can be placed at the top of the file.

```json
{
  "camera_model": "OPENCV_FISHEYE", // camera model type [OPENCV, OPENCV_FISHEYE]
  "fl_x": 1072.0, // focal length x
  "fl_y": 1068.0, // focal length y
  "cx": 1504.0, // principal point x
  "cy": 1000.0, // principal point y
  "w": 3008, // image width
  "h": 2000, // image height
  "k1": 0.0312, // first radial distorial parameter, used by [OPENCV, OPENCV_FISHEYE]
  "k2": 0.0051, // second radial distorial parameter, used by [OPENCV, OPENCV_FISHEYE]
  "k3": 0.0006, // third radial distorial parameter, used by [OPENCV_FISHEYE]
  "k4": 0.0001, // fourth radial distorial parameter, used by [OPENCV_FISHEYE]
  "p1": -6.47e-5, // first tangential distortion parameter, used by [OPENCV]
  "p2": -1.37e-7, // second tangential distortion parameter, used by [OPENCV]
  "frames": // ... per-frame intrinsics and extrinsics parameters
}
```

Per-frame intrinsics can also be defined in the `frames` field. If defined for a field (ie. `fl_x`), all images must have per-image intrinsics defined for that field. Per-frame `camera_model` is not supported.

```json
{
  // ...
  "frames": [
    {
      "fl_x": 1234
    }
  ]
}
```

### Camera extrinsics

For a transform matrix, the first 3 columns are the +X, +Y, and +Z defining the camera orientation, and the X, Y, Z values define the origin. The last row is to be compatible with homogeneous coordinates.

```json
{
  // ...
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
      // Additional per-frame info
    }
  ]
}
```

### Depth images

To train with depth supervision, you can also provide a `depth_file_path` for each frame in your `transforms.json` and use one of the methods that support additional depth losses (e.g., depth-nerfacto). The depths are assumed to be 16-bit or 32-bit and to be in millimeters to remain consistent with [Polyform](https://github.com/PolyCam/polyform). You can adjust this scaling factor using the `depth_unit_scale_factor` parameter in `NerfstudioDataParserConfig`. Note that by default, we resize the depth images to match the shape of the RGB images.

```json
{
  "frames": [
    {
      // ...
      "depth_file_path": "depth/0001.png"
    }
  ]
}
```

### Masks

:::{admonition} Warning
:class: Warning

The current implementation of masking is inefficient and will cause large memory allocations.
:::

There may be parts of the training image that should not be used during training (ie. moving objects such as people). These images can be masked out using an additional mask image that is specified in the `frame` data.

```json
{
  "frames": [
    {
      // ...
      "mask_path": "masks/mask.jpeg"
    }
  ]
}
```

The following mask requirements must be met:

- Must be 1 channel with only black and white pixels
- Must be the same resolution as the training image
- Black corresponds to regions to ignore
- If used, all images must have a mask
