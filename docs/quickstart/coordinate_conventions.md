# Coordinate conventions

Here we explain the coordinate conventions for using our repo.

## Camera/view space

We use the OpenGL/Blender (and original NeRF) coordinate convention for cameras. +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. Other codebases may use the COLMAP/OpenCV convention, where the Y and Z axes are flipped from ours but the +X axis remains the same.

The `transforms.json` has a similar format to [Instant NGP](https://github.com/NVlabs/instant-ngp). For a transform matrix, the first 3 columns are the +X, +Y, and +Z defining the camera orientation, and the X, Y, Z values define the origin. The last row is to be compatible with homogeneous coordinates.

```json
{
  "frames": [
    {
      "file_path": "images/frame_00001.jpeg",
      "transform_matrix": [
        // [+X0 +Y0 +Z0 X]
        // [+X1 +Y1 +Z1 Y]
        // [+X2 +Y2 +Z2 Y]
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

## World space

Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane. In the viewer, you'll notice that red, green, and blue vectors correspond to X, Y, and Z respectively.
