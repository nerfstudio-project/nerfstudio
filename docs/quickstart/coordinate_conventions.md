# Coordinate conventions

Here we explain the coordinate conventions for using our repo.

## Camera/view space

We use the OpenGL/Blender (and original NeRF) coordinate convention for cameras. +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. Other codebases such as [Instant NGP](https://github.com/NVlabs/instant-ngp/discussions/153?converting=1#discussioncomment-2187652) may use the COLMAP/OpenCV convention, where the Y and Z axes are flipped from ours but the +X axis remains the same.

## World space

Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane.
