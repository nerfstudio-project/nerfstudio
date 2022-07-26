# Using custom data

Training model on existing datasets is only so fun. If you would like to train on self captured data you will need to process the data into an existing format. Specifically we need to know the camera poses for each image. [COLMAP](https://github.com/colmap/colmap) is a standard tool for extracting poses. It is possible to use other methods like [SLAM](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping) or hardware recorded poses. We intend to add documentation for these other methods in the future.

#### Recover poses using COLMAP

:::{admonition} Notes
:class: warning

- This method assumes the images were captured with a standard _pinhole-like_ camera. 360° captures will not work.
- COLMAP is notoriously finicky. Try your best to capture overlapping, non-blurry images.
  :::

These instructions will recover poses for your own images using [COLMAP](https://github.com/colmap/colmap) and save the data in the `instant_ngp` format.

Please download [this file](https://github.com/NVlabs/instant-ngp/blob/07d8e2ca7232f97397ab73af9b56c7db639d3445/scripts/colmap2nerf.py) and place it at `scripts/colmap2nerf.py` courtesy of the [Instant-NGP](https://github.com/NVlabs/instant-ngp) authors. This script will use COLMAP to process images of a video file into a reconstruction with camera poses. You can see their write-up [here](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) for instructions.

For example, maybe we have a video named `bear.mp4`. We can use the commands described in the Instant-NGP repo to create our dataset. We first create a a data folder. Then, we navigate to this folder. Finally, we run the colmap2nerf.py script with a specified `video_fps` and `aabb_scale`. The recommended number of frames is around 50-150 and the `aabb_scale` is the extent of the scene. The origin is the center of the aabb and `aabb_scale` is the length of each side of the box.

```bash
# Navigate to the repo.
cd /path/to/nerfactory

# Setup the environment variables.
nerfactory_DIR=`pwd`
DATASET_FORMAT=instant_ngp
DATASET_NAME=bear

# Create and navigate to the data folder.
mkdir data/$DATASET_FORMAT/$DATASET_NAME
cd data/$DATASET_FORMAT/$DATASET_NAME

# Run the COLMAP script.
python $nerfactory_DIR/scripts/colmap2nerf.py \
    --video_in /path/to/bear.mp4 \
    --video_fps 2 \
    --run_colmap \
    --aabb_scale 16
```

In the config set:

```yaml

---
data_directory: data/instant_ngp/bear
dataset_format: instant_ngp
```
