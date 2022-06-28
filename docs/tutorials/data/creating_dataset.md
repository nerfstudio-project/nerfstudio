# Using custom data

#### Running COLMAP on your own images

Please download [this file](https://github.com/NVlabs/instant-ngp/blob/07d8e2ca7232f97397ab73af9b56c7db639d3445/scripts/colmap2nerf.py) and place it at `scripts/colmap2nerf.py` courtesy of the [Instant-NGP](https://github.com/NVlabs/instant-ngp) authors. This script will use [COLMAP](https://github.com/colmap/colmap) to process images of a video file into a reconstruction with camera poses. You can see their write-up [here](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) for instructions.

For example, maybe we have a video named `bear.mp4`. We can use the commands described in the Instant-NGP repo to create our dataset. We first create a a data folder. Then, we navigate to this folder. Finally, we run the colmap2nerf.py script with a specified `video_fps` and `aabb_scale`. The recommended number of frames is around 50-150 and the `aabb_scale` is the extent of the scene. The origin is the center of the aabb and `aabb_scale` is the length of each side of the box.

```bash
# Navigate to the repo.
cd /path/to/pyrad

# Setup the environment variables.
PYRAD_DIR=`pwd`
DATASET_FORMAT=instant_ngp
DATASET_NAME=bear

# Create and navigate to the data folder.
mkdir data/$DATASET_FORMAT/$DATASET_NAME
cd data/$DATASET_FORMAT/$DATASET_NAME

# Run the COLMAP script.
python $PYRAD_DIR/scripts/colmap2nerf.py \
    --video_in /path/to/bear.mp4 \
    --video_fps 2 \
    --run_colmap \
    --aabb_scale 16
```

In the config set:
```yaml
dataset_format: instant_ngp
```