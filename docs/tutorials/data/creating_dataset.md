# Using custom data

Training model on existing datasets is only so fun. If you would like to train on self captured data you will need to process the data into an existing format. Specifically we need to know the camera poses for each image. [COLMAP](https://github.com/colmap/colmap) is a standard tool for extracting poses. It is possible to use other methods like [SLAM](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping) or hardware recorded poses. We intend to add documentation for these other methods in the future.

## nerfstudio dataset

To assist running on custom data we have a script that will process a video or folder of images into a format that is compatible with nerfstudio. We use [COLMAP](https://colmap.github.io) and [FFmpeg](https://ffmpeg.org/download.html) in our data processing script, please have these installed. We have provided a quickstart to installing COLMAP below, FFmpeg can be downloaded from [here](https://ffmpeg.org/download.html)

To process your data run:

```bash
python scripts/process_data.py --data FOLDER_OR_VIDEO --output-dir PROCESSED_DATA_DIR
```

If your data was captured with a fisheye camera add `--camera-type fisheye`.

:::{admonition} Tip
:class: info

- COLMAP can be finicky. Try your best to capture overlapping, non-blurry images.
  :::

### Training on your data

Simply specify that you are using the `nerfstudio` dataparser and point the data directory to your processed data.

```bash
python scripts/train.py instant-ngp nerfstudio-data --pipeline.datamanager.dataparser.data-directory PROCESSED_DATA_DIR
```

### Installing COLMAP

The COLMAP maintainers recommend using VCPKG to install COLAMP. If the following command do not work, please refer to the [COLMAP installation guide](https://colmap.github.io/install.html) for additional installation methods.

::::::{tab-set}
:::::{tab-item} Linux

::::{tab-set}
:::{tab-item} CUDA

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap[cuda]:x64-linux
```

:::
:::{tab-item} CPU

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap:x64-linux
```

:::
::::

:::::

:::::{tab-item} OSX

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap
```

:::::

:::::{tab-item} Windows

::::{tab-set}
:::{tab-item} CUDA

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap[cuda]:x64-windows
```

:::
:::{tab-item} CPU

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap:x64-windows
```

:::
::::

:::::
::::::
