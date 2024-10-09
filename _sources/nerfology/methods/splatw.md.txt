# Splatfacto in the Wild

This is the implementation of [Splatfacto in the Wild: A Nerfstudio Implementation of Gaussian Splatting for Unconstrained Photo Collections](https://kevinxu02.github.io/splatfactow). The official code can be found [here](https://github.com/KevinXu02/splatfacto-w).

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://github.com/KevinXu02/splatfactow/blob/main/static/videos/interp_fountain2.mp4" type="video/mp4">

</video>

## Installation
This repository follows the nerfstudio method [template](https://github.com/nerfstudio-project/nerfstudio-method-template/tree/main)

### 1. Install Nerfstudio dependencies
Please follow the Nerfstudio [installation guide](https://docs.nerf.studio/quickstart/installation.html)  to create an environment and install dependencies.

### 2. Install the repository
Run the following commands:
`pip install git+https://github.com/KevinXu02/splatfacto-w`

Then, run `ns-install-cli`.

### 3. Check installation
Run `ns-train splatfacto-w --help`. You should see the help message for the splatfacto-w method.

## Downloading data
You can download the phototourism dataset from running.
```
ns-download-data phototourism --capture-name <capture_name>
```

## Running Splafacto-w
To train with it, download the train/test tsv file from the bottom of [nerf-w](https://nerf-w.github.io/) and put it under the data folder (or copy them from `./splatfacto-w/dataset_split`). For instance, for Brandenburg Gate the path would be `your-data-folder/brandenburg_gate/brandenburg.tsv`. You should have the following structure in your data folder:
```
|---brandenburg_gate
|   |---dense
|   |   |---images
|   |   |---sparse
|   |   |---stereo
|   |---brandenburg.tsv
```

Then, run the command:
```
ns-train splatfacto-w --data [PATH]
```

If you want to train datasets without nerf-w's train/test split or your own datasets, we provided a light-weight version of the method for general cases. To train with it, you can run the following command
```
ns-train splatfacto-w-light --data [PATH] [dataparser]
```
For phototourism, the `dataparser` should be `colmap` and you need to change the colmap path through the CLI because phototourism dataparser does not load 3D points.