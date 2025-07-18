# UnMix-NeRF

<h4>Spectral Unmixing Meets Neural Radiance Fields</h4>

```{button-link} https://www.arxiv.org/pdf/2506.21884
:color: primary
:outline:
Paper
```

```{button-link} https://www.factral.co/UnMix-NeRF/
:color: primary
:outline:
Project Page
```

<div style="text-align: center;">

TL;DR _We propose UnMix-NeRF, the first method integrating spectral unmixing into NeRF, enabling hyperspectral view synthesis, accurate unsupervised material segmentation, and intuitive material-based scene editing, significantly outperforming existing methods._

_ICCV 2025_

<img src="https://www.factral.co/UnMix-NeRF/assets/intro.jpg" alt="UnMix-NeRF Overview" style="width:50% !important;"/><br>

</div>

## Visual Results

<h3 style="text-align: center;">Hotdog Scene</h3>
<table style="width:100%; border: none;">
  <tr style="border: none;">
    <td style="text-align: center; border: none; padding: 5px;">
      <video autoplay="autoplay" loop="loop" muted="muted" playsinline="playsinline" style="width:100%">
        <source src="https://www.factral.co/UnMix-NeRF/assets/hotdog/rgb/rgb_hotdog.mp4" type="video/mp4">
      </video>
      <p>RGB</p>
    </td>
    <td style="text-align: center; border: none; padding: 5px;">
      <video autoplay="autoplay" loop="loop" muted="muted" playsinline="playsinline" style="width:100%">
        <source src="https://www.factral.co/UnMix-NeRF/assets/hotdog/seg/seg.mp4" type="video/mp4">
      </video>
      <p>Unsupervised Material Segmentation</p>
    </td>
    <td style="text-align: center; border: none; padding: 5px;">
      <video autoplay="autoplay" loop="loop" muted="muted" playsinline="playsinline" style="width:100%">
        <source src="https://www.factral.co/UnMix-NeRF/assets/hotdog/edit1/edit1.mp4" type="video/mp4">
      </video>
      <p>Scene Editing</p>
    </td>
  </tr>
</table>

<h3 style="text-align: center;">Ajar Scene</h3>
<table style="width:100%; border: none;">
  <tr style="border: none;">
    <td style="text-align: center; border: none; padding: 5px;">
      <video autoplay="autoplay" loop="loop" muted="muted" playsinline="playsinline" style="width:100%">
        <source src="https://www.factral.co/UnMix-NeRF/assets/ajar/seg_1_segment0.mp4" type="video/mp4">
      </video>
      <p>RGB</p>
    </td>
    <td style="text-align: center; border: none; padding: 5px;">
      <video autoplay="autoplay" loop="loop" muted="muted" playsinline="playsinline" style="width:100%">
        <source src="https://www.factral.co/UnMix-NeRF/assets/ajar/seg_1_segment2.mp4" type="video/mp4">
      </video>
      <p>Unsupervised Material Segmentation</p>
    </td>
    <td style="text-align: center; border: none; padding: 5px;">
      <video autoplay="autoplay" loop="loop" muted="muted" playsinline="playsinline" style="width:100%">
        <source src="https://www.factral.co/UnMix-NeRF/assets/ajar/seg_1_segment1.mp4" type="video/mp4">
      </video>
      <p>PCA Visualization</p>
    </td>
  </tr>
</table>

## Installation

Install nerfstudio dependencies following the [installation guide](https://docs.nerf.studio/quickstart/installation.html).

Then install UnMix-NeRF:

```bash
git clone https://github.com/Factral/UnMix-NeRF
cd UnMix-NeRF
pip install -r requirements.txt
pip install .
```

## Running UnMix-NeRF

Basic training command:

```bash
ns-train unmixnerf \
  --data <path_to_data> \
  --pipeline.num_classes <number_of_materials> \
  --pipeline.model.spectral_loss_weight 5.0 \
  --pipeline.model.temperature 0.4 \
  --experiment-name my_experiment
```

## Method

### Overview

[UnMix-NeRF](https://www.arxiv.org/pdf/2506.21884) Neural Radiance Field (NeRF)-based segmentation methods focus on object semantics and rely solely on RGB data, lacking intrinsic material properties. This limitation restricts accurate material perception, which is crucial for robotics, augmented reality, simulation, and other applications. We introduce UnMix-NeRF, a framework that integrates spectral unmixing into NeRF, enabling joint hyperspectral novel view synthesis and unsupervised material segmentation.

Our method models spectral reflectance via diffuse and specular components, where a learned dictionary of global endmembers represents pure material signatures, and per-point abundances capture their distribution. For material segmentation, we use spectral signature predictions along learned endmembers, allowing unsupervised material clustering. Additionally, UnMix-NeRF enables scene editing by modifying learned endmember dictionaries for flexible material-based appearance manipulation. Extensive experiments validate our approach, demonstrating superior spectral reconstruction and material segmentation to existing methods.

### Pipeline

![UnMix-NeRF Pipeline](https://www.factral.co/UnMix-NeRF/assets/framework.jpg)<br>

## Data Format

UnMix-NeRF extends standard nerfstudio data conventions to support hyperspectral data:

### Required Structure

```
data/
├── transforms.json          # Camera poses (standard)
├── images/                  # RGB images (standard)
│   ├── frame_00001.jpg
│   └── ...
├── hyperspectral/          # Hyperspectral data (NEW)
│   ├── frame_00001.npy     # Shape: (H, W, B)
│   └── ...
└── segmentation/           # Ground truth (optional)
    ├── frame_00001.png
    └── ...
```

### Hyperspectral Data

- **Format**: `.npy` files with dimensions `(H, W, B)`
- **Values**: Normalized between 0 and 1
- **Bands**: Number of spectral channels (B)

Update your `transforms.json`:

```json
{
  "frames": [
    {
      "file_path": "./images/frame_00001.jpg",
      "hyperspectral_file_path": "./hyperspectral/frame_00001.npy",
      "seg_file_path": "./segmentation/frame_00001.png",
      "transform_matrix": [...]
    }
  ]
}
```

## Key Parameters

| Parameter                               | Description                       | Default |
| --------------------------------------- | --------------------------------- | ------- |
| `--pipeline.num_classes`                | Number of material endmembers     | 6       |
| `--pipeline.model.spectral_loss_weight` | Weight for spectral loss          | 5.0     |
| `--pipeline.model.temperature`          | Temperature for abundance softmax | 0.4     |
| `--pipeline.model.load_vca`             | Initialize with VCA endmembers    | False   |
| `--pipeline.model.pred_specular`        | Enable specular component         | True    |

## Citation

```bibtex
@inproceedings{perez2025unmix,
  title={UnMix-NeRF: Spectral Unmixing Meets Neural Radiance Fields},
  author={Perez, Fabian and Rojas, Sara and Hinojosa, Carlos and Rueda-Chac{\'o}n, Hoover and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
