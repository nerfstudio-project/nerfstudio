# Using existing data

Nerfstudio comes with built-in support for a number of datasets, which can be downloaded with the [`ns-download-data` command][cli]. Each of the built-in datasets comes ready to use with various Nerfstudio methods (e.g. the recommended default Nerfacto), allowing you to get started in the blink of an eye.

[cli]: https://docs.nerf.studio/reference/cli/ns_download_data.html
[paper]: https://arxiv.org/pdf/2302.04264.pdf

## Example

Here are a few examples of downloading different scenes. Please see the [Training Your First NeRF](first_nerf.md) documentation for more details on how to train a model with them.

```bash
# Download all scenes from the Blender dataset, including the "classic" Lego model
ns-download-data blender

# Download the subset of data used in the SIGGRAPH 2023 Nerfstudio paper
ns-download-data nerfstudio --capture-name nerfstudio-dataset

# Download a few room-scale scenes from the EyefulTower dataset at different resolutions
ns-download-data eyefultower --capture-name riverview seating_area apartment --resolution-name jpeg_1k jpeg_2k

# Download the full D-NeRF dataset of dynamic synthetic scenes
ns-download-data dnerf
```

## Dataset Summary

Many of these datasets are used as baselines to evaluate new research in novel view synthesis, such as in the [original Nerfstudio paper][paper]. Scenes from these datasets lie at dramatically different points in the space of images, across axes such as photorealism (synthetic vs real), dynamic range (LDR vs HDR), scale (number of images), and resolution. The tables below describe some of this variation, and hopefully make it easier to pick an appropriate dataset for your research or application.

| Dataset | Synthetic | Real | LDR | HDR | Scenes | Image Count<sup>1</sup> | Image Resolution<sup>2</sup> |
| :-: | :-: | :-: | :-: | :-: | :------: | :-: | :-: |
| [Blender][blender] | ✔️ |  | ✔️ |  | 8 | ➖➕️➖➖ | ➕️➖➖➖➖ |
| [D-NeRF][dnerf] | ✔️ |  | ✔️ |  | 8 | ➕️➖➖➖ | ➕️➖➖➖➖ |
| [EyefulTower][eyefultower] |  | ✔️ | ✔️ | ✔️ | 11 | ➖➕️➕️➕️ | ➖➕️➕️➕️➕️ |
| [Mill 19][mill19] |  | ✔️ | ✔️ |  | 2 | ➖➖➕️➖ | ➖➖➖➕️➖ |
| [NeRF-OSR][nerfosr] |  | ✔️ | ✔️ |  | 9 | ➕➕️➕️➖ | ➖➕️➖➕️➖ |
| [Nerfstudio][nerfstudio] |  | ✔️ | ✔️ |  | 18 | ➕➕️➕️➖ | ➕️➕️➕️➖➖ |
| [PhotoTourism][phototourism] |  | ✔️ | ✔️ |  | 10 | ➖➕️➕️➖ | ➖➕️➖➖➖ |
| [Record3D][record3d] |  | ✔️ | ✔️ |  | 1 | ➖➖➕️➖ | ➕️➖➖➖➖ |
| [SDFStudio][sdfstudio] | ✔️ | ✔️ | ✔️ |  | 45 | ➕️➕️➕️➖ | ➕️➖➕️➖➖ |
| [sitcoms3D][sitcoms3d] |  | ✔️ | ✔️ |  | 10 | ➕️➖➖➖ | ➕️➕️➖➖➖ |

In the tables below, each dataset was placed into a bucket based on the table's chosen property. If a box contains a ✔️, the corresponding dataset will have *at least* one scene falling into the corresponding bucket for that property, though there may be multiple scenes at different points within the range.

<sub>
<b>1:</b> Condensed version of the "Scene Size: Number of RGB Images" table below. <br>
<b>2:</b> Condensed version of the "Scene RGB Resolutions: `max(width, height)`" table below.
</sub>

### Scene Size: Number of RGB Images

| Dataset | < 250 | 250 - 999 | 1000 - 3999 | ≥ 4000 |
| :-: | :-: | :-: | :-: | :-: |
| [Blender][blender] |  | ✔️ |  |  |
| [D-NeRF][dnerf] | ✔️ |  |  |  |
| [EyefulTower][eyefultower] |  | ✔️ | ✔️ | ✔️ |
| [Mill 19][mill19] |  |  | ✔️ |  |
| [NeRF-OSR][nerfosr] | ✔️ | ✔️ | ✔️ |  |
| [Nerfstudio][nerfstudio] | ✔️ | ✔️ | ✔️ |  |
| [PhotoTourism][phototourism] |  | ✔️ | ✔️ |  |
| [Record3D][record3d] |  |  | ✔️ |  |
| [SDFStudio][sdfstudio] | ✔️ | ✔️ | ✔️ |  |
| [sitcoms3D][sitcoms3d] | ✔️ |  |  |

### Scene RGB Resolutions: `max(width, height)`

| Dataset | < 1000 | 1000 - 1999 | 2000 - 3999 | 4000 - 7999 | ≥ 8000 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| [Blender][blender] | ✔️ |  |  |  |  |
| [D-NeRF][dnerf] | ✔️ |  |  |  |  |
| [EyefulTower][eyefultower] |  | ✔️ | ✔️ | ✔️ | ✔️ |
| [Mill 19][mill19] |  |  |  | ✔️ |  |
| [NeRF-OSR][nerfosr] |  | ✔️ |  | ✔️ |  |
| [Nerfstudio][nerfstudio] | ✔️ | ✔️ | ✔️ |  |  |
| [PhotoTourism][phototourism] |  | ✔️ |  |  |  |
| [Record3D][record3d] | ✔️ |  |  |  |  |
| [SDFStudio][sdfstudio] | ✔️ |  | ✔️ |  |  |
| [sitcoms3D][sitcoms3d] | ✔️ | ✔️ |  |  |  |

[blender]: https://github.com/bmild/nerf?tab=readme-ov-file#project-page--video--paper--data
[dnerf]: https://github.com/albertpumarola/D-NeRF?tab=readme-ov-file#download-dataset
[eyefultower]: https://github.com/facebookresearch/EyefulTower
[mill19]: https://github.com/cmusatyalab/mega-nerf?tab=readme-ov-file#mill-19
[nerfosr]: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/
[nerfstudio]: https://github.com/nerfstudio-project/nerfstudio
[phototourism]: https://www.cs.ubc.ca/~kmyi/imw2020/data.html
[record3d]: https://record3d.app/
[sdfstudio]: https://github.com/autonomousvision/sdfstudio/blob/master/docs/sdfstudio-data.md#Existing-dataset
[sitcoms3d]: https://github.com/ethanweber/sitcoms3D/blob/master/METADATA.md
