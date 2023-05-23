# SDFStudio

[project website](https://autonomousvision.github.io/sdfstudio/)

```{image} imgs/sdfstudio_overview.svg
:width: 800
:align: center
:alt: sdfstudio overview figure
```

## Overview

SDFStudio is built on top of nerfstudio. It implements multiple implicit surface reconstruction methods including:

- UniSurf
- VolSDF
- NeuS
- MonoSDF
- Mono-UniSurf
- Mono-NeuS
- Geo-NeuS
- Geo-UniSurf
- Geo-VolSDF
- NeuS-acc
- NeuS-facto
- NeuralReconW

You can learn more about these methods [here](https://github.com/autonomousvision/sdfstudio/blob/master/docs/sdfstudio-methods.md#Methods)

## Surface models in nerfstudio

We intend to integrate many of the SDFStudio improvements back into the nerfstudio core repository.

Supported methods:

- NeuS
- NeuS-facto

## Citation

If you use these surface based models in your research, you should consider citing the authors of SDFStudio,

```none
@misc{Yu2022SDFStudio,
    author    = {Yu, Zehao and Chen, Anpei and Antic, Bozidar and Peng, Songyou Peng and Bhattacharyya, Apratim
                 and Niemeyer, Michael and Tang, Siyu and Sattler, Torsten and Geiger, Andreas},
    title     = {SDFStudio: A Unified Framework for Surface Reconstruction},
    year      = {2022},
    url       = {https://github.com/autonomousvision/sdfstudio},
}
```
