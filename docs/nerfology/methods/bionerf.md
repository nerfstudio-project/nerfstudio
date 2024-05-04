# BioNeRF

<h4> Biologically Plausible Neural Radiance Fields for View Synthesis</h4>

```{button-link} https://arxiv.org/pdf/2402.07310
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/Leandropassosjr/BioNeRF
:color: primary
:outline:
Code
```


![](https://github.com/Leandropassosjr/BioNeRF/blob/dev/images/gifs.gif)

**Biologically Plausible Neural Radiance Fields**

## Installation

Install nerfstudio dependencies. 

## Running BioNeRF

Details for running BioNeRF (built with Nerfstudio!) can be found [here](https://github.com/cvachha/instruct-gs2gs). Once installed, run:

```bash
ns-train bionerf --help
```

## Method

### Overview

[BioNeRF](https://arxiv.org/pdf/2402.07310.pdf) (Biologically Plausible Neural Radiance Fields) extends [NeRF](http://www.matthewtancik.com/nerf) by implementing a cognitive-inspired mechanism that fuses inputs from multiple sources into a memory-like structure, thus improving the storing capacity and extracting more intrinsic and correlated information. BioNeRF also mimics a behavior observed in pyramidal cells concerning contextual information, in which the memory is provided as the context and combined with the inputs of two subsequent blocks of dense layers, one responsible for producing the volumetric densities and the other the colors used to render the novel view. 

## Pipeline

<img src='https://github.com/Leandropassosjr/BioNeRF/blob/dev/images/nerf_ship.jpeg'/>
<img src='https://github.com/Leandropassosjr/BioNeRF/blob/dev/images/BioNeRF.jpeg'/>

## Results

For results, view the [paper page](https://arxiv.org/pdf/2402.07310)!
