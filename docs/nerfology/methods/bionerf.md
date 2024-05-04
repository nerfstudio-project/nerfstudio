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

Once the nerfstudio dependencies are installed, run:

```bash
ns-train bionerf --help
```

## Method

### Overview

[BioNeRF](https://arxiv.org/pdf/2402.07310.pdf) (Biologically Plausible Neural Radiance Fields) extends [NeRF](http://www.matthewtancik.com/nerf) by implementing a cognitive-inspired mechanism that fuses inputs from multiple sources into a memory-like structure, thus improving the storing capacity and extracting more intrinsic and correlated information. BioNeRF also mimics a behavior observed in pyramidal cells concerning contextual information, in which the memory is provided as the context and combined with the inputs of two subsequent blocks of dense layers, one responsible for producing the volumetric densities and the other the colors used to render the novel view. 

## Pipeline
<img src='https://github.com/Leandropassosjr/BioNeRF/blob/dev/images/BioNeRF.png'/>

Here is an overview pipeline for BioNeRF, we will walk through each component in this guide.

### Positional Feature Extraction
The first step consists of feeding two neural models simultaneously, namely $M_{\Delta}$ and $M_c$, with the camera positional information. The output of these models encodes the positional information from the input image. Although the input is the same, the neural models do not share weights and follow a different flow in the next steps.

### Cognitive Filtering
This step performs a series of operations, called \emph{filters}, that work on the embeddings coming from the previous step. There are four filters this step derives: density, color, memory, and modulation.

### Memory Updating
Updating the memory requires the implementation of a mechanism capable of obliterating trivial information, which is performed using the memory filter (Step 3.1 in Figure~\ref{f.bionerf}). Fist, one needs to compute a signal modulation $\bm{\mu}$, for further introducing new experiences in the memory $\bm{\Psi}$ through the modulating variable $\bm{\mu}$ using a $\textit{tanh}$ function (Step 3.2 in the figure.

### Contextual Inference
This step is responsible for adding contextual information to BioNeRF. Two new embeddings are generate, i.e., $\bm{h}^{\prime}_\Delta$ and $\bm{h}^{\prime}_c$ based on density and color filters, respectively (Step 4 in the figure), which further feed two neural models, i.e., $M^{\prime}_\Delta$ and $M^{\prime}$. Subsequently, $M^{\prime}_\Delta$ outputs the volume density, while color information is predicted by $M^{\prime}_c$, further used to compute the final predicted pixel information and the loss function.

## Results

For results, view the [paper page](https://arxiv.org/pdf/2402.07310)!
