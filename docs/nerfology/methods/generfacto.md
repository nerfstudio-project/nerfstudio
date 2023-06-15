# Generfacto

<h4>Generate 3D models from text</h4>

**Our model that combines generative 3D with our latest NeRF methods**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install -e .[gen]
```

Two options for text to image diffusion models are provided: Stable Diffusion and DeepFloyd IF.
We use Deepfloyd IF by default because it trains faster and produces better results. Using this model requires users to sign a license agreement for the model card of DeepFloyd IF, which can be found [here](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). Once the licensed is signed, log into Huggingface locally by running the following command:

```bash
huggingface-cli login
```

If you do not want to sign the license agreement, you can use the Stable Diffusion model (instructions below).

## Running Generfacto

Once installed, run:

```bash
ns-train generfacto --prompt "a high quality photo of a pineapple"
```

The first time you run this method, the diffusion model weights will be downloaded and cached
from Hugging Face, which may take a couple minutes.

Specify which diffusion model to use with the diffusion_model flag:

```bash
ns-train generfacto --pipeline.model.diffusion_model ["stablediffusion", "deepfloyd"]
```
