# Generfacto

<h4>Generate 3D models from text</h4>

**Our model that combines generative 3D and NeRF advancements**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install -e .[gen]
```

Two options for text to image diffusion models are provided: Stable Diffusion and DeepFloyd IF.
We use Deepfloyd IF by default because it trains faster and produces better results. However this model
requires users to sign a license agreement on the model card of DeepFloyd IF, and login with

```bash
huggingface-cli login
```

More instructions can be found on the Hugging Face website [here](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).

If you do not want to sign the license agreement, you can use the Stable Diffusion model instead by specifying in the command line.

## Running Generfacto

Once installed, run:

```bash
ns-train generfacto
```

The first time you run this method, the diffusion model weights will be downloaded and cached
from Hugging Face, which may take a couple minutes.

Specify which diffusion model to use with the diffusion_model flag:

```bash
ns-train generfacto --pipeline.model.diffusion_model ["stablediffusion", "deepfloyd"]
```

Train using different prompts with the prompt flag:

```bash
ns-train generfacto --prompt "a high quality photo of a pineapple"
```
