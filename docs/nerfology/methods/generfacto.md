# Generfacto

<h4>Generate 3D models from text</h4>

**Our model that combines generative 3D and NeRF advancements**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install -e .[gen]
```

Two options for text to image diffusion models are provided: Stable Diffusion and DeepFloyd IF.
We use Deepfloyd IF by default because it trains faster and produces better results, however this model
requires signing a license agreement on the model card of DeepFloyd IF, and logging in with

```bash
huggingface-cli login
```
.
More instructions can be found on the Huggingface website [here](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).

If you do not want to sign the license agreement, you can specify using the Stable Diffusion model in the command line.

## Running Generfacto

Once installed, run:

```bash
ns-train generfacto
```

Specify the diffusion model to use with the --diffusion_model flag:

```bash
ns-train generfacto --pipeline.model.diffusion_model ["stablediffusion", "deepfloyd"]
```

Specify different prompts with the --prompt flag:

```bash
ns-train generfacto --prompt "a high quality photo of a pineapple"
```
