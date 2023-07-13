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

## Example Results

The following videos are renders of NeRFs generated from Generfacto. Each model was trained 30k steps, which took around 1 hour with DeepFloyd
and around 4 hours with Stable Diffusion.

"a high quality photo of a ripe pineapple" (Stable Diffusion)

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://user-images.githubusercontent.com/19509183/246646597-407ff7c8-7106-4835-acf3-c2f8188bbd1d.mp4" type="video/mp4">
</video>

"a high quality zoomed out photo of a palm tree" (DeepFloyd)

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://user-images.githubusercontent.com/19509183/246646594-05ffebce-a3d6-43af-9f11-e04ce2ce3237.mp4" type="video/mp4">
</video>

"a high quality zoomed out photo of a light grey baby shark" (DeepFloyd)

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://user-images.githubusercontent.com/19509183/246646599-b1f5b7c5-dd96-48b4-8db0-960632e7798b.mp4" type="video/mp4">
</video>
