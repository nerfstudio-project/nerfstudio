# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stable Diffusion helpers"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import appdirs
import mediapy
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from rich.console import Console
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.cuda.amp.grad_scaler import GradScaler
from torchtyping import TensorType

CONSOLE = Console(width=120)

try:
    from diffusers import PNDMScheduler, StableDiffusionPipeline
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215
SD_IDENTIFIERS = {
    "1-5": "runwayml/stable-diffusion-v1-5",
    "2-0": "stabilityai/stable-diffusion-2-base",
    "2-1": "stabilityai/stable-diffusion-2-1-base",
}


@dataclass
class UNet2DConditionOutput:
    """Class to hold traced model"""

    sample: torch.FloatTensor


class _SDSGradient(torch.autograd.Function):  # pylint: disable=abstract-method
    """Custom gradient function for SDS loss. Since it is already computed, we can just return it."""

    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):  # pylint: disable=arguments-differ
        del input_tensor
        ctx.save_for_backward(gt_grad)
        # Return magniture of gradient, not the actual loss.
        return torch.mean(gt_grad**2) ** 0.5

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):  # pylint: disable=arguments-differ
        del grad
        (gt_grad,) = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


class StableDiffusion(nn.Module):
    """Stable Diffusion implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, version="1-5") -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps

        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        self.scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=self.num_train_timesteps,
        )
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        sd_id = SD_IDENTIFIERS[version]
        pipe = StableDiffusionPipeline.from_pretrained(sd_id, torch_dtype=torch.float16)
        assert pipe is not None
        pipe = pipe.to(self.device)

        pipe.enable_attention_slicing()

        # use jitted unet
        filename_sd_id = sd_id.split("/")[-1]
        unet_traced_filename = Path(appdirs.user_data_dir("nerfstudio")) / f"{filename_sd_id}_unet_traced.pt"
        if unet_traced_filename.exists():
            CONSOLE.print("Loading traced UNet.")
            unet_traced = torch.jit.load(unet_traced_filename)

            class TracedUNet(torch.nn.Module):
                """Jitted version of UNet"""

                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):  # pylint: disable=no-self-use
                    """Forward pass"""
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)

            self.unet = TracedUNet()
            del pipe.unet
        else:
            CONSOLE.print("[bold yellow] Warning: Loading UNet without JIT acceleration.")
            CONSOLE.print(f"Run [yellow]ns-trace-sd --sd-version {version} [/yellow] for a speedup!")
            self.unet = pipe.unet
            self.unet.to(memory_format=torch.channels_last)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.auto_encoder = pipe.vae

        CONSOLE.print("Stable Diffusion loaded!")

    def get_text_embeds(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> TensorType[2, "max_length", "embed_dim"]:
        """Get text embeddings for prompt and negative prompt
        Args:
            prompt: Prompt text
            negative_prompt: Negative prompt text
        Returns:
            Text embeddings
        """

        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def sds_loss(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        guidance_scale: float = 100.0,
        grad_scaler: Optional[GradScaler] = None,
    ) -> torch.Tensor:
        """Score Distilation Sampling loss proposed in DreamFusion paper (https://dreamfusion3d.github.io/)
        Args:
            text_embeddings: Text embeddings
            image: Rendered image
            guidance_scale: How much to weigh the guidance
            grad_scaler: Grad scaler
        Returns:
            The loss
        """
        image = F.interpolate(image, (IMG_DIM, IMG_DIM), mode="bilinear")
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        latents = self.imgs_to_latent(image)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if grad_scaler is not None:
            latents = grad_scaler.scale(latents)
        loss = _SDSGradient.apply(latents, grad)

        return loss

    def produce_latents(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        height: int = IMG_DIM,
        width: int = IMG_DIM,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: Optional[TensorType["BS", 4, "H", "W"]] = None,
    ) -> TensorType["BS", 4, "H", "W"]:
        """Produce latents for a given text embedding
        Args:
            text_embeddings: Text embeddings
            height: Height of the image
            width: Width of the image
            num_inference_steps: Number of inference steps
            guidance_scale: How much to weigh the guidance
            latents: Latents to start with
        Returns:
            Latents
        """

        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device
            )

        self.scheduler.set_timesteps(num_inference_steps)  # type: ignore

        with torch.autocast("cuda"):
            for t in self.scheduler.timesteps:  # type: ignore
                assert latents is not None
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t.to(self.device), encoder_hidden_states=text_embeddings
                    ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]  # type: ignore
        return latents

    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prompt_to_img(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents=None,
    ) -> np.ndarray:
        """Generate an images from a prompts.
        Args:
            prompts: The prompt to generate an image from.
            negative_prompts: The negative prompt to generate an image from.
            num_inference_steps: The number of inference steps to perform.
            guidance_scale: The scale of the guidance.
            latents: The latents to start from, defaults to random.
        Returns:
            The generated image.
        """

        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts
        text_embeddings = self.get_text_embeds(prompts, negative_prompts)
        latents = self.produce_latents(
            text_embeddings,
            height=IMG_DIM,
            width=IMG_DIM,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, resolution, resolution]

        diffused_img = self.latents_to_img(latents.half())
        diffused_img = diffused_img.detach().cpu().permute(0, 2, 3, 1).numpy()
        diffused_img = (diffused_img * 255).round().astype("uint8")

        return diffused_img

    def forward(
        self, prompts, negative_prompts="", num_inference_steps=50, guidance_scale=7.5, latents=None
    ) -> np.ndarray:
        """Generate an image from a prompt.
        Args:
            prompts: The prompt to generate an image from.
            negative_prompts: The negative prompt to generate an image from.
            num_inference_steps: The number of inference steps to perform.
            guidance_scale: The scale of the guidance.
            latents: The latents to start from, defaults to random.
        Returns:
            The generated image.
        """
        return self.prompt_to_img(prompts, negative_prompts, num_inference_steps, guidance_scale, latents)


def generate_image(
    prompt: str, negative: str = "", seed: int = 0, steps: int = 50, save_path: Path = Path("test_sd.png")
):
    """Generate an image from a prompt using Stable Diffusion.
    Args:
        prompt: The prompt to use.
        negative: The negative prompt to use.
        seed: The random seed to use.
        steps: The number of steps to use.
        save_path: The path to save the image to.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cuda_device = torch.device("cuda")
    with torch.no_grad():
        sd = StableDiffusion(cuda_device)
        imgs = sd.prompt_to_img(prompt, negative, steps)
        mediapy.write_image(str(save_path), imgs[0])


if __name__ == "__main__":
    tyro.cli(generate_image)
