# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import gc
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import tyro

from jaxtyping import Float
from PIL import Image
from torch import Generator, Tensor, nn
from torch.cuda.amp.grad_scaler import GradScaler


from nerfstudio.generative.utils import CatchMissingPackages

try:
    from diffusers import IFPipeline as IFOrig
    from diffusers.pipelines.deepfloyd_if import IFPipelineOutput as IFOutputOrig

    from diffusers import IFPipeline, DiffusionPipeline
    from diffusers.pipelines.deepfloyd_if import IFPipelineOutput
    from transformers import T5EncoderModel

except ImportError:
    IFPipeline = IFPipelineOutput = T5EncoderModel = CatchMissingPackages()

IMG_DIM = 64


class DeepFloyd(nn.Module):
    """DeepFloyd diffusion model
    Args:
        device: device to use
    """

    def __init__(self, device: Union[torch.device, str]):
        super().__init__()
        self.device = device

        self.text_encoder = T5EncoderModel.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            subfolder="text_encoder",
            load_in_8bit=True,
            variant="8bit",
            device_map="auto",
        )

        self.pipe = IFPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            text_encoder=self.text_encoder,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        assert isinstance(self.pipe, DiffusionPipeline)
        self.pipe = self.pipe.to(self.device)

        self.pipe.enable_attention_slicing(1)

        self.unet = self.pipe.unet
        self.unet.to(memory_format=torch.channels_last)  # type: ignore
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config["num_train_timesteps"]
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

    def delete_text_encoder(self):
        """Delete text encoder from pipeline. T5 text encoder uses a lot of memory."""
        del self.text_encoder
        del self.pipe
        gc.collect()
        torch.cuda.empty_cache()

        self.pipe = IFPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0",
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        assert isinstance(self.pipe, DiffusionPipeline)
        self.pipe = self.pipe.to(self.device)

        self.pipe.enable_attention_slicing(1)

        self.unet = self.pipe.unet
        self.unet.to(memory_format=torch.channels_last)  # type: ignore

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

    def get_text_embeds(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Float[Tensor, "2 max_length embed_dim"]:
        """Get text embeddings for prompt and negative prompt
        Args:
            prompt: Prompt text
            negative_prompt: Negative prompt text
        Returns:
            Text embeddings
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        assert isinstance(self.pipe, DiffusionPipeline)
        with torch.no_grad():
            prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        assert isinstance(negative_embeds, Tensor)
        assert isinstance(prompt_embeds, Tensor)
        return torch.cat([negative_embeds, prompt_embeds])

    def sds_loss(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
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
        image = F.interpolate(image.half(), (IMG_DIM, IMG_DIM), mode="bilinear", align_corners=False)
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(image)
            image_noisy = self.scheduler.add_noise(image, noise, t)  # type: ignore
            # pred noise
            image_model_input = torch.cat((image_noisy,) * 2)
            noise_pred = self.unet(image_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)

        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (image - grad).detach()
        loss = 0.5 * F.mse_loss(image, target, reduction="sum") / image.shape[0]

        return loss

    def prompt_to_image(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = "",
        generator: Optional[Generator] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """Generate an image from a prompt.
        Args:
            prompts: The prompt to generate an image from.
            negative_prompts: The negative prompt to generate an image from.
            generator: Random seed.
            num_inference_steps: The number of inference steps to perform.
            guidance_scale: The scale of the guidance.
            latents: The latents to start from, defaults to random.
        Returns:
            The generated image.
        """

        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts
        assert isinstance(self.pipe, DiffusionPipeline)
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompts, negative_prompt=negative_prompts)

        assert isinstance(self.pipe, IFOrig)
        model_output = self.pipe(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator
        )
        assert isinstance(model_output, IFOutputOrig)
        output_image = model_output.images[0]

        return output_image


def generate_image(
    prompt: str, negative: str = "", seed: int = 0, steps: int = 50, save_path: Path = Path("test_deepfloyd.png")
):
    """Generate an image from a prompt using DeepFloyd IF.
    Args:
        prompt: The prompt to use.
        negative: The negative prompt to use.
        seed: The random seed to use.
        steps: The number of steps to use.
        save_path: The path to save the image to.
    """
    generator = torch.manual_seed(seed)
    cuda_device = torch.device("cuda")
    with torch.no_grad():
        df = DeepFloyd(cuda_device)
        img = df.prompt_to_image(prompt, negative, generator, steps)
        img.save(save_path)


if __name__ == "__main__":
    tyro.cli(generate_image)
