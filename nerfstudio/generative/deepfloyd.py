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



from diffusers import IFPipeline
from transformers import T5EncoderModel, T5Tokenizer

class DeepFloyd(nn.Module):

    # load in model
    def __init__(self, device: Union[torch.device, str]):
        super().__init__()
        self.device = device

        self.text_encoder = T5EncoderModel.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            subfolder="text_encoder",
            load_in_8bit=True,
            variant="8bit",
            device_map="auto",
        )

        self.pipe = IFPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            text_encoder=self.text_encoder,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)
        
        self.pipe.enable_attention_slicing(1)
        self.pipe.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipe.unet
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)


        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

    def get_text_embeds(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> TensorType[2, "max_length", "embed_dim"]:        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        return torch.cat([prompt_embeds, negative_embeds])

    def sds_loss(
        self,
        text_embeddings, 
        image, 
        guidance_scale, 
        grad_scaler,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            image = F.interpolate(image.half(), (IMG_DIM, IMG_DIM), mode="bilinear", align_corners=False)
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(image)
                image_noisy = self.scheduler.add_noise(image, noise, t)  # type: ignore
                # pred noise
                image_model_input = torch.cat([image_noisy] * 2)
                noise_pred = self.unet(image_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)

            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # w(t), sigma_t^2
            w = 1 - self.alphas[t]

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            if grad_scaler is not None:
                latents = grad_scaler.scale(latents)
            loss = _SDSGradient.apply(latents, grad)

        return loss

    def prompt_to_img(
        self,
        prompts: Union[str, List[str]],
        negative_prompts: Union[str, List[str]] = "",
        generator=None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
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
        # text_embeddings = self.get_text_embeds(prompts, negative_prompts)
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompts, negative_prompt=negative_prompts)
        image = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator).images

        return image

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
        imgs = df.prompt_to_img(prompt, negative, generator, steps)
        imgs[0].save(save_path)
        # mediapy.write_image(str(save_path), imgs[0])

if __name__ == "__main__":
    tyro.cli(generate_image)