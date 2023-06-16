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

"""Stable Diffusion helpers"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, cast

import mediapy
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torch.autograd import Variable
from torch.cuda.amp.grad_scaler import GradScaler
from torchviz import make_dot

from nerfstudio.generative.utils import _SDSGradient
from nerfstudio.utils.rich_utils import CONSOLE

try:
    from diffusers import (
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        PNDMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.models.embeddings import TimestepEmbedding
    from transformers import AutoTokenizer, CLIPTextModel, logging

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
    "2-1": "stabilityai/stable-diffusion-2-1",
    "2-1-base": "stabilityai/stable-diffusion-2-1-base",
}


@dataclass
class UNet2DConditionOutput:
    """Class to hold traced model"""

    sample: torch.FloatTensor


class StableDiffusion(nn.Module):
    """Stable Diffusion implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(
        self, device: Union[torch.device, str], num_train_timesteps: int = 1000, version="2-1", use_sds=True
    ) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps

        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        self.weights_dtype = torch.float16

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline

        # self.scheduler = PNDMScheduler(
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     num_train_timesteps=self.num_train_timesteps,
        # )

        sd_id = SD_IDENTIFIERS[version]
        pipe = StableDiffusionPipeline.from_pretrained(sd_id, torch_dtype=torch.float16).to(self.device)
        assert isinstance(pipe, StableDiffusionPipeline)

        pipe_lora = StableDiffusionPipeline.from_pretrained(sd_id, torch_dtype=torch.float16).to(self.device)
        del pipe_lora.vae
        pipe_lora.vae = pipe.vae

        # pipe_lora = pipe

        pipe.enable_attention_slicing(1)

        # self.scheduler = DDPMScheduler.from_pretrained(
        #     sd_id,
        #     subfolder="scheduler",
        #     # torch_dtype=self.weights_dtype,
        # )
        # pipe.scheduler = self.scheduler
        # pipe_lora = pipe

        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora)

        # self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        # self.unet = copy.deepcopy(pipe.unet.eval())
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe_lora.unet.to(memory_format=torch.channels_last)
        # pipe_lora = StableDiffusionPipeline.from_pretrained(sd_id, torch_dtype=torch.float16).to(self.device)
        # self.unet_lora = pipe_lora.unet
        # self.unet_lora = pipe.unet

        # OLD
        # self.tokenizer = pipe.tokenizer
        # self.text_encoder = pipe.text_encoder

        self.tokenizer = AutoTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder").to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        # self.auto_encoder = pipe.vae

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.unet_lora.parameters():
            p.requires_grad_(False)
        # for p in self.text_encoder.parameters():
        #     p.requires_grad_(False)

        # # from https://github.com/threestudio-project/threestudio/commit/12d8f1c52f6ef4379db4e23261b274d36e12a531
        self.camera_embedding = TimestepEmbedding(16, 1280)
        self.unet_lora.class_embedding = self.camera_embedding

        lora_attn_procs = {}
        for name in self.unet_lora.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else self.unet_lora.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet_lora.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet_lora.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet_lora.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        self.unet_lora.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors)
        self.lora_layers._load_state_dict_pre_hooks.clear()
        self.lora_layers._state_dict_hooks.clear()

        self.scheduler = DDPMScheduler.from_pretrained(
            sd_id,
            subfolder="scheduler",
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            sd_id,
            subfolder="scheduler",
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(self.pipe_lora.scheduler.config)

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(self.device)

        CONSOLE.print("Stable Diffusion loaded!")

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.submodules.pipe.unet

    @property
    def unet_lora(self):
        return self.submodules.pipe_lora.unet

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

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

    def lora_loss(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        latents: Float[Tensor, "BS 4 64 64"],
        guidance_scale: float = 100.0,
        grad_scaler: Optional[GradScaler] = None,
    ) -> torch.Tensor:
        B = latents.shape[0]
        t = torch.randint(0, int(self.scheduler.config.num_train_timesteps), [B], dtype=torch.long, device=self.device)
        latents = latents.detach().repeat(1, 1, 1, 1)

        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler_lora.add_noise(latents, noise, t)  # type: ignore
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(f"Unknown prediction type {self.scheduler_lora.config.prediction_type}")

        _, text_embeddings = text_embeddings.chunk(2)
        # camera_condition = torch.zeros((B, 4, 4), device=self.device)
        camera_condition = torch.eye(4, device=self.device)[None, ...]

        noise_pred = self.forward_unet(
            self.unet_lora,
            latents_noisy,
            t,
            encoder_hidden_states=text_embeddings,
            class_labels=camera_condition.view(B, -1),
            cross_attention_kwargs={"scale": 1.0},
        )

        print("NOISE_PRED REQUIRES GRAD:", noise_pred.requires_grad, noise_pred.grad_fn)

        # q = [noise_pred.grad_fn]
        # while len(q) > 0 and q[0] is not None:
        #     curr = q.pop(0)
        #     print(curr.name())
        #     for next_fn in curr.next_functions:
        #         next_fn = next_fn[0]
        #         if next_fn is not None:
        #             q.append(next_fn)
        #         break

        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs=None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    def vsd_loss(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        latents: Float[Tensor, "BS 4 64 64"],
        guidance_scale: float = 1.0,
        grad_scaler: Optional[GradScaler] = None,
    ) -> torch.Tensor:
        B = latents.shape[0]

        # tmp = self.lora_loss(text_embeddings, latents, guidance_scale=guidance_scale)
        # print("LORA LOSS REQUIRES GRAD 1:", tmp.requires_grad)

        print("TEXT EMBEDDINGS SHAPE:", text_embeddings.shape)

        with torch.no_grad():
            # add noise
            t = torch.randint(self.min_step, self.max_step + 1, [B], dtype=torch.long, device=self.device)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0}
                # unet = copy.deepcopy(self.unet)
                noise_pred_pretrain = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            camera_condition = torch.eye(4, device=self.device)[None, ...]

            noise_pred_lora = self.forward_unet(
                self.unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                class_labels=torch.cat(
                    [
                        camera_condition.view(B, -1),
                        torch.zeros_like(camera_condition.view(B, -1)),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            )

        # exit()

        noise_pred_pretrain_uncond, noise_pred_pretrain_text = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        noise_pred_lora_uncond, noise_pred_lora_text = noise_pred_lora.chunk(2)

        # TODO: move to config
        guidance_scale_lora = 1.0
        noise_pred_lora = noise_pred_lora_uncond + guidance_scale_lora * (noise_pred_lora_text - noise_pred_lora_uncond)

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_lora)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / B

        return loss_vsd

        # if grad_scaler is not None:
        #     latents = grad_scaler.scale(latents)
        # loss = cast(Tensor, _SDSGradient.apply(latents, grad))

        # return loss

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
        image = F.interpolate(image, (IMG_DIM, IMG_DIM), mode="bilinear")
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        latents = self.imgs_to_latent(image)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            # pred noise
            latent_model_input = torch.cat((latents_noisy,) * 2)
            noise_pred = self.forward_unet(self.unet, latent_model_input, t, encoder_hidden_states=text_embeddings)

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / image.shape[0]

        return loss

        # if grad_scaler is not None:
        #     latents = grad_scaler.scale(latents)
        # loss = cast(Tensor, _SDSGradient.apply(latents, grad))

        # return loss

    def produce_latents(
        self,
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        height: int = IMG_DIM,
        width: int = IMG_DIM,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        latents: Optional[Float[Tensor, "BS 4 H W"]] = None,
    ) -> Float[Tensor, "BS 4 H W"]:
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
                (text_embeddings.shape[0] // 2, self.unet.config.in_channels, height // 8, width // 8),
                device=self.device,
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

    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
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

    @torch.cuda.amp.autocast(enabled=False)
    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        input_dtype = imgs.dtype
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents.to(input_dtype)

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

    def vsd_forward(
        self,
        text_embeddings_vd: Float[Tensor, "N max_length embed_dim"],
        text_embeddings: Float[Tensor, "N max_length embed_dim"],
        image: Float[Tensor, "BS 3 H W"],
        guidance_scale: float = 1.0,
        grad_scaler: Optional[GradScaler] = None,
    ) -> torch.Tensor:
        B = image.shape[0]
        if image.shape[1] != 512:
            image = F.interpolate(image, (IMG_DIM, IMG_DIM), mode="bilinear")
        latents = self.imgs_to_latent(image)

        loss_vsd = self.vsd_loss(text_embeddings_vd, latents, guidance_scale=guidance_scale)
        loss_lora = self.lora_loss(text_embeddings, latents, guidance_scale=1.0)

        print("LOSSES:", loss_vsd.item(), loss_lora.item())
        # exit()
        return loss_vsd + loss_lora


def vsd_generate_image(prompt: str, negative: str = "", seed: int = 0, save_path: Path = Path("test_sd.png")):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    cuda_device = torch.device("cuda")

    sd = StableDiffusion(cuda_device, use_sds=False).to(cuda_device).float()
    im = torch.rand((1, 3, 512, 512)).float().to(cuda_device)
    im.requires_grad_(True)
    im.retain_grad()
    # im = im.half()  # .to(cuda_device)
    text_embedding = sd.get_text_embeds(prompt, negative)
    text_embedding_vd = text_embedding

    # WORKS
    # text_embedding = torch.load("text_embeddings.pt").to(cuda_device)
    # one, two = text_embedding.chunk(2)
    # text_embedding = torch.cat([two, one], dim=0)
    # text_embedding_vd = torch.load("text_embeddings_vd.pt").to(cuda_device)
    # one, two = text_embedding_vd.chunk(2)
    # text_embedding_vd = torch.cat([two, one], dim=0)

    # AdamOptimizerConfig(lr=5e-3, eps=1e-15)
    # optimizer = torch.optim.SGD([im] + list(sd.parameters()), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam([im] + list(sd.parameters()), lr=0.01, eps=1e-15)

    # with torch.autocast(device_type="cuda", enabled=True):
    for i in range(5000):
        print(f"STEP: {i}; {im.std()}; {torch.cat([torch.flatten(p) for p in sd.unet_lora.parameters()]).std()}")
        optimizer.zero_grad()

        # loss = sd.sds_loss(text_embedding, im, guidance_scale=20.0)
        loss = sd.vsd_forward(text_embedding_vd, text_embedding, im, guidance_scale=7.5)
        loss.backward()

        # for name, param in sd.unet_lora.named_parameters():
        #     if param.requires_grad and param.grad is not None:
        #         print(name, param.grad.norm())

        optimizer.step()

        if i % 10 == 0:
            mediapy.write_image(str(save_path), im.clip(0.0, 1.0).permute(0, 2, 3, 1).detach().cpu()[0])

    im = im.clip(0.0, 1.0).permute(0, 2, 3, 1).detach().cpu()
    print("IM SHAPE:", im.shape)
    mediapy.write_image(str(save_path), im[0])


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
        sd = StableDiffusion(cuda_device, use_sds=True).to(cuda_device)
        imgs = sd.prompt_to_img(prompt, negative, steps)
        print("IMGS SHAPE:", imgs.shape)
        mediapy.write_image(str(save_path), imgs[0])


if __name__ == "__main__":
    # tyro.cli(generate_image)
    tyro.cli(vsd_generate_image)
