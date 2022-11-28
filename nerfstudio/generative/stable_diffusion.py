import torch
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, logging


class StableDiffusion(nn.Module):
    """Stable Diffusion implementation
    """

    def __init__(
        self,
        device
    ) -> None:
        super().__init__()
