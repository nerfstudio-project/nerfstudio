
import mediapy

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from tqdm import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
# from diffusers.hub_utils import init_git_repo, push_to_hub




sd_prompt_token = 'concepts/nicechair-concept/learned_embeds.bin'

prompt = f'a high quality picture of <nicechair2>, side view'

sd_model_path = 'concepts/nicechair2-concept'
pipe = StableDiffusionPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

pipe = pipe.to(device)

print("model loaded")
print(pipe.device)

for i in tqdm(range(5)):
    images = pipe([prompt], num_inference_steps = 50, guidance_scale = 7.5).images
    save_path = f'txt_inversion_imgs/outputs/nicechair2_{i}_side.png'
    mediapy.write_image(str(save_path), images[0])

