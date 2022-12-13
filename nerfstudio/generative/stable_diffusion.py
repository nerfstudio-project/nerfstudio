
# From https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from rich.console import Console
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, logging

CONSOLE = Console(width=120)
logging.set_verbosity_error()
IMG_DIM = 512
const_scale = 0.18215

class StableDiffusion(nn.Module):
    """Stable Diffusion implementation
    """

    def __init__(
        self,
        device
    ) -> None:
        super().__init__()

        self.device = device

        try:
            with open('./HF_TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')
                CONSOLE.print(f'Hugging face access token loaded.')
        except FileNotFoundError as e:
            raise AssertionError(f'Cannot read hugging face token, make sure to save your Hugging Face token at `./HF_TOKEN`')

        SD_SOURCE = "runwayml/stable-diffusion-v1-5"
        CLIP_SOURCE = "openai/clip-vit-large-patch14"

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        CONSOLE.print(f'Loading Stable Diffusion model from {SD_SOURCE}.')

        self.auto_encoder = AutoencoderKL.from_pretrained(SD_SOURCE, subfolder="vae", use_auth_token=self.token).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet", use_auth_token=self.token).to(self.device)

        CONSOLE.print(f'Loading CLIP model from {CLIP_SOURCE}.')

        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_SOURCE)
        self.text_encoder = CLIPTextModel.from_pretrained(CLIP_SOURCE).to(self.device)

        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        CONSOLE.print(f'Stable Diffusion loaded!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def sds_loss(self, text_embeddings, nerf_output, guidance_scale=100):

        nerf_output_np = nerf_output.view(3, 64, 64).permute(1, 2, 0).detach().cpu().numpy()
        nerf_output_np = np.clip(nerf_output_np, 0.0, 1.0)
        plt.imsave('nerf_output.png', nerf_output_np)
        nerf_output = F.interpolate(nerf_output, (IMG_DIM, IMG_DIM), mode='bilinear')

        print(torch.max(nerf_output), torch.min(nerf_output))

        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        latents = self.imgs_to_latent(nerf_output)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # noise_difference = self.latents_to_img(grad)
        # plt.imsave('noise_difference.png', noise_difference.view(3, 512, 512).permute(1, 2, 0).detach().cpu().numpy())

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()

        noise_loss = torch.mean(torch.nan_to_num(torch.square(noise_pred - noise)))

        # print('SDS LOSS:', noise_loss)

        return noise_loss, latents, grad

    def produce_latents(self, text_embeddings, height=IMG_DIM, width=IMG_DIM, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for t in self.scheduler.timesteps:
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def latents_to_img(self, latents):

        latents = 1 / const_scale * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def imgs_to_latent(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * const_scale

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', num_inference_steps=50, guidance_scale=7.5, latents=None):
       
        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [negative_prompts] if isinstance(negative_prompts, str) else negative_prompts
        text_embeddings = self.get_text_embeds(prompts, negative_prompts)
        latents = self.produce_latents(text_embeddings, height=IMG_DIM, width=IMG_DIM, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        diffused_img = self.latents_to_img(latents)
        diffused_img = diffused_img.detach().cpu().permute(0, 2, 3, 1).numpy()
        diffused_img = (diffused_img * 255).round().astype('uint8')

        return diffused_img

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.steps)

    plt.imsave('test_sd.png', imgs[0])