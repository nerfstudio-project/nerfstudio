"""Trace Stable Diffusion UNet for speed improvement"""

import functools
import time
from pathlib import Path

import appdirs
import torch
from diffusers import StableDiffusionPipeline


def jit_unet():

    # torch disable grad
    torch.set_grad_enabled(False)

    # set variables
    n_experiments = 2
    unet_runs_per_experiment = 50

    # load inputs
    def generate_inputs():
        sample = torch.randn(2, 4, 64, 64).half().cuda()
        timestep = torch.rand(1).half().cuda() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half().cuda()
        return sample, timestep, encoder_hidden_states

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.enable_attention_slicing()

    unet = pipe.unet
    unet.eval()
    unet.to(memory_format=torch.channels_last)  # use channels_last memory format
    unet.forward = functools.partial(unet.forward, return_dict=False)  # set return_dict=False as default

    # warmup
    for _ in range(3):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet(*inputs)

    # trace
    print("tracing..")
    unet_traced = torch.jit.trace(unet, inputs)
    unet_traced.eval()
    print("done tracing")

    # warmup and optimize graph
    for _ in range(5):
        with torch.inference_mode():
            inputs = generate_inputs()
            orig_output = unet_traced(*inputs)

    # benchmarking
    with torch.inference_mode():
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet_traced(*inputs)
            torch.cuda.synchronize()
            print(f"unet traced inference took {time.time() - start_time:.2f} seconds")
        for _ in range(n_experiments):
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(unet_runs_per_experiment):
                orig_output = unet(*inputs)
            torch.cuda.synchronize()
            print(f"unet inference took {time.time() - start_time:.2f} seconds")

    # save the traced model
    unet_traced_filename = Path(appdirs.user_data_dir("nerfstudio")) / "unet_traced.pt"
    if not unet_traced_filename.exists():
        unet_traced_filename.parent.mkdir(parents=True, exist_ok=True)
        unet_traced.save(unet_traced_filename)


if __name__ == "__main__":
    jit_unet()
