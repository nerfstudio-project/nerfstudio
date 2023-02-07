"""Trace Stable Diffusion UNet for speed improvement"""

import functools
import sys
from pathlib import Path

import appdirs
import torch
import tyro
from rich.console import Console

CONSOLE = Console(width=120)

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)


def jit_unet(save_dir: Path = Path(appdirs.user_data_dir("nerfstudio"))):
    """Trace Stable Diffusion UNet for speed improvement

    Args:
        save_dir: directory to save the traced model
    """

    # torch disable grad
    torch.set_grad_enabled(False)

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
    inputs = None
    for _ in range(3):
        with torch.inference_mode():
            inputs = generate_inputs()
            _ = unet(*inputs)

    # trace
    print("tracing..")
    assert inputs is not None
    unet_traced = torch.jit.trace(unet, inputs)
    unet_traced.eval()
    print("done tracing")

    # save the traced model
    unet_traced_filename = save_dir / "sd_unet_traced.pt"
    if not unet_traced_filename.exists():
        unet_traced_filename.parent.mkdir(parents=True, exist_ok=True)
        unet_traced.save(unet_traced_filename)


if __name__ == "__main__":
    tyro.cli(jit_unet)
