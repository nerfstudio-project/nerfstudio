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

"""Trace Stable Diffusion UNet for speed improvement"""

import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import appdirs
import torch
import tyro

from nerfstudio.generative.stable_diffusion import SD_IDENTIFIERS
from nerfstudio.utils.rich_utils import CONSOLE

try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)


@dataclass
class TraceSD:
    """Traces the Stable Diffusion U-Net for better performance."""

    sd_version: str
    """Stable diffusion version."""
    save_dir: Path = Path(appdirs.user_data_dir("nerfstudio"))
    """Directory to save the traced model."""

    def main(self):
        """Trace the Stable Diffusion U-Net and save it to disk."""
        # torch disable grad
        torch.set_grad_enabled(False)

        if self.sd_version not in SD_IDENTIFIERS:
            CONSOLE.print(f"[bold red]Invalid Stable Diffusion version; choose from {set(SD_IDENTIFIERS.keys())}")
            sys.exit(1)

        sd_id = SD_IDENTIFIERS[self.sd_version]
        pipe = StableDiffusionPipeline.from_pretrained(
            sd_id,
            torch_dtype=torch.float16,
        )
        assert pipe is not None
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()

        # load inputs
        def generate_inputs():
            sample = torch.randn(2, 4, 64, 64).half().cuda()
            timestep = torch.rand(1).half().cuda() * 999

            # get shape of encoder hidden state (varies by SD version)
            text_input = pipe.tokenizer("", return_tensors="pt")
            _, w, d = pipe.text_encoder(text_input.input_ids.cuda())[0].shape

            encoder_hidden_states = torch.randn(2, w, d).half().cuda()
            return sample, timestep, encoder_hidden_states

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
        unet_traced.eval()  # type: ignore
        print("done tracing")

        # save the traced model
        filename_sd_id = sd_id.split("/")[-1]
        unet_traced_filename = self.save_dir / f"{filename_sd_id}_unet_traced.pt"
        if not unet_traced_filename.exists():
            unet_traced_filename.parent.mkdir(parents=True, exist_ok=True)
            unet_traced.save(unet_traced_filename)  # type: ignore


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(TraceSD).main()


if __name__ == "__main__":
    entrypoint()
