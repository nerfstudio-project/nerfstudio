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

"""
Stable diffusion utils
"""

from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.generative.stable_diffusion import StableDiffusion


class PositionalTextEmbeddings:
    """Postional Prompts. Currently the following location based text embeddings are supported:
        - discrete: Choose the embedding closest to the camera position
        - interpolated: Interpolate between the embeddings based on the camera position
        - off: Don't modify the text embedding based on the camera position

    Args:
        base_prompt: Prompt for base view
        top_prompt: Prompt for top view
        side_prompt: Prompt for side view
        back_prompt: Prompt for back view
        front_prompt: Prompt for front view
        stable_diffusion: Instance of StableDiffusion
        positional_prompting: how to incorporate position into prompt.
    """

    def __init__(
        self,
        base_prompt: str,
        top_prompt: str,
        side_prompt: str,
        back_prompt: str,
        front_prompt: str,
        stable_diffusion: StableDiffusion,
        positional_prompting: Literal["discrete", "interpolated", "off"] = "discrete",
    ):
        assert positional_prompting in ["discrete", "interpolated", "off"]
        self.positional_prompting = positional_prompting
        self.sd_device = stable_diffusion.device
        self.sd = stable_diffusion
        self.update_prompt(base_prompt, top_prompt, side_prompt, back_prompt, front_prompt)

    def update_prompt(self, base_prompt: str, top_prompt: str, side_prompt: str, back_prompt: str, front_prompt: str):
        """Update the text embeddings based on the new prompts.

        Args:
            base_prompt: Prompt for base view
            top_prompt: Prompt for top view
            side_prompt: Prompt for side view
            back_prompt: Prompt for back view
            front_prompt: Prompt for front view
        """
        self.base_embed = self.sd.get_text_embeds(base_prompt, "")
        self.top_embed = self.sd.get_text_embeds(top_prompt, "")
        self.side_embed = self.sd.get_text_embeds(side_prompt, "")
        self.back_embed = self.sd.get_text_embeds(back_prompt, "")
        self.front_embed = self.sd.get_text_embeds(front_prompt, "")

    def get_text_embedding(self, vertical_angle: TensorType[1], horizontal_angle: TensorType[1]):
        """Get text embedding based on the position of the camera relative to the scene.
        This trick is used in Dreamfusion (https://dreamfusion3d.github.io/).

        Args:
            vertical_angle: vertical angle of the camera
            horizonal_angle: horizonal angle of the camera
        """

        if self.positional_prompting == "discrete":
            if vertical_angle < 40:
                text_embedding = self.top_embed
            elif 315 < horizontal_angle or horizontal_angle <= 45:
                text_embedding = self.front_embed
            elif 45 < horizontal_angle <= 135:
                text_embedding = self.side_embed
            elif 135 < horizontal_angle <= 225:
                text_embedding = self.back_embed
            else:  # horizontal_angle > 225 and horizontal_angle <= 315:
                text_embedding = self.side_embed
        elif self.positional_prompting == "interpolated":
            horiz = horizontal_angle.to(self.sd_device)
            vert = max(vertical_angle.to(self.sd_device), 0)

            if 0 < horizontal_angle <= 90:
                text_embedding = (horiz) * self.side_embed + (90 - horiz) * self.front_embed
            elif 90 < horizontal_angle <= 180:
                text_embedding = (horiz - 90) * self.back_embed + (180 - horiz) * self.side_embed
            elif 180 < horizontal_angle <= 270:
                text_embedding = (horiz - 180) * self.side_embed + (270 - horiz) * self.back_embed
            else:  # 270 < horizontal_angle <= 360:
                text_embedding = (horiz - 270) * self.front_embed + (360 - horiz) * self.side_embed

            text_embedding = text_embedding / 90.0
            text_embedding = (vert * text_embedding + (90 - vert) * self.top_embed) / 90.0
        else:
            print("here")
            text_embedding = self.base_embed

        return text_embedding
