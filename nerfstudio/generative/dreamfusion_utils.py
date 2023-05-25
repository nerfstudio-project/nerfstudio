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

"""Utility helper functions for generative 3D models"""

import torch
from torch import Tensor
from jaxtyping import Float
from typing_extensions import Literal
from typing import Union

from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.generative.deepfloyd import DeepFloyd


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
        diffusion_model: Instance of StableDiffusion
        positional_prompting: how to incorporate position into prompt.
    """

    def __init__(
        self,
        base_prompt: str,
        top_prompt: str,
        side_prompt: str,
        back_prompt: str,
        front_prompt: str,
        diffusion_model: Union[StableDiffusion, DeepFloyd],
        positional_prompting: Literal["discrete", "interpolated", "off"] = "discrete",
    ):
        self.positional_prompting = positional_prompting
        self.diffusion_device = diffusion_model.device
        self.diffusion = diffusion_model
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
        self.base_embed = self.diffusion.get_text_embeds(base_prompt, "")
        self.top_embed = self.diffusion.get_text_embeds(top_prompt, "")
        self.side_embed = self.diffusion.get_text_embeds(side_prompt, "")
        self.back_embed = self.diffusion.get_text_embeds(back_prompt, "")
        self.front_embed = self.diffusion.get_text_embeds(front_prompt, "")

        if isinstance(self.diffusion, DeepFloyd):
            self.diffusion.delete_text_encoder()

    def get_text_embedding(
        self, vertical_angle: Float[Tensor, "1"], horizontal_angle: Float[Tensor, "1"]
    ) -> Float[Tensor, "2 max_length embed_dim"]:
        """Get text embedding based on the position of the camera relative to the scene.
        This trick is used in Dreamfusion (https://dreamfusion3d.github.io/).

        Args:
            vertical_angle: vertical angle of the camera
            horizonal_angle: horizonal angle of the camera
        """
        # set horizontal_angle between 0, 360
        horizontal_angle = torch.fmod(horizontal_angle, 360)
        horizontal_angle = torch.where(horizontal_angle < 0, horizontal_angle + 360, horizontal_angle)

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
            horiz = horizontal_angle.detach().numpy()[0]
            vert = max(vertical_angle.detach().numpy()[0], 0)

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
            text_embedding = self.base_embed

        return text_embedding
