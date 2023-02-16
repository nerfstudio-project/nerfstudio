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

"""Utility functions for Stable Diffusion"""

def get_text_embedding(batch, prompting_type, prompts, device):
    """util function for getting correct text_embedding from view direction
    Args:
        batch: contains ray batch information
        prompting_type: how to sample text embeddings
        prompts: dictionary of text embeddings

    """
    assert prompting_type in ("location_based", "interpolated", "base")

    if prompting_type == "location_based":
        if batch["vertical"] < 40:
            text_embedding = prompts["top_text_embedding"]
        elif batch["central"] > 315 or batch["central"] <= 45:
            text_embedding = prompts["front_text_embedding"]
        elif batch["central"] > 45 and batch["central"] <= 135:
            text_embedding = prompts["side_text_embedding"]
        elif batch["central"] > 135 and batch["central"] <= 225:
            text_embedding = prompts["back_text_embedding"]
        else:  # batch["central"] > 225 and batch["central"] <= 315:
            text_embedding = prompts["side_text_embedding"]

    elif prompting_type == "interpolated":
        horiz = batch["central"].to(device)
        vert = max(batch["vertical"].to(device), 0)

        if batch["central"] > 0 or batch["central"] <= 90:
            text_embedding = (horiz) * prompts["side_text_embedding"] + (90 - horiz) * prompts["front_text_embedding"]
        elif batch["central"] > 90 and batch["central"] <= 180:
            text_embedding = (horiz - 90) * prompts["back_text_embedding"] + (180 - horiz) * prompts["side_text_embedding"]
        elif batch["central"] > 180 and batch["central"] <= 270:
            text_embedding = (horiz - 180) * prompts["side_text_embedding"] + (270 - horiz) * prompts["back_text_embedding"]
        else:  # batch["central"] > 270 and batch["central"] <= 360:
            text_embedding = (horiz - 270) * prompts["front_text_embedding"] + (360 - horiz) * prompts["side_text_embedding"]
            
        text_embedding = text_embedding / 90.0
        text_embedding = (vert * text_embedding + (90 - vert) * prompts["top_text_embedding"]) / 90.0

    else:
        text_embedding = prompts["base_text_embedding"]

    return text_embedding
