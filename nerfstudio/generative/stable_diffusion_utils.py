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

def get_text_embedding(batch, prompting_type, text_embeddings, sd_device):
    """"""

    base = text_embeddings["base_text_embedding"]
    front = text_embeddings["front_text_embedding"]
    side = text_embeddings["side_text_embedding"]
    top = text_embeddings["top_text_embedding"]
    back = text_embeddings["back_text_embedding"]

    assert prompting_type in ("location_based", "interpolated", "base")

    if prompting_type == "location_based":
        if batch["vertical"] < 40:
            text_embedding = top
        elif batch["central"] > 315 or batch["central"] <= 45:
            text_embedding = front
        elif batch["central"] > 45 and batch["central"] <= 135:
            text_embedding = side
        elif batch["central"] > 135 and batch["central"] <= 225:
            text_embedding = back
        else:  # batch["central"] > 225 and batch["central"] <= 315:
            text_embedding = side

    elif prompting_type == "interpolated":
        horiz = batch["central"].to(sd_device)
        vert = max(batch["vertical"].to(sd_device), 0)

        if batch["central"] > 0 or batch["central"] <= 90:
            text_embedding = (horiz) * side + (90 - horiz) * front
        elif batch["central"] > 90 and batch["central"] <= 180:
            text_embedding = (horiz - 90) * back + (180 - horiz) * side
        elif batch["central"] > 180 and batch["central"] <= 270:
            text_embedding = (horiz - 180) * side + (270 - horiz) * back
        else:  # batch["central"] > 270 and batch["central"] <= 360:
            text_embedding = (horiz - 270) * front + (360 - horiz) * side
            
        text_embedding = text_embedding / 90.0
        text_embedding = (vert * text_embedding + (90 - vert) * top) / 90.0

    elif prompting_type == "base":
        text_embedding = base

    return text_embedding
