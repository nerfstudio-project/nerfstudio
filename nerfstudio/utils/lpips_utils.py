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
This file contains utils for LPIPS in order to support masking.
We effectively copy of the relevant code from torchmetrics.image.lpip
and then return full images which we can later mask. Some functions are copied over
simply to override functions.
"""

from torch import Tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def _valid_img(img: Tensor, normalize: bool):
    """check that input is a valid image to the network."""
    value_check = img.max() <= 1.0 and img.min() >= 0.0 if normalize else img.min() >= -1
    return img.ndim == 4 and img.shape[1] == 3 and value_check


class LearnedPerceptualImagePatchSimilarityWithMasking(LearnedPerceptualImagePatchSimilarity):
    """LearnedPerceptualImagePatchSimilarity module that will allow for masking capabilities."""

    def update(self, img1: Tensor, img2: Tensor) -> None:  # pylint: disable=arguments-differ
        """Update internal states with lpips score with masking."""

        # hardcode this to True for now to avoid touching a lot of the torchmetrics code
        self.net.spatial = True

        if not (_valid_img(img1, self.normalize) and _valid_img(img2, self.normalize)):
            raise ValueError(
                "Expected both input arguments to be normalized tensors with shape [N, 3, H, W]."
                f" Got input with shape {img1.shape} and {img2.shape} and values in range"
                f" {[img1.min(), img1.max()]} and {[img2.min(), img2.max()]} when all values are"
                f" expected to be in the {[0,1] if self.normalize else [-1,1]} range."
            )
        loss = self.net(img1, img2, normalize=self.normalize)
        # now loss is the shape [batch size, H, W]
        # we set loss to self.sum_scores to use the existing API from torchvision
        self.sum_scores = loss  # pylint: disable=attribute-defined-outside-init

    def compute(self) -> Tensor:
        """Compute final perceptual similarity metric."""
        # note that we don't use self.reduction anymore
        return self.sum_scores
