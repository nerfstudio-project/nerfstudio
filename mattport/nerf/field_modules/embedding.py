"""
Code for embeddings.
"""


import torch
from torchtyping import TensorType
from mattport.nerf.field_modules.base import FieldModule


class Embedding(FieldModule):
    """Index into embeddings."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """
        Args:
            in_dim (int): Number of embeddings
            out_dim (int): Dimension of the embedding vectors
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.build_nn_modules()

    def build_nn_modules(self) -> None:
        self.embedding = torch.nn.Embedding(self.in_dim, self.out_dim)

    def forward(self, in_tensor: TensorType[..., "input_dim"]) -> TensorType[..., "output_dim"]:
        """Call forward"""
        return self.embedding(in_tensor)
