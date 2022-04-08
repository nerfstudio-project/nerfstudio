"""
The Graph module contains all trainable parameters.
"""
from dataclasses import dataclass
from torch import nn
from mattport.nerf.modules.positional_encoder import PositionalEncoder

@dataclass
class GraphInputs:
    """Datastucture to encode the inputs to the graph."""

    points: float


class Graph(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, modules_config: dict) -> None:
        super().__init__()


        # # for key in ...
        # temp = graph_config.positional_encoder
        # PositionalEncoder(name=temp.name, name=temp.inputs, )
        # self.modules = {}
        # self.modules["name"] = MLP()


        self.modules = {}


        self.encoder_0 = Encoder()
        self.mlp_0 = MLP()


        raise NotImplementedError

    def forward(self, x):
        """_summary_"""
        # raise NotImplementedError
        
        for name in queue: # "encoder_0",
            # assert name.output is not None
            x = self.modules[name](x)
        # self.mlp_0