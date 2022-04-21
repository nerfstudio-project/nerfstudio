"""
The Graph module contains all trainable parameters.
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import nn
from torch.nn import Parameter
from torchtyping import TensorType


@dataclass
class GraphInputs:
    """Datastucture to encode the inputs to the graph."""

    points: float


@dataclass
class Node:
    """Node datastructure for graph composition."""

    name: str
    children: Dict[str, "Node"]
    parents: Dict[str, "Node"]
    visited_order: Optional[bool] = False
    visited_in_dim: Optional[bool] = False

    def __hash__(self):
        return hash(self.name)


class Graph(nn.Module):
    """_summary_"""

    def __init__(self, intrinsics=None, camera_to_world=None) -> None:
        super().__init__()
        self.intrinsics = intrinsics
        self.camera_to_world = camera_to_world
        self.populate_modules()  # populate the modules

    @abstractmethod
    def populate_modules(self):
        """Initializes the modules that are part of the network."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups of the network in a dictionary format"""

    def get_in_dim(self, curr_node: Node) -> None:
        """Dynamically calculates and sets the input dimensions of the modules based on dependency graph

        Args:
            curr_node (Node): pointer to current node in process
        """
        curr_node.visited_in_dim = True
        if len(curr_node.parents) > 0:
            in_dim = 0
            for parent_name in curr_node.parents.keys():
                in_dim += self[parent_name].get_out_dim()
            self[curr_node.name].set_in_dim(in_dim)
            self.modules[curr_node.name].meta_data.in_dim = in_dim

        for child_node in curr_node.children.values():
            if not child_node.visited_in_dim:
                self.get_in_dim(child_node)

    @abstractmethod
    def forward(self, ray_indices: TensorType["num_rays", 3]):
        """Forward function that needs to be overridden."""

    @abstractmethod
    def get_losses(self, batch, graph_outputs):
        """Computes and returns the losses."""
