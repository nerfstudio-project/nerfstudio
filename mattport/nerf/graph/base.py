"""
The Graph module contains all trainable parameters.
"""
import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from torch import nn
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

    def construct_graph(self) -> Set[Node]:
        """Constructs a dependency graph given the module configuration

        Args:
            config (dict): module definitions that make up the network

        Returns:
            set: all root nodes of the constructed dependency graph
        """
        processed_modules = {}
        roots = set()
        for module_name, module_dict in self.modules.items():
            if not module_name in processed_modules:
                curr_module = Node(name=module_name, children={}, parents={})
                processed_modules[module_name] = curr_module
            else:
                curr_module = processed_modules[module_name]
            inputs = module_dict.inputs
            for input_module in inputs:
                if not input_module in processed_modules:
                    parent_module = Node(
                        name=input_module,
                        children={module_name: curr_module},
                        parents={},
                    )
                    processed_modules[input_module] = parent_module
                else:
                    processed_modules[input_module].children[module_name] = curr_module
                if input_module == "x":
                    roots.add(curr_module)
                else:
                    curr_module.parents[input_module] = parent_module
        return roots

    def topological_sort(self, curr_node: Node, ordering_stack: List[str]) -> None:
        """utility function to sort the call order in the dependency graph

        Args:
            curr_node (Node): pointer to current node in process
            ordering_stack (list): cumulative ordering of graph nodes
        """
        curr_node.visited_order = True
        for child_node in curr_node.children.values():
            if not child_node.visited_order:
                self.topological_sort(child_node, ordering_stack)
        ordering_stack.append(curr_node.name)

    def get_module_order(self) -> List[str]:
        """Generates a graph and determines order of module operations using topological sorting

        Args:
            config (dict): module definitions that make up the network

        Returns:
            list: ordering of the module names that should be executed
        """
        roots = self.roots
        ordering_stack = []
        for root in roots:
            if not root.visited_order:
                self.topological_sort(root, ordering_stack)
        return ordering_stack[::-1]

    @abstractmethod
    def forward(self, ray_indices: TensorType["num_rays", 3]):
        """Forward function that needs to be overridden."""

    @abstractmethod
    def get_losses(self, batch, graph_outputs):
        """Computes and returns the losses."""
