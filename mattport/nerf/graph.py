"""
The Graph module contains all trainable parameters.
"""
import importlib
from dataclasses import dataclass
from typing import Optional

from torch import nn


@dataclass
class GraphInputs:
    """Datastucture to encode the inputs to the graph."""

    points: float


@dataclass
class Node:
    """Node datastructure for graph composition."""

    name: str
    children: dict
    visited: Optional[bool] = False

    def __hash__(self):
        return hash(self.name)


class Graph(nn.Module):
    """_summary_"""

    def __init__(self, modules_config: dict) -> None:
        super().__init__()
        self.modules_config = modules_config
        # calculate input dimensions based on module dependencies
        self.modules = {}
        for module_name, module_dict in modules_config.items():
            module = getattr(importlib.import_module("mattport.nerf.modules"), module_dict["class_name"])
            if module_dict["class_name"] != "Encoding":
                in_dim = 0
                for inputs in module_dict["inputs"]:
                    in_dim += modules_config[inputs]["meta_data"]["out_dim"]
                module_dict["meta_data"]["in_dim"] = in_dim
            self.modules[module_name] = module(**module_dict["meta_data"])
        # generate dependency ordering for module calls
        self.module_order = self.get_module_order()

    def construct_graph(self) -> dict:
        """Constructs a dependency graph given the module configuration

        Args:
            modules_config (dict): module definitions that make up the network

        Returns:
            list: root nodes for the constructed dependency graph
        """
        processed_modules = {}
        roots = set()
        for module_name, module_dict in self.modules_config.items():
            if not module_name in processed_modules:
                curr_module = Node(name=module_name, children={})
                processed_modules[module_name] = curr_module
            else:
                curr_module = processed_modules[module_name]
            inputs = module_dict["inputs"]
            for input_module in inputs:
                if not input_module in processed_modules:
                    parent_module = Node(name=input_module, children={module_name: curr_module})
                    processed_modules[input_module] = parent_module
                else:
                    processed_modules[input_module].children[module_name] = curr_module
                if input_module == "x":
                    roots.add(curr_module)
        return roots

    def topological_sort(self, curr_node: "Node", ordering_stack: list) -> None:
        """utility function to sort the call order in the dependency graph

        Args:
            curr_node (Node): pointer to current node in process
            ordering_stack (list): cumulative ordering of graph nodes
        """
        curr_node.visited = True
        for child_node in curr_node.children.values():
            if not child_node.visited:
                self.topological_sort(child_node, ordering_stack)
        ordering_stack.append(curr_node.name)

    def get_module_order(self) -> list:
        """Generates a graph and determines order of module operations using topological sorting

        Args:
            modules_config (dict): module definitions that make up the network

        Returns:
            list: ordering of the module names that should be executed
        """
        roots = self.construct_graph()
        ordering_stack = []
        for root in roots:
            if not root.visited:
                self.topological_sort(root, ordering_stack)

        return ordering_stack[::-1]

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        # for name in self.module_order:
        #     x = self.modules[name](x)

        ## can index into previous results using dictionary
        raise NotImplementedError
