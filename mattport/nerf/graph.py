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
    parents: dict
    visited_order: Optional[bool] = False
    visited_in_dim: Optional[bool] = False

    def __hash__(self):
        return hash(self.name)


class Graph(nn.Module):
    """_summary_"""

    def __init__(self, modules_config: dict) -> None:
        super().__init__()
        self.modules_config = modules_config
        # create graph and get ordering
        self.roots = self.construct_graph()
        self.module_order = self.get_module_order()

        # initialize graph with known input dimensions; set default in_dim to 0
        for module_name, module_dict in modules_config.items():
            module = getattr(importlib.import_module("mattport.nerf.modules"), module_dict["class_name"])
            if "in_dim" not in module_dict["meta_data"]:
                module_dict["meta_data"]["in_dim"] = 0
            self.modules[module_name] = module(**module_dict["meta_data"])

        # calculate input dimensions based on module dependencies
        for root in self.roots:
            self.set_in_dim(root)

    def set_in_dim(self, curr_node: "Node") -> None:
        """Dynamically calculates and sets the input dimensions of the modules based on dependency graph

        Args:
            curr_node (Node): pointer to current node in process
        """
        curr_node.visited_in_dim = True

        if len(curr_node.parents) > 0:
            in_dim = 0
            for parent_name in curr_node.parents.keys():
                in_dim += self.modules[parent_name].get_out_dim()
            self.modules[curr_node.name].set_in_dim()
            self.modules_config[curr_node.name]["meta_data"]["in_dim"] = in_dim

        for child_node in curr_node.children.values():
            if not child_node.visited_order:
                self.set_in_dim(child_node)

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
                curr_module = Node(name=module_name, children={}, parents={})
                processed_modules[module_name] = curr_module
            else:
                curr_module = processed_modules[module_name]
            inputs = module_dict["inputs"]
            for input_module in inputs:
                if not input_module in processed_modules:
                    parent_module = Node(name=input_module, children={module_name: curr_module}, parents={})
                    processed_modules[input_module] = parent_module
                else:
                    processed_modules[input_module].children[module_name] = curr_module
                curr_module.parents[input_module] = parent_module
                if input_module == "x":
                    roots.add(curr_module)
        return roots

    def topological_sort(self, curr_node: "Node", ordering_stack: list) -> None:
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

    def get_module_order(self) -> list:
        """Generates a graph and determines order of module operations using topological sorting

        Args:
            modules_config (dict): module definitions that make up the network

        Returns:
            list: ordering of the module names that should be executed
        """
        roots = self.roots
        ordering_stack = []
        for root in roots:
            if not root.visited_order:
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
