"""
The Graph module contains all trainable parameters.
"""
import importlib
from dataclasses import dataclass

from torch import nn


@dataclass
class GraphInputs:
    """Datastucture to encode the inputs to the graph."""

    points: float


@dataclass
class Node:
    """Node datastructure for graph composition."""

    name: str
    children: set

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

    def construct_graph(self) -> set:
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
                curr_module = Node(name=module_name, children=set())
                processed_modules[module_name] = curr_module
            else:
                curr_module = processed_modules[module_name]
            inputs = module_dict["inputs"]
            for input_module in inputs:
                if not input_module in processed_modules:
                    parent_module = Node(name=input_module, children=set([curr_module]))
                    processed_modules[input_module] = parent_module
                else:
                    processed_modules[input_module].children.add(curr_module)
                if input_module == "x":
                    roots.add(curr_module)
        return roots

    def get_module_order(self) -> list:
        """Generates a graph and determines order of module operations using topological sorting

        Args:
            modules_config (dict): module definitions that make up the network

        Returns:
            list: ordering of the module names that should be executed
        """

        # call construct graph
        # function to do topological sort to get ordering
        return self.modules_config.keys()

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        # for name in self.module_order:
        #     x = self.modules[name](x)

        ## can index into previous results using dictionary
        raise NotImplementedError
