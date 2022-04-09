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


class Graph(nn.Module):
    """_summary_ """

    def __init__(self, modules_config: dict) -> None:
        super().__init__()
        self.modules = {}
        self.module_order = self.get_module_order(modules_config)
        for module_name, module_dict in modules_config.items():
            module = getattr(importlib.import_module("mattport.nerf.modules"), module_dict["class_name"])
            if module_dict["class_name"] != "Encoding":
                in_dim = 0
                for inputs in module_dict["inputs"]:
                    in_dim += modules_config[inputs]["meta_data"]["out_dim"]
                module_dict["meta_data"]["in_dim"] = in_dim
            self.modules[module_name] = module(**module_dict["meta_data"])

    def get_module_order(self, modules_config: dict) -> list:
        """Generates a graph to determine the order in which the modules should execute

        Args:
            modules_config (dict): module definitions that make up the network

        Returns:
            list: ordering of the module names that should be executed
        """
        raise NotImplementedError

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        # for name in self.module_order:
        #     x = self.modules[name](x)
        raise NotImplementedError
