"""
Optimizers class.
"""

from typing import Dict, List
from omegaconf import DictConfig
import torch
from torch.nn import Parameter


class Optimizers:
    """_summary_"""

    def __init__(self, config: DictConfig, param_groups: Dict[str, List[Parameter]]):
        """_summary_

        Args:
            config (DictConfig): _description_
            param_dict (Dict[str, List[Parameter]]): _description_
        """
        # TODO() add param_name `learning_rate`` and `weight_decay`, etc... to config file
        self.config = config
        self.optimizers = {}
        self.schedulers = {}
        for param_name, params in param_groups.items():
            optimizer = getattr(torch.optim, config.param_name.optimizer.type)
            kwargs = {k: v for k, v in config.param_name.optimizer.items() if k != "type"}
            self.optimizers[param_name] = optimizer(params, **kwargs)
            if config.param_name.scheduler:
                scheduler = getattr(torch.optim.lr_scheduler, config.param_name.scheduler.type)
                kwargs = {k: v for k, v in config.param_name.scheduler.items() if k != "type"}
                self.schedulers[param_name] = scheduler(optimizer, **kwargs)

    def optimizer_step(self, param_name: str) -> None:
        """fetch and step corresponding optimizer

        Args:
            param_name (str): name of optimizer to step forward
        """
        self.optimizers[param_name].step()

    def scheduler_step(self, param_name: str) -> None:
        """fetch and step corresponding scheduler

        Args:
            param_name (str): name of scheduler to step forward
        """
        if self.config.param_name.scheduler:
            self.schedulers[param_name].step()
