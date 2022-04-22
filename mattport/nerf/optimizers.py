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
        self.config = config
        self.optimizers = {}
        self.schedulers = {}
        for param_name, params in param_groups.items():
            optimizer_config = config[param_name].optimizer
            optimizer = getattr(torch.optim, optimizer_config.type)
            kwargs = {k: v for k, v in optimizer_config.items() if k != "type"}
            self.optimizers[param_name] = optimizer(params, **kwargs)
            if config[param_name].scheduler:
                scheduler_config = config[param_name].scheduler
                scheduler = getattr(torch.optim.lr_scheduler, scheduler_config.type)
                kwargs = {k: v for k, v in scheduler_config.items() if k != "type"}
                self.schedulers[param_name] = scheduler(self.optimizers[param_name], **kwargs)

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

    def zero_grad_all(self):
        """zero the gradients for all optimizer parameters"""
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def optimizer_step_all(self):
        """Run step for all optimizers."""
        # logging.info("optimizer_step_all")
        for param_name, optimizer in self.optimizers.items():
            optimizer.step()

    def scheduler_step_all(self, step):
        """Run step for all schedulers."""
        # logging.info("optimizer_step_all")
        pass
