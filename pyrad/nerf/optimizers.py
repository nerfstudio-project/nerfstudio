"""
Optimizers class.
"""

from typing import Dict, List

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR

from pyrad.utils import writer


class ExponentialDecaySchedule(LambdaLR):
    """Exponential learning rate decay function.
    See https://github.com/google-research/google-research/blob/
    fd2cea8cdd86b3ed2c640cbe5561707639e682f3/jaxnerf/nerf/utils.py#L360
    for details.
    """

    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1.0) -> None:
        def func(step):
            if lr_delay_steps > 0:
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            multiplier = (
                log_lerp / lr_init
            )  # divided by lr_init because the multiplier is with the initial learning rate
            return delay_rate * multiplier

        super().__init__(optimizer, lr_lambda=func)


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
        for param_group_name, params in param_groups.items():
            lr_init = config[param_group_name].optimizer.lr
            self.optimizers[param_group_name] = instantiate(config[param_group_name].optimizer, params=params)
            if config[param_group_name].scheduler:
                self.schedulers[param_group_name] = instantiate(
                    config[param_group_name].scheduler, optimizer=self.optimizers[param_group_name], lr_init=lr_init
                )

    def optimizer_step(self, param_group_name: str) -> None:
        """fetch and step corresponding optimizer

        Args:
            param_group_name (str): name of optimizer to step forward
        """
        self.optimizers[param_group_name].step()

    def scheduler_step(self, param_group_name: str) -> None:
        """fetch and step corresponding scheduler

        Args:
            param_group_name (str): name of scheduler to step forward
        """
        if self.config.param_group_name.scheduler:
            self.schedulers[param_group_name].step()

    def zero_grad_all(self):
        """zero the gradients for all optimizer parameters"""
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def optimizer_step_all(self):
        """Run step for all optimizers."""
        for _, optimizer in self.optimizers.items():
            # note that they key is the parameter name
            optimizer.step()

    def scheduler_step_all(self, step):
        """Run step for all schedulers."""
        for param_group_name, scheduler in self.schedulers.items():
            scheduler.step()
            # TODO(ethan): clean this up. why is there indexing into a list?
            lr = scheduler.get_last_lr()[0]
            writer.put_scalar(name=f"learning_rate/{param_group_name}", scalar=lr, step=step)
