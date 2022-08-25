# Copyright 2022 The Plenoptix Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Optimizers class.
"""
from __future__ import annotations

from typing import Any, Dict, List, Union

from omegaconf import DictConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from nerfactory.utils import writer


def setup_optimizers(
    config: Union[DictConfig, Dict[str, Any]], param_groups: Dict[str, List[Parameter]]
) -> "Optimizers":
    """Helper to set up the optimizers

    Args:
        config: The optimizer configuration object.
        param_groups: A dictionary of parameter groups to optimize.

    Returns:
        The optimizers object.
    """
    return Optimizers(config, param_groups)


class Optimizers:
    """A set of optimizers.

    Args:
        config: The optimizer configuration object.
        param_groups: A dictionary of parameter groups to optimize.
    """

    def __init__(self, config: Union[DictConfig, Dict[str, Any]], param_groups: Dict[str, List[Parameter]]):
        self.config = config
        self.optimizers = {}
        self.schedulers = {}
        for param_group_name, params in param_groups.items():
            lr_init = config[param_group_name]["optimizer"].lr
            self.optimizers[param_group_name] = config[param_group_name]["optimizer"].setup(params=params)
            if config[param_group_name]["scheduler"]:
                self.schedulers[param_group_name] = config[param_group_name]["scheduler"].setup(
                    optimizer=self.optimizers[param_group_name], lr_init=lr_init
                )

    def optimizer_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding optimizer.

        Args:
            param_group_name: name of optimizer to step forward
        """
        self.optimizers[param_group_name].step()

    def scheduler_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding scheduler.

        Args:
            param_group_name: name of scheduler to step forward
        """
        if self.config.param_group_name.scheduler:  # type: ignore
            self.schedulers[param_group_name].step()

    def zero_grad_all(self) -> None:
        """Zero the gradients for all optimizer parameters."""
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def optimizer_scaler_step_all(self, grad_scaler: GradScaler) -> None:
        """Take an optimizer step using a grad scaler.

        Args:
            grad_scaler: GradScaler to use
        """
        for _, optimizer in self.optimizers.items():
            grad_scaler.step(optimizer)

    def optimizer_step_all(self):
        """Run step for all optimizers."""
        for _, optimizer in self.optimizers.items():
            # note that they key is the parameter name
            optimizer.step()

    def scheduler_step_all(self, step: int) -> None:
        """Run step for all schedulers.

        Args:
            step: the current step
        """
        for param_group_name, scheduler in self.schedulers.items():
            scheduler.step()
            # TODO(ethan): clean this up. why is there indexing into a list?
            lr = scheduler.get_last_lr()[0]
            writer.put_scalar(name=f"learning_rate/{param_group_name}", scalar=lr, step=step)

    def load_optimizers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the optimizer state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.optimizers[k].load_state_dict(v)
