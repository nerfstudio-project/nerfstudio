# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""Scheduler Classes"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config"""

    _target: Type = field(default_factory=lambda: Scheduler)
    """_target: target class to instantiate"""


class Scheduler:  # pylint: disable=too-few-public-methods
    """Exponential learning rate decay function."""

    config: SchedulerConfig

    def __init__(self, config: SchedulerConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """


@dataclass
class ExponentialDecaySchedulerConfig(SchedulerConfig):
    """Exponential learning rate decay config"""

    _target: Type = field(default_factory=lambda: ExponentialDecayScheduler)
    lr_final: float = 0.000005
    """The final learning rate."""
    max_steps: int = 1000000
    """The maximum number of steps."""
    lr_delay_steps: int = 0
    """The number of steps to delay the learning rate."""
    lr_delay_mult: float = 1.0
    """The multiplier for the learning rate after the delay."""


class ExponentialDecayScheduler(Scheduler):  # pylint: disable=too-few-public-methods
    """Exponential learning rate decay function.
    See https://github.com/google-research/google-research/blob/
    fd2cea8cdd86b3ed2c640cbe5561707639e682f3/jaxnerf/nerf/utils.py#L360
    for details.
    """

    config: ExponentialDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        def func(step):
            if self.config.lr_delay_steps > 0:
                delay_rate = self.config.lr_delay_mult + (1 - self.config.lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / self.config.lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / self.config.max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(self.config.lr_final) * t)
            multiplier = (
                log_lerp / lr_init
            )  # divided by lr_init because the multiplier is with the initial learning rate
            return delay_rate * multiplier

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
