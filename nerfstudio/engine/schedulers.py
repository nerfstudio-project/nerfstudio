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

# pylint: disable=too-few-public-methods

"""Scheduler Classes"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler
from typing_extensions import Literal

from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config"""

    _target: Type = field(default_factory=lambda: Scheduler)
    """target class to instantiate"""


class Scheduler:
    """Base scheduler"""

    config: SchedulerConfig

    def __init__(self, config: SchedulerConfig) -> None:
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
    """Config for exponential decay scheduler with warmup"""

    _target: Type = field(default_factory=lambda: ExponentialDecayScheduler)
    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    max_steps: int = 100000
    """The maximum number of steps."""
    ramp: Literal["linear", "cosine"] = "cosine"
    """The ramp function to use during the warmup."""


class ExponentialDecayScheduler(Scheduler):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: ExponentialDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final

        def func(step):
            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (1 - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                        self.config.lr_pre_warmup
                        + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler


@dataclass
class CosineDecaySchedulerConfig(SchedulerConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: CosineDecayScheduler)
    """target class to instantiate"""
    warm_up_end: int = 5000
    """Iteration number where warmp ends"""
    learning_rate_alpha: float = 0.05
    """Learning rate alpha value"""
    max_steps: int = 300000
    """The maximum number of steps."""


class CosineDecayScheduler(Scheduler):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    config: CosineDecaySchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler._LRScheduler:
        def func(step):
            if step < self.config.warm_up_end:
                learning_factor = step / self.config.warm_up_end
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.config.warm_up_end) / (self.config.max_steps - self.config.warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
