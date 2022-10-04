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

from dataclasses import dataclass, field
from typing import Any, Optional, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

from nerfstudio.configs.base_config import InstantiateConfig


@dataclass
class SchedulerConfig(InstantiateConfig):
    """Basic scheduler config with self-defined exponential decay schedule"""

    _target: Type = field(default_factory=lambda: ExponentialDecaySchedule)
    lr_final: float = 0.000005
    max_steps: int = 1000000

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, optimizer=None, lr_init=None, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(optimizer, lr_init, self.lr_final, self.max_steps)


class ExponentialDecaySchedule(lr_scheduler.LambdaLR):
    """Exponential learning rate decay function.
    See https://github.com/google-research/google-research/blob/
    fd2cea8cdd86b3ed2c640cbe5561707639e682f3/jaxnerf/nerf/utils.py#L360
    for details.

    Args:
        optimizer: The optimizer to update.
        lr_init: The initial learning rate.
        lr_final: The final learning rate.
        max_steps: The maximum number of steps.
        lr_delay_steps: The number of steps to delay the learning rate.
        lr_delay_mult: The multiplier for the learning rate after the delay.
    """

    config: SchedulerConfig

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


class DelayerScheduler(lr_scheduler.LambdaLR):
    """Starts with a flat lr schedule until it reaches N epochs then applies a given scheduler"""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init,  # pylint: disable=unused-argument
        lr_final,  # pylint: disable=unused-argument
        max_steps,  # pylint: disable=unused-argument
        delay_epochs: int = 500,
        after_scheduler: Optional[lr_scheduler.LambdaLR] = None,
    ) -> None:
        def func(step):
            if step > delay_epochs:
                if after_scheduler is not None:
                    multiplier = after_scheduler.lr_lambdas[0](step - delay_epochs)  # type: ignore
                    return multiplier
                return 1.0
            return 0.0

        super().__init__(optimizer, lr_lambda=func)


class DelayedExponentialScheduler(DelayerScheduler):
    """Delayer Scheduler with an Exponential Scheduler initialized afterwards."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_init,
        lr_final,
        max_steps,
        delay_epochs: int = 200,
    ):
        after_scheduler = ExponentialDecaySchedule(
            optimizer,
            lr_init,
            lr_final,
            max_steps,
        )
        super().__init__(optimizer, lr_init, lr_final, max_steps, delay_epochs, after_scheduler=after_scheduler)
