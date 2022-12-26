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

"""
Callback code used for training iterations
"""

from dataclasses import InitVar, dataclass
from enum import Enum, auto
from inspect import signature
from typing import Callable, Dict, List, Optional, Tuple

from nerfstudio.configs.base_config import TrainerConfig


@dataclass
class TrainingCallbackAttributes:
    """Attributes that can be used to configure training callbacks.
    The callbacks can be specified in the Dataloader or Model implementations.
    Instead of providing access to the entire Trainer object, we only provide these attributes.
    This should be least prone to errors and fairly clean from a user perspective."""

    # TODO(ethan): type this without circular imports
    optimizers: Optional[InitVar]
    """optimizers for training"""
    grad_scaler: Optional[InitVar]
    """gradient scalers"""
    pipeline: Optional[InitVar]
    """reference to training pipeline"""
    config: TrainerConfig
    """the trainer config"""


class TrainingCallbackLocation(Enum):
    """Enum for specifying where the training callback should be run."""

    BEFORE_TRAIN_ITERATION = auto()
    AFTER_TRAIN_ITERATION = auto()


class TrainingCallback:
    """Callback class used during training.
    The function 'func' with 'args' and 'kwargs' will be called every 'update_every_num_iters' training iterations,
    including at iteration 0. The function is called after the training iteration.

    Args:
        where_to_run: List of locations for when to run callbak (before/after iteration)
        func: The function that will be called.
        update_every_num_iters: How often to call the function `func`.
        iters: Tuple of iteration steps to perform callback
        args: args for the function 'func'.
        kwargs: kwargs for the function 'func'.
    """

    def __init__(
        self,
        where_to_run: List[TrainingCallbackLocation],
        func: Callable,
        update_every_num_iters: Optional[int] = None,
        iters: Optional[Tuple[int, ...]] = None,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
    ):
        assert (
            "step" in signature(func).parameters.keys()
        ), f"'step: int' must be an argument in the callback function 'func': {func.__name__}"
        self.where_to_run = where_to_run
        self.update_every_num_iters = update_every_num_iters
        self.iters = iters
        self.func = func
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}

    def run_callback(self, step: int):
        """Callback to run after training step

        Args:
            step: current iteration step
        """
        if self.update_every_num_iters is not None:
            if step % self.update_every_num_iters == 0:
                self.func(*self.args, **self.kwargs, step=step)
        elif self.iters is not None:
            if step in self.iters:
                self.func(*self.args, **self.kwargs, step=step)

    def run_callback_at_location(self, step: int, location: TrainingCallbackLocation):
        """Runs the callback if it's supposed to be run at the given location.

        Args:
            step: current iteration step
            location: when to run callback (before/after iteration)
        """
        if location in self.where_to_run:
            self.run_callback(step=step)
