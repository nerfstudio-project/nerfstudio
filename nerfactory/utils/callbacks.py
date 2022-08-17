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
Callback code used for training iterations
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

# from nerfactory.optimizers.optimizers import Optimizers


@dataclass
class TrainingCallbackAttributes:
    """Attributes that can be used to configure training callbacks.
    The callbacks can be specified in the Dataloader or Model implementations.
    Instead of providing access to the entire Trainer object, we only provide these attributes.
    This should be least prone to errors and fairly clean from a user perspective."""

    optimizers = None #: Optimizers
    grad_scaler = None #: GradScaler
    pipeline = None #: Pipeline


class TrainingCallback:
    """Callback class used during training.
    The function 'func' with 'args' and 'kwargs' will be called every 'update_every_num_iters' training iterations,
    including at iteration 0. The function is called after the training iteration.

    Args:
        update_every_num_iters: How often to call the function `func`.
        func: The function that will be called.
        args: args for the function 'func'.
        kwargs: kwargs for the function 'func'.
    """

    def __init__(
        self,
        func: Callable,
        update_every_num_iters: Optional[int] = None,
        iters: Optional[Tuple[int, ...]] = None,
        reinit: bool = False,
        *args,
        **kwargs
    ):
        # TODO(ethan): how do we type args and kwargs?
        self.update_every_num_iters = update_every_num_iters
        self.iters = iters
        self.func = func
        self.reinit = reinit
        self.args = args
        self.kwargs = kwargs

    def after_train_iteration(self, step: int):
        """Callback to run after training step

        Args:
            step (int): current iteration step
        """
        if self.update_every_num_iters is not None:
            if step % self.update_every_num_iters == 0:
                self.func(*self.args, **self.kwargs, step=step)
        elif self.iters is not None:
            if step in self.iters:
                self.func(*self.args, **self.kwargs, step=step)
