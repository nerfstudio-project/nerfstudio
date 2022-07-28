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

from typing import Callable


class Callback:
    """Callback class used during training.
    The function 'func' with 'args' and 'kwargs' will be called every 'update_every_num_iters' training iterations,
    including at iteration 0.

    Args:
        update_every_num_iters: How often to call the function `func`.
        func: The function that will be called.
        args: args for the function 'func'.
        kwargs: kwargs for the function 'func'.
    """

    def __init__(self, update_every_num_iters: int, func: Callable, *args, **kwargs):
        # TODO(ethan): how do we type args and kwargs?
        self.update_every_num_iters = update_every_num_iters
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def before_step(self, step: int):
        """callback to run before training step

        Args:
            step (int): current iteration step
        """
        raise NotImplementedError

    def after_step(self, step: int):
        """callback to run after training step

        Args:
            step (int): current iteration step
        """
        if step % self.update_every_num_iters == 0:
            self.func(*self.args, **self.kwargs, step=step)
