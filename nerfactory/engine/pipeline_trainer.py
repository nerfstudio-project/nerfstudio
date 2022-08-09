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
Code to train model.
"""
import logging

from nerfactory.utils import writer
from nerfactory.utils.writer import EventName, TimeWriter

from nerfactory.pipeline.base import Pipeline

logging.getLogger("PIL").setLevel(logging.WARNING)


class BarebonesTrainer:
    """Training class for the pipeline, with no visualizer code at all.

    This is to lift a lot of the viewing/logging code up out of the pipeline itself. This should
    be mostly concerned with viewer and logging, although it will control the pipeline's
    training at a high level of abstraction

    Args:
        config (Config): The configuration object.
        local_rank (int, optional): Local rank of the process. Defaults to 0.
        world_size (int, optional): World size of the process. Defaults to 1.
    """

    def __init__(self, pipeline: Pipeline, model_dir, steps_per_log=20, steps_per_save=20, steps_per_test=20):
        self.pipeline: Pipeline = pipeline
        self.model_dir = model_dir
        self.steps_per_log = steps_per_log
        self.steps_per_save = steps_per_save
        self.steps_per_test = steps_per_test

    def run(self, num_iterations) -> None:
        """Train the model."""
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            for step in range(num_iterations):
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as t:
                    loss_metric_dict = self.pipeline.get_train_loss_dict()
                    self.pipeline.optimize_step(loss_metric_dict)

                if step != 0 and step % self.steps_per_log == 0:
                    writer.put_dict(name="Loss/train-loss_dict", scalar_dict=loss_metric_dict, step=step)
                if step != 0 and self.steps_per_save and step % self.steps_per_save == 0:
                    self.pipeline.save_checkpoint(self.model_dir, step)
                self._write_out_storage(step)
        self._write_out_storage(num_iterations)

    def _write_out_storage(self, step: int) -> None:
        """Perform writes only during appropriate time steps

        Args:
            step (int): Current training step.
        """
        if (
            step % self.steps_per_log == 0
            or (self.steps_per_save and step % self.steps_per_save == 0)
            or step % self.steps_per_test == 0
        ):
            writer.write_out_storage()
