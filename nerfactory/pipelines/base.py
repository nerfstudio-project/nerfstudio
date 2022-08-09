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
Abstracts for the Pipeline class.
"""

from abc import abstractmethod

from nerfactory.dataloaders.base import Dataloader, setup_dataloader
from nerfactory.models.base import Model, setup_model
from nerfactory.utils.config import PipelineConfig
from nerfactory.utils import profiler
from pydoc import locate


class Pipeline:
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.

    This class will contain high level functions for the model like getting the loss
    dictionaries and visualization code. It should have ways to get the next iterations
    training loss, evaluation loss, and generate whole images for visualization. Each model
    class should be 1:1 with a pipeline that can act as a standardized interface and hide
    differences in how each model takes in and outputs data.

    This class's function is to hide the dataloader and model classes from the trainer,
    worrying about:
    1) Fetching data with the dataloader
    2) Feeding the model the data and fetching the loss
    3) (TODO) Visualization
    Hopefully this provides a higher level interface for the trainer to use, and
    simplifying the model classes, which each may have different forward() methods
    and so on.


    TODO: For visualizer functionality to be added down the line, we should make sure
    that we are still abstracting away the Model from the end user. The visualizer function
    should probably be done by adding a new iterator on the base dataloader that will
    talk to the actual visualizer. This is probably ideal to have the visualizer be
    primarily located in the dataloader (first because it makes sense as the
    visualizers main job in this context is to feed in data for the model to load)
    so that we can have an easier time ensuring that the visualizer is always
    spitting out the same formatted data as for in train / eval.


    Args:
        model: The model to be used in the pipeline.
        dataloader: The dataloader to be used in the pipeline.
        loss_coefficients: A dictionary of loss coefficients that will be used

    Attributes:
        self.dataloader (Dataloader): The dataloader that will be used
        self.dataloader_train_iter (Iterator): The iterator for the training dataloader
        self.dataloader_eval_iter (Iterator): The iterator for the evaluation dataloader
        self.model (Model): The model that will be used
    """

    def __init__(self, dataloader: Dataloader, model: Model):
        self.dataloader: Dataloader = dataloader
        self.dataloader_train_iter = self.dataloader.iter_train()
        self.dataloader_eval_iter = self.dataloader.iter_eval()
        self.model: Model = model

    @abstractmethod
    def get_train_loss_dict(self):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the dataloader and interfacing with the
        Model class, feeding the data to the model's forward function."""
        rays, batch = self.dataloader_train_iter.next()
        accumulated_color, _, _, mask = self.model(rays, batch)
        masked_batch = get_masked_dict(batch, mask)
        loss_dict = self.model.get_loss_dict(accumulated_color, masked_batch, mask)
        return loss_dict

    @abstractmethod
    def get_eval_loss_dict(self):
        """This function gets your evaluation loss dict. It needs to get the data
        from the dataloader and feed it to the model's forward function"""
        rays, batch = self.dataloader_eval_iter.next()
        accumulated_color, _, _, mask = self.model(rays, batch)
        masked_batch = get_masked_dict(batch, mask)
        loss_dict = self.model.get_loss_dict(accumulated_color, masked_batch, mask)
        return loss_dict

    @abstractmethod
    def log_test_image_outputs(self) -> None:
        """Log the test image outputs"""

    def load_pipeline(self):
        """Restore state of the pipeline from a checkpoint."""
        self.dataloader.load_dataloader()
        self.model.load_model()


@profiler.time_function
def setup_pipeline(config: PipelineConfig, device: str) -> Pipeline:
    """Setup the pipeline. The dataset inputs should be set with the training data.

    Args:
        config: The pipeline config.
    """
    # dataset_inputs
    dataloader: Dataloader = setup_dataloader(config.dataloader, device=device)
    # TODO(ethan): get rid of scene_bounds from the model
    model: Model = setup_model(config.model, scene_bounds=dataloader.train_datasetinputs.scene_bounds, device=device)
    pipeline_class = locate(config._target_)  # pylint: disable=protected-access
    pipeline = pipeline_class(dataloader=dataloader, model=model)  # type: ignore
    return pipeline
