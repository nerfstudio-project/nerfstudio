from dataclasses import dataclass, field
from typing import Type

from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.data.datamanagers.dreamfusion_datamanager import (
    DreamFusionDataManagerConfig,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.generative.stable_diffusion import StableDiffusion
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler


@dataclass
class DreamfusionPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DreamfusionPipeline)
    """target class to instantiate"""
    datamanager: DreamFusionDataManagerConfig = DreamFusionDataManagerConfig()
    """specifies the datamanager config"""

class DreamfusionPipeline(VanillaPipeline):
    
    def __init__(
        self,
        config: DreamfusionPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(
            config,
            device,
            test_mode,
            world_size,
            local_rank
        )
        self.sd = StableDiffusion(device)
        self.text_embedding = self.sd.get_text_embeds("A high quality photo of a pineapple.", "")

    @profiler.time_function
    def custom_step(self, step: int, grad_scaler: GradScaler, optimizers: Optimizers):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle)

        # Just uses albedo for now
        albedo_output = model_outputs["rgb"].view(1, 64, 64, 3).permute(0, 3, 1, 2)

        sds_loss, latents, grad = self.sd.sds_loss(self.text_embedding, albedo_output)

        grad_scaler.scale(latents).backward(gradient=grad, retain_graph=True)
        optimizers.optimizer_scaler_step_all(grad_scaler)
        grad_scaler.update()
        optimizers.scheduler_step_all(step)

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        loss_dict["sds_loss"] = sds_loss

        return model_outputs, loss_dict, metrics_dict