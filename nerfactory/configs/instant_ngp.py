@dataclass
class InstantNGPPipelineConfig(PipelineConfig):
    """Configuration for pipeline instantiation"""

    from nerfactory.pipelines import base

    _target: ClassVar[Type] = base.Pipeline
    dataloader: DataloaderConfig = BlenderDataloaderConfig()
    model: ModelConfig = InstantNGPPModelConfig()


@dataclass
class InstantNGPPConfig(Config):
    method_name: str = "instant_ngp"
    pipeline: PipelineConfig = InstantNGPPipelineConfig()
