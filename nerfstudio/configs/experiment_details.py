import warnings
from dataclasses import dataclass
from typing import Optional

from nerfstudio.nerfstudio.configs.base_config import PrintableConfig

warnings.filterwarnings("ignore", module="torchvision")
@dataclass
class InferredExperimentDetails(PrintableConfig):
    """Experiment fields which are generated after the static configuration is provided. These fields can't be overriden beacuse they depend on other config fields."""
    data_size: Optional[int] = None
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    split_ratio: Optional[float] = None
    train_images: Optional[list[str]] = None
    val_images: Optional[list[str]] = None
    
