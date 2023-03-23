# from typing import Tuple, Type
# from torch.utils.data import Dataset

# from dataclasses import dataclass, field
# from nerfstudio.configs import base_config as cfg

# @dataclass
# class LERFDatasetConfig(cfg.InstantiateConfig):
#     _target: Type = field(default_factory=lambda: LERFDataset)
#     clip_tile_size_range: Tuple[float] = (0.05, 0.5)
#     clip_tile_size_res: int = 7
#     clip_stride_scaler: float = 0.5

# class LERFDataset(Dataset):
#     def __init__(self, config: LERFDatasetConfig):
#         pass

#     def __len__(self):
#         pass

#     def __getitem__(self, index):
#         # make this return "clip" "dino"
#         pass