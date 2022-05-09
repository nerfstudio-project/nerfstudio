"""
NeRF-W (NeRF in the wild) implementation.
TODO:
"""

from mattport.nerf.graph.vanilla_nerf import NeRFGraph
from mattport.structures.rays import RayBundle


class NerfWGraph(NeRFGraph):
    """NeRF-W graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, **kwargs) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def get_outputs(self, ray_bundle: RayBundle):
        raise NotImplementedError

    def get_loss_dict(self, outputs, batch):
        raise NotImplementedError
