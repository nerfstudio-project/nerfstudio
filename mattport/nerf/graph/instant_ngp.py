"""
Implementation of Instant NGP.
"""

from mattport.nerf.field_modules.encoding import HashEncoding, SHEncoding
from mattport.nerf.graph.vanilla_nerf import NeRFField, NeRFGraph


class NGPGraph(NeRFGraph):
    """NeRF-W graph"""

    def __init__(self, intrinsics=None, camera_to_world=None, **kwargs) -> None:
        super().__init__(intrinsics=intrinsics, camera_to_world=camera_to_world, **kwargs)

    def populate_fields(self):
        """Set the fields."""

        position_encoding = HashEncoding()
        direction_encoding = SHEncoding()

        self.field_coarse = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=3,
            base_mlp_layer_width=64,
            head_mlp_num_layers=2,
            head_mlp_layer_width=32,
        )
        self.field_fine = NeRFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
            base_mlp_num_layers=3,
            base_mlp_layer_width=64,
            head_mlp_num_layers=2,
            head_mlp_layer_width=32,
        )
