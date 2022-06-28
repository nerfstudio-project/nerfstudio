
# Custom Data Formats

In this page, we explain how to use pyRad with your own data. We've implemented many common datasets inside the `pyrad/format/<dataset_format>.py` files. Each of these files implements a function called `load_<dataset_format>_data`. Each of these functions returns a `DatasetInputs` object. Furthermore, `get_dataset_inputs` in `pyrad/data/utils.py` will have to be updated to handle the new function.

```python
# The DatasetInputs dataclass that is passed around to create datasets and initialize the graphs.
# See `pyrad/data/structs.py` for the code.
@dataclass
class DatasetInputs:
    """Dataset inputs are used to initialize datasets and the NeRF graph."""

    image_filenames: List[str]
    downscale_factor: int = 1
    intrinsics: torch.tensor = None
    camera_to_world: torch.tensor = None
    mask_filenames: List[str] = None
    depth_filenames: List[str] = None
    scene_bounds: SceneBounds = SceneBounds()
    semantics: Semantics = Semantics()
    point_cloud: PointCloud = PointCloud()
    alpha_color: Optional[TensorType[3]] = None

# The method signature for the `instant_ngp` dataset format.
# See `pyrad/data/format/instant_ngp.py` for the code.
def load_instant_ngp_data(
    basedir: str,
    downscale_factor: int = 1,
    split: str = "train", camera_translation_scalar=0.33
) -> DatasetInputs:
    """Returns a DatasetInputs struct."""

# The method signature for get_dataset_inputs.
# Notice that the parameters match the yaml config parameters above.
# See `pyrad/data/utils.py` for the code.
def get_dataset_inputs(
    data_directory: str,
    dataset_format: str,
    split: str,
    downscale_factor: int = 1,
    alpha_color: Optional[Union[str, list, ListConfig]] = None,
) -> DatasetInputs:
    """Makes a call to `load_<dataset_format>_data` and returns a DatasetInputs struct."""
```
