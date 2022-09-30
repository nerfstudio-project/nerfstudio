# Custom dataset formats

We've implemented many common datasets inside the `nerfstudio/format/<dataset_format>.py` files. If you have a posed image dataset that does not match these existing formats can create a custom dataset format.

1. Create a function `nerfstudio/format/<dataset_format>.py` and implements a function called `load_<dataset_format>_data` returns a `DatasetInputs` object.

```python
# The DatasetInputs dataclass that is passed around to create datasets and initialize the graphs.
# See `nerfstudio/data/structs.py` for the code.
@dataclass
class DatasetInputs:
    """Dataset inputs are used to initialize datasets and the NeRF graph."""

    image_filenames: List[str]
    intrinsics: torch.tensor = None
    camera_to_world: torch.tensor = None
    mask_filenames: List[str] = None
    depth_filenames: List[str] = None
    scene_bounds: SceneBounds = SceneBounds()
    semantics: Semantics = Semantics()
    point_cloud: PointCloud = PointCloud()
    alpha_color: Optional[TensorType[3]] = None

# The method signature for the `instant_ngp` dataset format.
# See `nerfstudio/data/format/instant_ngp.py` for the code.
def load_instant_ngp_data(
    basedir: str,
    split: str = "train", camera_translation_scalar=0.33
) -> DatasetInputs:
    """Returns a DatasetInputs struct."""
```

2. Add a new case to `get_dataset_inputs()` in `nerfstudio/data/utils.py` for the new dataset.

```python
# The method signature for get_dataset_inputs.
# Notice that the parameters match the yaml config parameters above.
# See `nerfstudio/data/utils.py` for the code.
def get_dataset_inputs(
    data_directory: str,
    dataset_format: str,
    split: str,
    alpha_color: Optional[Union[str, list, ListConfig]] = None,
) -> DatasetInputs:
    """Makes a call to `load_<dataset_format>_data` and returns a DatasetInputs struct."""
```
