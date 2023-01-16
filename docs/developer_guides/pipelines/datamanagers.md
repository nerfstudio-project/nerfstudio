# DataManagers

```{image} imgs/pipeline_datamanager-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_datamanager-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a DataManager?

The DataManager returns RayBundle and RayGT objects. Let's first take a look at the most important abstract methods required by the DataManager.

```python
class DataManager(nn.Module):
    """Generic data manager's abstract class
    """

    @abstractmethod
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data for train."""

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data for eval."""

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Returns the next eval image.

        Returns:
            The image index from the eval dataset, the CameraRayBundle, and the RayGT dictionary.
        """
```

## Example

We've implemented a VanillaDataManager that implements the standard logic of most NeRF papers. It will randomly sample training rays with corresponding ground truth information, in RayBundle and RayGT objects respectively. The config for the VanillaDataManager is the following.

```python
@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """target class to instantiate"""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """specifies the dataparser used to unpack the data"""
    train_num_rays_per_batch: int = 1024
    """number of rays per batch to use per training iteration"""
    train_num_images_to_sample_from: int = -1
    """number of images to sample during training iteration"""
    eval_num_rays_per_batch: int = 1024
    """number of rays per batch to use per eval iteration"""
    eval_num_images_to_sample_from: int = -1
    """number of images to sample during eval iteration"""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """specifies the image indices to use during eval; if None, uses all"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """specifies the camera pose optimizer used during training"""
```

Let's take a quick look at how the `run_train` method is implemented. Here we sample images, then pixels, and then return the RayBundle and RayGT information.

```python
def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
    """Returns the next batch of data from the train dataloader."""
    self.train_count += 1
    # sample a batch of images
    image_batch = next(self.iter_train_image_dataloader)
    # sample pixels from this batch of images
    batch = self.train_pixel_sampler.sample(image_batch)
    ray_indices = batch["indices"]
    # generate rays from this image and pixel indices
    ray_bundle = self.train_ray_generator(ray_indices)
    # return RayBundle and RayGT information
    return ray_bundle, batch
```

You can see our code for more details.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/data/datamanagers.py
:color: primary
:outline:
See the code!
```

## Creating Your Own

We currently don't have other implementations because most papers follow the VanillaDataManager implementation. However, it should be straightforward to add a VanillaDataManager with logic that progressively adds cameras, for instance, by relying on the step and modifying RayBundle and RayGT generation logic.
