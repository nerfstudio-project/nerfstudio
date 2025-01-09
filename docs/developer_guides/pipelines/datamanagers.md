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
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """specifies the camera pose optimizer used during training"""
```

Let's take a quick look at how the `next_train` method is implemented. Here we sample images, then pixels, and then return the RayBundle and RayGT information.

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

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/datamanagers/base_datamanager.py
:color: primary
:outline:
See the code!
```

## Creating Your Own

We currently don't have other implementations because most papers follow the VanillaDataManager implementation. However, it should be straightforward to add a VanillaDataManager with logic that progressively adds cameras, for instance, by relying on the step and modifying RayBundle and RayGT generation logic.

## Migrating Your Datamanager to the New Datamanager 

As of January 2025, the FullImageDatamanager and ParallelImageDatamanager implementation now supports parallelized dataloading and dataloading from disk to preserve CPU RAM. If you would like your custom datamanager to also support these new features, you can migrate any custom dataloading logic to the `custom_view_processor` API. Let's take a look at an example for the LERF method, which was built on Nerfstudio's VanillaDataManager. 

```python
class LERFDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: LERFDataManagerConfig

    def __init__(
        self,
        config: LERFDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.image_encoder: BaseImageEncoder = kwargs["image_encoder"]
        images = [self.train_dataset[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.train_dataset))]
        images = torch.cat(images)

        cache_dir = f"outputs/{self.config.dataparser.data.name}"
        clip_cache_path = Path(osp.join(cache_dir, f"clip_{self.image_encoder.name}"))
        dino_cache_path = Path(osp.join(cache_dir, "dino.npy"))
        # NOTE: cache config is sensitive to list vs. tuple, because it checks for dict equality
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=self.device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
        )
        torch.cuda.empty_cache()
        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=images,
            device=self.device,
            cfg={
                "tile_size_range": [0.05, 0.5],
                "tile_size_res": 7,
                "stride_scaler": 0.5,
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
        batch["dino"] = self.dino_dataloader(ray_indices)
        ray_bundle.metadata["clip_scales"] = clip_scale
        # assume all cameras have the same focal length and image width
        ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
        ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
        ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
        ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
        return ray_bundle, batch
```

To migrate this custom datamanager to the new datamanager, we can shift the data customization process in `next_train()` to `custom_view_processor()`.

```python
class LERFDataManager(ParallelDataManager, Generic[TDataset]):

    ...

    def custom_ray_processor(
            self, ray_bundle: RayBundle, batch: Dict
        ) -> Tuple[RayBundle, Dict]:
            """An API to add latents, metadata, or other further customization to the RayBundle dataloading process that is parallelized"""
            ray_indices = batch["indices"]
            batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
            batch["dino"] = self.dino_dataloader(ray_indices)
            ray_bundle.metadata["clip_scales"] = clip_scale
            # assume all cameras have the same focal length and image width
            ray_bundle.metadata["fx"] = self.train_dataset.cameras[0].fx.item()
            ray_bundle.metadata["width"] = self.train_dataset.cameras[0].width.item()
            ray_bundle.metadata["fy"] = self.train_dataset.cameras[0].fy.item()
            ray_bundle.metadata["height"] = self.train_dataset.cameras[0].height.item()
            return ray_bundle, batch
```