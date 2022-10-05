# Models

```{image} imgs/pipeline_model-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_model-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a Model?

A Model is probably what you think of when you think of a NeRF paper. Often the phrases "Model" and "Method" are used interchangeably and for this reason, our implemented [Methods](/nerfology/methods/index) typically only change the model code.

A model, at a high level, takes in regions of space described by RayBundle objects, samples points along these rays, and returns rendered values for each ray. So, let's take a look at what it takes to create your own model!

## Functions to Implement

[The code](https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/models/base_model.py) is quite verbose, so here we distill the most important functions with succint descriptions.

```python
class Model:

    config: ModelConfig
    """Set the model config so that Python gives you typed autocomplete!"""

    def populate_modules(self):
        """Set the fields and modules."""

        # Fields

        # Ray Samplers

        # Colliders

        # Renderers

        # Losses

        # Metrics

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimizer your model components."""

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
        """Returns the training callbacks, such as updating a density grid for Instant NGP."""

    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with wandb or tensorboard."""

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
```

## Pythonic Configs with Models

Our config system is most useful when it comes to models. Let's take a look at our Nerfacto model config.

```python
@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["background", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""
    num_proposal_samples_per_ray: Tuple[int] = (64,)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    num_proposal_network_iterations: int = 1
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
```

There are a lot of options! Thankfully, our config system makes this easy to handle. If you want to add another argument, you simply add a value to this config and when you type in `ns-train nerfacto --help`, it will show in the terminal as a value you can modify.

Furthermore, you have Python autocomplete and static checking working in your favor. At the top of every Model, we specify the config and then can easily pull of values throughout the implementation. Let's take a look at the beginning of the NerfactoModel implementation.

```python
class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        ...
        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding, # notice self.config
        )
        ...
        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color) # notice self.config
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
```

We invite you to take a look at the Nerfacto model and others to see how our models are formatted.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/models/nerfacto.py
:color: primary
:outline:
See the code!
```

## Implementing a Model

Now that you understand how the model is structured, you can create a model by populating these functions. We provide a library of model components to pull from when creating your model. Check out those tutorials here!

One of these components is a Field, which you can learn more about in the next section. Fields associate a quantity of space with a value (e.g., density and color) and are used in every model.
