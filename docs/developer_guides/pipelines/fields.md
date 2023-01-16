# Fields

```{image} imgs/pipeline_field-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_field-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a Field?

A Field is a model component that associates a region of space with some sort of quantity. In the most typical case, the input to a field is a 3D location and viewing direction, and the output is density and color. Let's take a look at the code.

```python
class Field(nn.Module):
    """Base class for fields."""

    @abstractmethod
    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType[..., 1], TensorType[..., "num_features"]]:
        """Computes and returns the densities. Returns a tensor of densities and a tensor of features.

        Args:
            ray_samples: Samples locations to compute density.
        """

    @abstractmethod
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)
        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)

        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        return field_outputs
```

## Separate density and outputs

The forward function is the main function you'll use, which takes in RaySamples returns quantities for each sample. You'll notice that the get_density function is called for every field, followed by the get_outputs function.

The get_outputs function is what you need to implement to return custom data. For example, check out of SemanticNerfField where we rely on different FieldHeads to produce correct dimensional outputs for typical quantities. Our implemented FieldHeads have the following FieldHeadNames names.

```python
class FieldHeadNames(Enum):
    """Possible field outputs"""

    RGB = "rgb"
    SH = "sh"
    DENSITY = "density"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS = "semantics"
```

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/field_components/field_heads.py
:color: primary
:outline:
See the code!
```

Sometimes all you need is the density from a Field, so we have a helper method called density_fn which takes positions and returns densities.

## Using Frustums instead of positions

Let's say you want to query a region of space, rather than a point. Our RaySamples data structure contains Frustums which can be used for exactly this purpose. This enables methods like Mip-NeRF to be implemented in our framework.

```python
@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    ...

@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: TensorType["bs":..., 3]
    """xyz coordinate for ray origin."""
    directions: TensorType["bs":..., 3]
    """Direction of ray."""
    starts: TensorType["bs":..., 1]
    """Where the frustum starts along a ray."""
    ends: TensorType["bs":..., 1]
    """Where the frustum ends along a ray."""
    pixel_area: TensorType["bs":..., 1]
    """Projected area of pixel a distance 1 away from origin."""
    ...
```

Take a look at our RaySamples class for more information on the input to our Field classes.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/fields/base_field.py
:color: primary
:outline:
See the code!
```
