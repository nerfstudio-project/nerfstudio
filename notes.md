# Notes for development

Download data for testing with

```bash
ns-download-data nerfstudio --capture-name aspen
```

Run the model normally.

```bash
ns-train nerfacto --vis viewer+wandb --data data/nerfstudio/aspen --pipeline.model.implementation nvidia
```

You can speed up the data loading initialization by downscaling the images. You simply append `nerfstudio-data --downscale-factor 8` to your command.

```bash
ns-train nerfacto --vis viewer+wandb --data data/nerfstudio/aspen --pipeline.model.implementation nvidia nerfstudio-data --downscale-factor 8
```

Run the model with pytorch (WIP - work in progress)

```bash
ns-train nerfacto --vis viewer+wandb --data data/nerfstudio/aspen --pipeline.model.implementation pytorch nerfstudio-data --downscale-factor 8
```
