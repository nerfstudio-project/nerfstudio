# Notes for development

Download data for testing with

```bash
ns-download-data nerfstudio --capture-name aspen
```

Run the model normally.

```bash
ns-train nerfacto --vis viewer+wandb --data data/nerfstudio/aspen --pipeline.model.implementation tcnn
```

You can speed up the data loading initialization by downscaling the images. You simply append `nerfstudio-data --downscale-factor 8` to your command.

```bash
ns-train nerfacto --vis viewer+wandb --experiment-name aspen-tcnn --data data/nerfstudio/aspen --pipeline.model.implementation tcnn nerfstudio-data --downscale-factor 8
```

Run the model with pytorch

```bash
ns-train nerfacto --vis viewer+wandb --experiment-name aspen-torch --data data/nerfstudio/aspen --pipeline.model.implementation torch nerfstudio-data --downscale-factor 8
```
