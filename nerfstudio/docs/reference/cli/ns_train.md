# ns-train

Primary interface for training a NeRF model. `--help` is your friend when navigating command arguments. We also recommend installing the tab completion `ns-install-cli`.

```bash
usage: ns-train {method} [method args] {dataparser} [dataparser args]
```

If you are using a nerfstudio data set, the minimal command is:

```bash
ns-train nerfacto --data YOUR_DATA
```

To learn about the available methods:

```bash
ns-train --help
```

To learn about a methods parameters:

```bash
ns-train {method} --help
```

By default the nerfstudio dataparser is used. If you would like to use a different dataparser it can be specified after all of the method arguments. For a list of dataparser options:

```bash
ns-train {method} {dataparser} --help
```

For example, if you want to specify the `eval_mode` of the nerfstudio dataparser to be `filename` when training your `splatfacto` model via the `ns-train` cli, you can do

```
ns-train splatfacto [method args] nerfstudio-data --eval-mode filename
```

Notice that the custom dataparser and its arguments are passed after specifying the training method and its arguments.
