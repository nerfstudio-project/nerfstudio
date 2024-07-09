# Methods

We provide a set of pre implemented nerfstudio methods.

**The goal of nerfstudio is to modularize the various NeRF techniques as much as possible.**

As a result, many of the techniques from these pre-implemented methods can be mixed 🎨.

## Running a method

It's easy!

```bash
ns-train {METHOD_NAME}
```

To list the available methods run:

```bash
ns-train --help
```

## Implemented Methods

The following methods are supported in nerfstudio:

```{toctree}
    :maxdepth: 1
    Instant-NGP<instant_ngp.md>
    Splatfacto<splat.md>
    Splatfacto-W<splatw.md>
    Instruct-NeRF2NeRF<in2n.md>
    Instruct-GS2GS<igs2gs.md>
    SIGNeRF<signerf.md>
    K-Planes<kplanes.md>
    LERF<lerf.md>
    Mip-NeRF<mipnerf.md>
    NeRF<nerf.md>
    Nerfacto<nerfacto.md>
    Nerfbusters<nerfbusters.md>
    NeRFPlayer<nerfplayer.md>
    Tetra-NeRF<tetranerf.md>
    TensoRF<tensorf.md>
    Generfacto<generfacto.md>
    PyNeRF<pynerf.md>
    SeaThru-NeRF<seathru_nerf.md>
    Zip-NeRF<zipnerf.md>
    BioNeRF<bionerf.md>
    NeRFtoGSandBack<nerf2gs2nerf.md>
    OpenNeRF<opennerf.md>
```

(own_method_docs)=

## Adding Your Own Method

If you're a researcher looking to develop new NeRF-related methods, we hope that you find nerfstudio to be a useful tool. We've provided documentation about integrating with the nerfstudio codebase, which you can find [here](../../developer_guides/new_methods.md).

We also welcome additions to the list of methods above. To do this, simply create a pull request with the following changes,

1. Add a markdown file describing the model to the `docs/nerfology/methods` folder
2. Update the above list of implement methods in this file.
3. Add the method to {ref}`this<third_party_methods>` list in `docs/index.md`.
4. Add a new `ExternalMethod` entry to the `nerfstudio/configs/external_methods.py` file.

For the method description, please refer to the [Instruct-NeRF2NeRF](in2n) page as an example of the layout. Please try to include the following information:

- Installation instructions
- Instructions for running the method
- Requirements (dataset, GPU, ect)
- Method description (the more detailed the better, treat it like a blog post)

You are welcome to include assets (such as images or video) in the description, but please host them elsewhere.

:::{admonition} Note
:class: note

Please ensure that the documentation is clear and easy to understand for other users who may want to try out your method.
:::
