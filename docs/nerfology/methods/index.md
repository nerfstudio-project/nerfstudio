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

## Methods

The following methods are supported in nerfstudio:

```{toctree}
    :maxdepth: 1
    NeRF<nerf.md>
    Mip-NeRF<mipnerf.md>
    Nerfacto<nerfacto.md>
    Instant-NGP<instant_ngp.md>
    Instruct-NeRF2NeRF<in2n.md>
    Semantic NeRF-W<semantic_nerfw.md>
```
