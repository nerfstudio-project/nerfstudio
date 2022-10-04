# Methods

We provide a set of pre implemented nerfstudio methods. One of the goals of nerfstudio is to modularize the various NeRF techniques as much as possible. As a result, many of the techniques from these pre-implemented methods can be mixed.

## Running a method

It's easy!

```bash
ns-train METHOD_NAME
```

To list the available methods run:

```bash
ns-train --help
```

## Guides

In addition to their implementations, we have provided guides that walk through each of these method.

```{toctree}
    :maxdepth: 1
    NeRF<nerf.md>
    Mip-NeRF<mipnerf.md>
    Nerfacto<nerfacto.md>
    Instant-NGP<instant_ngp.md>
    Semantic NeRF-W<semantic_nerfw.md>
```
