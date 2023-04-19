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
    NeRF<nerf.md>
    Mip-NeRF<mipnerf.md>
    Nerfacto<nerfacto.md>
    Instant-NGP<instant_ngp.md>
    Instruct-NeRF2NeRF<in2n.md>
    LERF<lerf.md>
    Semantic NeRF-W<semantic_nerfw.md>
```

(own_method_docs)=

## Adding Your Own Method

If you're a researcher looking to develop new NeRF-related methods, we hope that you find nerfstudio to be a useful tool. We've provided documentation about integrating with the nerfstudio codebase, which you can find [here](../../developer_guides/new_methods.md).

We also welcome additions to the list of methods above. To do this, simply create a pull request that adds a markdown file describing the model to the docs/nerfology/methods folder, and update the list in this file. For reference on the layout, you can check out the [Instruct-NeRF2NeRF](in2n) page. Please try to include the following information:

- Installation instructions
- Instructions for running the method
- Requirements (dataset, GPU, ect)
- Method description (the more detailed the better, treat it like a blog post)

You are welcome to include assets (such as images or video) in the description, but please host them elsewhere.

:::{admonition} Note
:class: note

Please ensure that the documentation is clear and easy to understand for other users who may want to try out your method.
:::
