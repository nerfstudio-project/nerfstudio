---
myst:
  substitutions:
    luma: |
      ```{image} _static/imgs/luma_light.png
      :alt: Luma AI
      :width: 300px
      :class: only-light
      :target: https://lumalabs.ai/
      ```

      ```{image} _static/imgs/luma_dark.png
      :alt: Luma AI
      :width: 300px
      :class: only-dark
      :target: https://lumalabs.ai/
      ```
    bair: |
      ```{image} _static/imgs/bair_light.png
      :alt: BAIR
      :width: 300px
      :class: only-light
      :target: https://bcommons.berkeley.edu/home
      ```

      ```{image} _static/imgs/bair_dark.png
      :alt: BAIR
      :width: 300px
      :class: only-dark
      :target: https://bcommons.berkeley.edu/home
      ```
---

```{eval-rst}
:og:description: Nerfstudio Documentation
:og:image: https://assets.nerf.studio/opg.png
```

<br/>

```{image} _static/imgs/logo.png
:width: 400
:align: center
:alt: nerfstudio
:class: only-light
```

```{image} _static/imgs/logo-dark.png
:width: 400
:align: center
:alt: nerfstudio
:class: only-dark
```

<br/>

<img src="https://user-images.githubusercontent.com/3310961/194017985-ade69503-9d68-46a2-b518-2db1a012f090.gif" width="52%"/> <img src="https://user-images.githubusercontent.com/3310961/194020648-7e5f380c-15ca-461d-8c1c-20beb586defe.gif" width="46%"/>

<br/>

Nerfstudio provides a simple API that allows for a simplified end-to-end process of creating, training, and testing NeRFs.
The library supports a **more interpretable implementation of NeRFs by modularizing each component.**
With more modular NeRFs, we hope to create a more user-friendly experience in exploring the technology.

This is a contributor-friendly repo with the goal of building a community where users can more easily build upon each other's contributions.
Nerfstudio initially launched as an opensource project by Berkeley students in [KAIR lab](https://people.eecs.berkeley.edu/~kanazawa/index.html#kair) at [Berkeley AI Research (BAIR)](https://bair.berkeley.edu/) in October 2022 as a part of a research project ([paper](https://arxiv.org/abs/2302.04264)). It is currently developed by Berkeley students and community contributors.

We are committed to providing learning resources to help you understand the basics of (if you're just getting started), and keep up-to-date with (if you're a seasoned veteran) all things NeRF. As researchers, we know just how hard it is to get onboarded with this next-gen technology. So we're here to help with tutorials, documentation, and more!

Have feature requests? Want to add your brand-spankin'-new NeRF model? Have a new dataset? **We welcome [contributions](reference/contributing)!**
Please do not hesitate to reach out to the nerfstudio team with any questions via [Discord](https://discord.gg/uMbNqcraFc).

Have feedback? We'd love for you to fill out our [Nerfstudio Feedback Form](https://forms.gle/sqN5phJN7LfQVwnP9) if you want to let us know who you are, why you are interested in Nerfstudio, or provide any feedback!

We hope nerfstudio enables you to build faster 🔨 learn together 📚 and contribute to our NeRF community 💖.

## Contents

```{toctree}
:hidden:
:caption: Getting Started

quickstart/installation
quickstart/first_nerf
quickstart/existing_dataset
quickstart/custom_dataset
quickstart/viewer_quickstart
quickstart/export_geometry
quickstart/data_conventions
Contributing<reference/contributing>
```

```{toctree}
:hidden:
:caption: Extensions
extensions/blender_addon
extensions/unreal_engine
extensions/sdfstudio
```

```{toctree}
:hidden:
:caption: NeRFology

nerfology/methods/index
nerfology/model_components/index
```

```{toctree}
:hidden:
:caption: Developer Guides

developer_guides/new_methods
developer_guides/pipelines/index
developer_guides/viewer/index
developer_guides/config
developer_guides/debugging_tools/index
```

```{toctree}
:hidden:
:caption: Reference

reference/cli/index
reference/api/index
```

This documentation is organized into 3 parts:

- **🏃‍♀️ Getting Started**: a great place to start if you are new to nerfstudio. Contains a quick tour, installation, and an overview of the core structures that will allow you to get up and running with nerfstudio.
- **🧪 Nerfology**: want to learn more about the tech itself? We're here to help with our educational guides. We've provided some interactive notebooks that walk you through what each component is all about.
- **🤓 Developer Guides**: describe all of the components and additional support we provide to help you construct, train, and debug your NeRFs. Learn how to set up a model pipeline, use the viewer, create a custom config, and more.
- **📚 Reference**: describes each class and function. Develop a better understanding of the core of our technology and terminology. This section includes descriptions of each module and component in the codebase.

## Supported Methods

### Included Methods

- [**Nerfacto**](nerfology/methods/nerfacto.md): Recommended method, integrates multiple methods into one.
- [Instant-NGP](nerfology/methods/instant_ngp.md): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
- [NeRF](nerfology/methods/nerf.md): OG Neural Radiance Fields
- [Mip-NeRF](nerfology/methods/mipnerf.md): A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
- [TensoRF](nerfology/methods/tensorf.md): Tensorial Radiance Fields
- [Splatfacto](nerfology/methods/splat.md): Nerfstudio's Gaussian Splatting implementation

(third_party_methods)=

### Third-party Methods

- [BioNeRF](nerfology/methods/bionerf.md): Biologically Plausible Neural Radiance Fields for View Synthesis
- [Instruct-NeRF2NeRF](nerfology/methods/in2n.md): Editing 3D Scenes with Instructions
- [Instruct-GS2GS](nerfology/methods/igs2gs.md): Editing 3DGS Scenes with Instructions
- [SIGNeRF](nerfology/methods/signerf.md): Controlled Generative Editing of NeRF Scenes
- [K-Planes](nerfology/methods/kplanes.md): Unified 3D and 4D Radiance Fields
- [LERF](nerfology/methods/lerf.md): Language Embedded Radiance Fields
- [Nerfbusters](nerfology/methods/nerfbusters.md): Removing Ghostly Artifacts from Casually Captured NeRFs
- [NeRFPlayer](nerfology/methods/nerfplayer.md): 4D Radiance Fields by Streaming Feature Channels
- [Tetra-NeRF](nerfology/methods/tetranerf.md): Representing Neural Radiance Fields Using Tetrahedra
- [PyNeRF](nerfology/methods/pynerf.md): Pyramidal Neural Radiance Fields
- [SeaThru-NeRF](nerfology/methods/seathru_nerf.md): Neural Radiance Field for subsea scenes
- [Zip-NeRF](nerfology/methods/zipnerf.md): Anti-Aliased Grid-Based Neural Radiance Fields

**Eager to contribute a method?** We'd love to see you use nerfstudio in implementing new (or even existing) methods! Please view our {ref}`guide<own_method_docs>` for more details about how to add to this list!

## Quicklinks

|                                                            |                             |
| ---------------------------------------------------------- | --------------------------- |
| [Github](https://github.com/nerfstudio-project/nerfstudio) | Official Github Repo        |
| [Discord](https://discord.gg/RyVk6w5WWP)                   | Join Discord Community      |
| [Feedback Form](https://forms.gle/sqN5phJN7LfQVwnP9)       | Provide Nerfstudio Feedback |

## Sponsors

Sponsors of this work includes [Luma AI](https://lumalabs.ai/) and the [BAIR commons](https://bcommons.berkeley.edu/home).

|          |          |
| -------- | -------- |
| {{luma}} | {{bair}} |

## Built On

```{image} https://brentyi.github.io/tyro/_static/logo-light.svg
:width: 150
:alt: tyro
:class: only-light
:target: https://github.com/brentyi/tyro
```

```{image} https://brentyi.github.io/tyro/_static/logo-dark.svg
:width: 150
:alt: tyro
:class: only-dark
:target: https://github.com/brentyi/tyro
```

- Easy to use config system
- Developed by [Brent Yi](https://brentyi.com/)

```{image} https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png
:width: 250
:alt: tyro
:class: only-light
:target: https://github.com/KAIR-BAIR/nerfacc
```

```{image} https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png
:width: 250
:alt: tyro
:class: only-dark
:target: https://github.com/KAIR-BAIR/nerfacc
```

- Library for accelerating NeRF renders
- Developed by [Ruilong Li](https://www.liruilong.cn/)

## Citation

You can find a paper writeup of the framework on [arXiv](https://arxiv.org/abs/2302.04264).

If you use this library or find the documentation useful for your research, please consider citing:

```none
@inproceedings{nerfstudio,
	title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
	author       = {
		Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
		and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
		Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
		Angjoo
	},
	year         = 2023,
	booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
	series       = {SIGGRAPH '23}
}
```

## Contributors

<a href="https://github.com/nerfstudio-project/nerfstudio/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nerfstudio-project/nerfstudio" />
</a>

## Maintainers

|                                                 | Nerfstudio Discord | Affiliation                          |
| ----------------------------------------------- | ------------------ | ------------------------------------ |
| [Justin Kerr](https://kerrj.github.io/)         | justin.kerr        | UC Berkeley                          |
| [Jonáš Kulhánek](https://jkulhanek.com/)        | jkulhanek          | Czech Technical University in Prague |
| [Matt Tancik](https://www.matthewtancik.com)    | tancik             | Luma AI                              |
| [Matias Turkulainen](https://maturk.github.io/) | maturk             | ETH Zurich                           |
| [Ethan Weber](https://ethanweber.me/)           | ethanweber         | UC Berkeley                          |
| [Brent Yi](https://github.com/brentyi)          | brent              | UC Berkeley                          |
