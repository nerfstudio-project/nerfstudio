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

Nerfstudio provides a simple API that allows for a simplified end-to-end process of creating, training, and visualizing NeRFs.
The library supports an **interpretable implementation of NeRFs by modularizing each component.**
With modular NeRF components, we hope to create a user-friendly experience in exploring the technology.
Nerfstudio is a contributor-friendly repo with the goal of building a community where users can easily build upon each other's contributions.

It's as simple as plug and play with nerfstudio!

On top of our API, we are committed to providing learning resources to help you understand the basics of (if you're just getting started), and keep up-to-date with (if you're a seasoned veteran) all things NeRF.
As researchers, we know just how hard it is to get onboarded with this next-gen technology. So we're here to help with tutorials, documentation, and more!

Finally, have feature requests? Want to add your brand-spankin'-new NeRF model? Have a new dataset? **We welcome any and all [contributions](reference/contributing)!**
Please do not hesitate to reach out to the nerfstudio team with any questions via [Discord](https://discord.gg/uMbNqcraFc).

We hope nerfstudio enables you to build faster 🔨 learn together 📚 and contribute to our NeRF community 💖.

## Contents

```{toctree}
:hidden:
:caption: Getting Started

quickstart/installation
quickstart/first_nerf
quickstart/custom_dataset
quickstart/viewer_quickstart
quickstart/export_geometry
quickstart/data_conventions
Contributing<reference/contributing>
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

developer_guides/pipelines/index
developer_guides/viewer/viewer_overview
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

- [**Nerfacto**](https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/models/nerfacto.py): our de facto NeRF method combines modules focused on quality with modules focused on faster rendering. Nerfstudio easily lets us experiment with the best of both worlds!
- [NeRF](https://www.matthewtancik.com/nerf): Representing Scenes as Neural Radiance Fields for View Synthesis
- [Instant NGP](https://nvlabs.github.io/instant-ngp/): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
- [Mipnerf](https://jonbarron.info/mipnerf/): A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
- [NerfW](https://nerf-w.github.io/): Neural Radiance Fields for Unconstrained Photo Collections
- [Semantic NeRF](https://shuaifengzhi.com/Semantic-NeRF/): In-Place Scene Labelling and Understanding with Implicit Scene Representation

We'll be constantly growing this list! So make sure to check back in to see our updates.

**Eager to contribute?** We'd love to see you use nerfstudio in implementing new (or even existing) methods! Feel free to contact us directly or view our [Contributor's Guide](reference/contributing) to see how you can get your model on this list!

## Quicklinks

|                                                            |                        |
| ---------------------------------------------------------- | ---------------------- |
| [Github](https://github.com/nerfstudio-project/nerfstudio) | Official Github Repo   |
| [Discord](https://discord.gg/RyVk6w5WWP)                   | Join Discord Community |
| [Viewer](https://viewer.nerf.studio/)                      | Web-based Nerf Viewer  |

### How-to Videos

|                                                                 |                                                           |
| --------------------------------------------------------------- | --------------------------------------------------------- |
| [Using the Viewer](https://www.youtube.com/watch?v=nSFsugarWzk) | Demo video on how to run nerfstudio and use the viewer.   |
| [Using Record3D](https://www.youtube.com/watch?v=XwKq7qDQCQk)   | Demo video on how to run nerfstudio without using COLMAP. |

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

If you use this library or find the documentation useful for your research, please consider citing:

```none
@misc{nerfstudio,
      title={Nerfstudio: A Framework for Neural Radiance Field Development},
      author={Matthew Tancik*, Ethan Weber*, Evonne Ng*, Ruilong Li, Brent Yi,
              Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi,
              Abhik Ahuja, David McAllister, Angjoo Kanazawa},
      year={2022},
      url={https://github.com/nerfstudio-project/nerfstudio},
}
```

## Contributors

<a href="https://github.com/nerfstudio-project/nerfstudio/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nerfstudio-project/nerfstudio" />
</a>
