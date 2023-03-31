# Semantic NeRF-W

<h4>Semantic NeRF</h4>

```{button-link} https://shuaifengzhi.com/Semantic-NeRF/
:color: primary
:outline:
Paper Website
```

<h4>NeRF in the Wild</h4>

```{button-link} https://nerf-w.github.io/
:color: primary
:outline:
Paper Website
```

```{admonition} Coming Soon
The transient embeddings are still under development. Please stay tuned.
```

### Running Model

Download the sitcoms3D Dataset

```bash
ns-download-data sitcoms3d
```

```bash
ns-train semantic-nerfw
```

This model defaults to using the "sitcoms3D" dataset from the paper ["The One Where They Reconstructed 3D Humans and Environments in TV Shows"](https://ethanweber.me/sitcoms3D/).

<video src="https://ethanweber.me/sitcoms3D/media/trimmed_from_supplementary/sfm_and_nerf.mp4" width=300></video>
