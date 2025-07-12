# Viewer

> We have a real-time web viewer that requires no installation. It's available at [https://viewer.nerf.studio/](https://viewer.nerf.studio/), where you can connect to your training job.

The viewer is built on [Viser](https://github.com/nerfstudio-project/viser) using [ThreeJS](https://threejs.org/) and packaged into a [ReactJS](https://reactjs.org/) application. This client viewer application will connect via a websocket to a server running on your machine.

```{toctree}
:titlesonly:

custom_gui
viewer_control
local_viewer
```

## Acknowledgements and references

We thank the authors and contributors to the following repos, which we've started, used, and modified for our use-cases.

- [Viser](https://github.com/brentyi/viser/) - made by [Brent Yi](https://github.com/brentyi)
- [meshcat-python](https://github.com/rdeits/meshcat-python) - made by [Robin Deits](https://github.com/rdeits)
- [meshcat](https://github.com/rdeits/meshcat) - made by [Robin Deits](https://github.com/rdeits)
- [ThreeJS](https://threejs.org/)
- [ReactJS](https://reactjs.org/)
