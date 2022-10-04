# Training First Model

## Downloading data

Download the original NeRF Blender dataset. We support the major datasets and allow users to create their own dataset, described in detail [here TODO].

```
ns-download-data --dataset=blender
ns-download-data --dataset=nerfstudio --capture=poster
```

:::{admonition} Tip
:class: info

Use `ns-download-data --help` to view all currently available datasets.
  :::

The resulting script should download and unpack the dataset as follows:

```
|─ nerfstudio/
   ├─ data/
   |  ├─ blender/
   |     ├─ fern/
   |     ├─ lego/
         ...
      |- <dataset_format>/
         |- <scene>
         ...
```

## Training a model

To run with all the defaults, e.g. vanilla nerf method with the blender lego image

Run a vanilla nerf model.

```bash
ns-train vanilla-nerf
```

Run a nerfacto model.

```bash
ns-train nerfacto
```

Run with nerfstudio data with our web-based viewer enabled. 
```
ns-train nerfacto --vis viewer --viewer.zmq-port 8001 --viewer.websocket-port 8002 nerfstudio-data --pipeline.datamanager.dataparser.data-directory data/nerfstudio/poster --pipeline.datamanager.dataparser.downscale-factor 4
```

:::{admonition} Tip
:class: info

* You may have to change the ports, and be sure to forward the "websocket-port".
* All data configurations must go at the end. In this case, `nerfstudio-data` and all of its corresponding configurations come at the end after the model and viewer specification.
  :::

## Visualizing training runs

If you using a fast NeRF variant (ie. Nerfacto/Instant-NGP), we reccomend using our viewer. See our [viewer docs](viewer_quickstart.md) for a tutorial on using the viewer. The viewer will allow interactive, real-time visualization of training.

We provide default `--vis` options depending on the model. For all other slower methods for which the viewer is not recommended, we default to [Wandb](https://wandb.ai/site) to log all training curves, test images, and other stats. 

:::{admonition} Note
:class: info

Currently we only support using a single viewer at a time. To toggle between Wandb, Tensorboard, or our Web-based Viewer, you can specify `--vis VIS_OPTION`, where VIS_OPTION is one of {viewer,wandb,tensorboard}.
  :::

## Rendering a Trajectory

To evaluate the trained NeRF, we provide an evaluation script that allows you to do benchmarking (see our [benchmarking workflow](../developer_guides/benchmarking.md)).

We also provide options to render out the scene with a custom trajectory and save the output to a video.

```bash
ns-render --load-config={PATH_TO_CONFIG} --traj=spiral --output-path=output.mp4
```

:::{admonition} See Also
:class: info
This quickstart allows you to preform everything in a headless manner. We also provide a web-based viewer that allows you to easily monitor training or render out trajectories. See our [viewer docs](viewer_quickstart.md) for more.
  :::

## FAQ

- [TinyCUDA installation errors out with cuda mismatch](tiny-cuda-error)
- [Installation errors, File "setup.py" not found](pip-install-error)
- [Runtime errors, "len(sources) > 0".](cuda-sources-error)

(tiny-cuda-error)=

#### TinyCUDA installation errors out with cuda mismatch

While installing tiny-cuda, you run into: `The detected CUDA version mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.`

**Solution**:

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

(pip-install-error)=

#### Installation errors, File "setup.py" not found

When installing dependencies and nerfstudio with `pip install -e .`, you run into: `ERROR: File "setup.py" not found. Directory cannot be installed in editable mode`

**Solution**:
This can be fixed by upgrading pip to the latest version:

```
python -m pip install --upgrade pip
```


(cuda-sources-error)=

#### Runtime errors: "len(sources) > 0", "ctype = \_C.ContractionType(type.value) ; TypeError: 'NoneType' object is not callable".

When running `train.py `, an error occurs when installing cuda files in the backend code.

**Solution**:
This is a problem with not being able to detect the correct CUDA version, and can be fixed by updating the CUDA path environment variables:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
