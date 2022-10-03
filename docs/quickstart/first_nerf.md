# Training First Model

## Downloading data

Download the original NeRF Blender dataset. We support the major datasets and allow users to create their own dataset, described in detail [here TODO].

```
ns-download-data --dataset=blender
ns-download-data --dataset=nerfstudio --capture=poster
```

Use `--help` to view all currently available datasets. The resulting script should download and unpack the dataset as follows:

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

Run with nerfstudio data. You'll may have to change the ports, and be sure to forward the "websocket-port".

```
ns-train nerfacto --vis viewer --viewer.zmq-port 8001 --viewer.websocket-port 8002 nerfstudio-data --pipeline.datamanager.dataparser.data-directory data/nerfstudio/poster --pipeline.datamanager.dataparser.downscale-factor 4
```

## Visualizing training runs

If you using a fast NeRF variant (ie. Instant-NGP), we reccomend using our viewer. See our [viewer docs](viewer_quickstart.md) for more details. The viewer will allow interactive visualization of training in realtime.

Additionally, if you run everything with the default configuration, by default, we use [TensorBoard](https://www.tensorflow.org/tensorboard) to log all training curves, test images, and other stats. Once the job is launched, you will be able to track training by launching the tensorboard in `outputs/blender_lego/vanilla_nerf/<timestamp>/<events.tfevents>`.

```bash
tensorboard --logdir outputs
```

## Rendering a Trajectory

To evaluate the trained NeRF, we provide an evaluation script that allows you to do benchmarking (see our [benchmarking workflow](../developer_guides/benchmarking.md)) or to render out the scene with a custom trajectory and save the output to a video.

```bash
ns-eval render-trajectory --load-config=outputs/blender_lego/instant_ngp/2022-07-07_230905/config.yml --traj=spiral --output-path=output.mp4
```

Please note, this quickstart allows you to preform everything in a headless manner. We also provide a web-based viewer that allows you to easily monitor training or render out trajectories. See our [viewer docs](viewer_quickstart.md) for more.

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
