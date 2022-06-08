# Speeding up code with TinyCuda

Documentation for running the code with CUDA.
Please see https://github.com/NVlabs/tiny-cuda-nn for how to install tiny-cuda-nn.

```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

To run instant-ngp with tcnn, you can do the following. This is with the fox dataset.

```
python scripts/run_train.py --config-name=instant_ngp_tcnn.yaml data/dataset=instant_ngp_fox
```