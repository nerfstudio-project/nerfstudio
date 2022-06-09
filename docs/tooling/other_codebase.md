# Run other repos with our data

If you are looking to test out other existing repo's with our data, the pyRad structure makes it easy!

For instance, you can run nerf-pytorch and jaxnerf with the following commands:

```bash
# nerf-pytorch
cd external
python run_nerf.py --config configs/chair.txt --datadir /path/to/pyrad/data/blender/chair

# jaxnerf
cd external
conda activate jaxnerf
python -m jaxnerf.train --data_dir=/path/to/pyrad/data/blender/chair --train_dir=/path/to/pyrad/outputs/blender_chair_jaxnerf --config=/path/to/pyrad/external/jaxnerf/configs/demo --render_every 100
```