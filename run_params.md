# Parameters Used to 

### train the helical 
    
ns-train instant-ngp-bounded \
--load-dir /workspaces/nerfstudio/outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-01_021655/nerfstudio_models \
--viewer.skip-openrelay True \
--viewer.websocket-port 7008 \
--viewer.start-train False \
--viewer.max-num-display-images 100 \
--pipeline.datamanager.train-num-rays-per-batch 65536 \
--pipeline.model.cone-angle 0.0 \
--pipeline.model.grid-resolution 256 \
--machine.num-gpus 1 \
instant-ngp-data \
--data /workspaces/nerfstudio/data/dataset_hel_4k_train \
--scene-scale 2.0

### make helical render
``` bash
ns-render \
--load-config outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-05_172157/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_hel_4k_train/camera_paths/2023-03-05_172157.json \
--output-path renders/dataset_hel_4k_train/hel_vid__depth_.mp4 \
--rendered-output-names depth
```

```bash
ns-render --load-config outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-05_172157/config.yml --traj filename --camera-path-filename /workspaces/nerfstudio/data/dataset_hel_4k_train/camera_paths/2023-03-05_172157.json --output-path renders/dataset_hel_4k_train/2023-03-05_172157.mp4
```


### make helical orbit to compare 
``` bash
ns-render \
--load-config /workspaces/nerfstudio/outputs/-workspaces-nerfstudio-data-dataset_hel_4k_train/instant-ngp-bounded/2023-03-01_021655/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_flyby_4k_train/camera_paths/test_path.json \
--output-path renders/dataset_flyby_4k_train/hel_compare_flyby_depth_4.mp4 \
--rendered-output-names depth
```

## Flyby

### train the flyby
``` bash   
ns-train instant-ngp-bounded \
--load-dir /workspaces/nerfstudio/outputs/-workspaces-nerfstudio-data-dataset_flyby_4k_train/instant-ngp-bounded/2023-03-01_213857/nerfstudio_models \
--viewer.skip-openrelay True \
--viewer.websocket-port 7008 \
--viewer.start-train False \
--viewer.max-num-display-images 50 \
--pipeline.datamanager.train-num-rays-per-batch 65536 \
--pipeline.model.cone-angle 0.0 \
--pipeline.model.grid-resolution 128 \
--machine.num-gpus 1 \
instant-ngp-data \
--data /workspaces/nerfstudio/data/dataset_flyby_4k_train \
--scene-scale 2.0
```
``` bash
ns-render \
--load-config outputs/-workspaces-nerfstudio-data-dataset_flyby_4k_train/instant-ngp-bounded/2023-03-01_014552/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_flyby_4k_train/camera_paths/conpare_flyby.json \
--output-path renders/dataset_flyby_4k_train/conpare_flyby.mp4
```

``` bash
ns-render \
--load-config outputs/-workspaces-nerfstudio-data-dataset_flyby_4k_train/instant-ngp-bounded/2023-03-02_211158/config.yml \
--traj filename \
--camera-path-filename /workspaces/nerfstudio/data/dataset_flyby_4k_train/camera_paths/test_path.json \
--output-path renders/dataset_flyby_4k_train/compare_flyby_4.mp4
```
[flyby_compare_out]: renders/dataset_flyby_4k_train/compare_flyby_3.mp4
