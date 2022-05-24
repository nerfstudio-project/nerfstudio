#!/bin/bash

DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")
GPU_IDX=1

date
now=$(date)
tag=$(date +'%m-%d-%Y')

for dataset in ${DATASETS[@]}; do
    export CUDA_VISIBLE_DEVICES=${!GPU_IDX}
    python scripts/run_train.py \
           '~logging.writer.LocalWriter' \
           data.dataset.data_directory=data/blender/${dataset} \
           data.dataset.downscale_factor=1 \
           experiment_name=blender_${dataset}_${tag} \
           graph.model_dir=mattport_models/ \
           graph.steps_per_save=25000 \
           graph.max_num_iterations=2000000 \
           logging.enable_profiler=False &
    echo "Launched ${dataset} on gpu ${!GPU_IDX}, ${tag}"
    GPU_IDX=$((GPU_IDX+1))
done