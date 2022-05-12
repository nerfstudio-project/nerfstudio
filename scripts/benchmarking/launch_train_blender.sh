#!/bin/bash

DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")
GPU_IDX=1
for dataset in ${DATASETS[@]}; do
    export CUDA_VISIBLE_DEVICES=${!GPU_IDX}
    python scripts/run_train.py \
           data.dataset.data_directory=data/blender/${dataset} \
           experiment_name=blender_${dataset} \
           graph.model_dir=mattport_models/ \
           graph.steps_per_save=25000 \
           graph.max_num_iterations=2000000 \
           logging.enable_stats=False \
           logging.enable_profiler=False &
    echo "Launched ${dataset} on gpu ${!GPU_IDX}"
    GPU_IDX=$((GPU_IDX+1))
done