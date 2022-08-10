#!/bin/bash

helpFunction_launch_train()
{
   echo "Usage: $0 -c config_name -g gpu_list"
   echo -e "\t-c name of config to benchmark (e.g. graph_mipnerf, graph_instant_ngp)"
   echo -e "\t-g list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

while getopts "c:g" opt; do
    case "$opt" in
        c ) config_name="$OPTARG" ;;
        g ) gpu_list="$OPTARG" ;;
        ? ) helpFunction ;; 
    esac
done

if [ -z "$config_name" ]; then
    echo "Missing method name"
    helpFunction_launch_train
fi

shift $((OPTIND-1))

# Deal with gpu's. If passed in, use those.
GPU_IDX=("$@")
if [ -z "$GPU_IDX" ]; then
    echo "no gpus set... finding available gpus"
    # Find available devices
    num_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    START=0
    END=${num_device}-1
    GPU_IDX=()

    for (( id=$START; id<$END; id++ )); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo [0-9]+)
        if [[ $free_mem -gt 10000 ]]; then
            GPU_IDX+=( $id )
        fi
    done
fi
echo available gpus... ${GPU_IDX[@]}

DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")
date
now=$(date)
tag=$(date +'%m-%d-%Y')
idx=0
len=${#GPU_IDX[@]}
((len=len-1))

for dataset in ${DATASETS[@]}; do
    export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}
    python scripts/run_train.py \
           --config-name ${config_name} \
           '~logging.writer.LocalWriter' \
           data.dataset_inputs_train.data_directory=data/blender/${dataset} \
           data.dataset_inputs_eval.data_directory=data/blender/${dataset} \
           experiment_name=blender_${dataset}_${tag} \
           trainer.model_dir=nerfactory_models/ \
           trainer.steps_per_save=25000 \
           trainer.max_num_iterations=2000000 \
           viewer.enable=False \
           logging.enable_profiler=False &
    echo "Launched ${config_name} ${dataset} on gpu ${GPU_IDX[$idx]}, ${tag}"
    
    # update gpu
    if [ $idx == $len ]; then
        idx=0
    else
        ((idx=idx+1))
    fi
done