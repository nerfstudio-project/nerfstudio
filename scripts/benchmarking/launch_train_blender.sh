#!/bin/bash

helpFunction_launch_train()
{
   echo "Usage: $0 -m method_name -g [OPTIONAL] gpu_list"
   echo -e "\t-m name of config to benchmark (e.g. mipnerf, instant_ngp)"
   echo -e "\t-g [OPTIONAL] list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

while getopts "m:g" opt; do
    case "$opt" in
        m ) method_name="$OPTARG" ;;
        g ) gpu_list="$OPTARG" ;;
        ? ) helpFunction ;; 
    esac
done

if [ -z "$method_name" ]; then
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
tag=$(date +'%Y-%m-%d')
idx=0
len=${#GPU_IDX[@]}
((len=len-1))

for dataset in ${DATASETS[@]}; do
    export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}
    ns-train ${method_name} \
             --data=data/blender/${dataset} \
             --experiment-name=blender_${dataset}_${tag} \
             --trainer.relative-model-dir=nerfstudio_models/ \
             --trainer.steps-per-save=1000 \
             --trainer.max-num-iterations=16500 \
             --logging.local-writer.no-enable  \
             --logging.no-enable-profiler \
             --vis wandb \
             blender-data &
    echo "Launched ${method_name} ${dataset} on gpu ${GPU_IDX[$idx]}, ${tag}"
    
    # update gpu
    if [ $idx == $len ]; then
        idx=0
    else
        ((idx=idx+1))
    fi
done