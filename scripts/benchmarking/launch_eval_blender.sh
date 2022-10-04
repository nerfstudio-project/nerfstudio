helpFunction_launch_eval()
{
   echo "Usage: $0 -m method_name -o output_dir -t timestamp -g [OPTIONAL] gpu_list"
   echo -e "\t-m name of method to benchmark (e.g. nerfacto, instant-ngp)"
   echo -e "\t-o base directory for where all the benchmarks are stored (e.g. outputs/)"
   echo -e "\t-t timstamp, if using launch_train_blender.sh will be of format %Y-%m-%d_%H%M%S"
   echo -e "\t-g [OPTIONAL] list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

while getopts "m:o:t:g" opt; do
    case "$opt" in
        m ) method_name="$OPTARG" ;;
        o ) output_dir="$OPTARG" ;;
        t ) timestamp="$OPTARG" ;;
        g ) gpu_list="$OPTARG" ;;
        ? ) helpFunction_launch_eval ;; 
    esac
done

if [ -z "$method_name" ]; then
    echo "Missing method name"
    helpFunction_launch_eval
fi

if [ -z "$output_dir" ]; then
    echo "Missing output directory location"
    helpFunction_launch_eval
fi

if [ -z "$timestamp" ]; then
    echo "Missing timestamp specification"
    helpFunction_launch_eval
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
idx=0
len=${#GPU_IDX[@]}
((len=len-1))

for dataset in ${DATASETS[@]}; do
    export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}
    config_path="${output_dir}/blender_${dataset}_${timestamp::-7}/${method_name}/${timestamp}/config.yml"
    ns-eval --load-config=${config_path} \
            --output-path=${output_dir}/${method_name}/blender_${timestamp}.json &
    echo "Launched ${config_path} on gpu ${GPU_IDX[$idx]}"

    # update gpu
    if [ $idx == $len ]; then
        idx=0
    else
        ((idx=idx+1))
    fi

done
