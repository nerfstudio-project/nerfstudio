helpFunction_launch_eval()
{
   echo "Usage: $0 -c config_name -g gpu_list"
   echo -e "\t-c name of method to benchmark (e.g. graph_mipnerf, graph_instant_ngp)"
   echo -o "\t-o base directory for where all the benchmarks are stored (e.g. outputs/)"
   echo -e "\t-m month of the benchmark of format xx (e.g. 08)"
   echo -e "\t-d date of the benchmark of format xx (e.g. 01)"
   echo -e "\t-y year of the benchmark of format xxxx (e.g. 2022)"
   echo -e "\t-s second timestamp of the benchmark of format xxxxxx (e.g. 003603)"
   echo -e "\t-g list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

while getopts "c:o:m:d:y:s:g" opt; do
    case "$opt" in
        c ) config_name="$OPTARG" ;;
        o ) output_dir="$OPTARG" ;;
        m ) month="$OPTARG" ;;
        d ) date="$OPTARG" ;;
        y ) year="$OPTARG" ;;
        s ) seconds="$OPTARG" ;;
        g ) gpu_list="$OPTARG" ;;
        ? ) helpFunction_launch_eval ;; 
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
idx=0
len=${#GPU_IDX[@]}
((len=len-1))

for dataset in ${DATASETS[@]}; do
    export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}
    base_config_name=${config_name/"graph_"/""}
    config_path="${output_dir}/blender_${dataset}_${month}-${date}-${year}/${base_config_name}/${year}-${month}-${date}_${seconds}/nerfactory_models/"
    python scripts/run_eval.py compute-psnr \
        --load-config=${config_path} \
        --output-path=${output_dir}/${base_config_name}/blender_${dataset}_${month}-${date}-${year}_${seconds}.json &
    echo "Launched ${checkpoint_dir} on gpu ${GPU_IDX[$idx]}"

    # update gpu
    if [ $idx == $len ]; then
        idx=0
    else
        ((idx=idx+1))
    fi

done
