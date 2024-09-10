#!/usr/bin/env bash
set -e

find_directories() {
    local directory_path="$1"
    # Use the find command to list directories only (depth 1 for non-recursive)
    find "$directory_path" -maxdepth 1 -type d | grep -v '^'"$directory_path"'$'
}

# Validate the input arguments
if [ $# -lt 1 ]; then
  echo "Usage: $0 <zip file> [<method1> <method2> ...] [--skip-preprocess]"
  echo "Available methods: arkit colmap, loftr, lightglue, glomap"
  echo "Default method: arkit"
  exit 1
fi

zip_file_path=$1
shift
methods=()
skip_preprocess=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-preprocess)
      skip_preprocess=true
      shift
      ;;
    *)
      methods+=("$1")
      shift
      ;;
  esac
done

# If no methods specified, default to arkit
if [ ${#methods[@]} -eq 0 ]; then
  methods=("arkit")
fi

# unzip file
input_base_path="${zip_file_path%.*}"
echo "unzipping file ${zip_file_path} to ${input_base_path}"
mkdir -p "$input_base_path"
unzip "$zip_file_path" -d "$input_base_path"

directories=$(find_directories ${input_base_path})
colmap_directory_path="${directories[0]}/colmap"
if [ ${#directories[@]} -gt 0 ]; then
    echo "Scan directory: ${directories[0]}"
else
    echo "No directories found"
fi

echo "input_base_path: ${input_base_path}"
echo "colmap_directory_path  ${colmap_directory_path}" 
echo "methods: ${methods[@]}"
echo "skip_preprocess: ${skip_preprocess}"

remove_and_create_folder() {
  if [ -d "$1" ]; then
    rm -rf "$1"
  fi
  mkdir -p "$1"
}

if [ "$skip_preprocess" = false ]; then
  echo "=== Preprocess ARkit data === "
  remove_and_create_folder "${colmap_directory_path}/post"
  remove_and_create_folder "${colmap_directory_path}/post/sparse"
  remove_and_create_folder "${colmap_directory_path}/post/sparse/online"
  remove_and_create_folder "${colmap_directory_path}/post/sparse/online_loop"

  echo "1. Undistort image using AVFoundation calibration data"
  python arkit_utils/undistort_images/undistort_image_cuda.py --input_base ${colmap_directory_path}

  echo "2. Transform ARKit mesh to point3D"
  python arkit_utils/mesh_to_points3D/arkitobj2point3D.py --input_base_path ${colmap_directory_path}

  echo "3. Transform ARKit pose to COLMAP coordinate"
  python arkit_utils/arkit_pose_to_colmap.py --input_database_path ${colmap_directory_path}

  echo "4. Optimize pose using selected methods"
  if [[ ! " ${methods[@]} " =~ " arkit " ]]; then
    remove_and_create_folder "${colmap_directory_path}/post/sparse/offline"
    python arkit_utils/pose_optimization/optimize_pose.py --input_database_path ${colmap_directory_path} --methods ${methods[@]}
  else
    echo "Skipping pose optimization"
  fi

  echo "5. Prepare dataset for nerfstudio"
  python arkit_utils/prepare_nerfstudio_dataset.py --input_path ${colmap_directory_path}

  echo "Dataset preparation completed."
fi

echo "6. Start training nerfstudio"
python arkit_utils/run_nerfstudio_dataset.py --input_path ${colmap_directory_path} --method ${methods[@]}