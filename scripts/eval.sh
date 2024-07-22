#!/bin/bash

set -eux

# sh scripts/eval.sh arg_1 arg_2
# arg_1: config path. e.g. outputs/farglory95_colmap/splatfacto-w/2024-07-12_013146/config.yml
# arg_2: render set (train / val / test / train+test) e.g. train+test
# output path = arg 1 dir + output.json
# render path = arg 1 dir + render

# Extract the directory from the first argument
DIR=$(dirname "$1")

# Set output and render paths
OUTPUT_PATH="$DIR/output.json"
RENDER_PATH="$DIR/render"
RENDER_EVAL_PATH="$DIR/render/eval"

# metric results
# e.g.
#   "psnr": 19.985273361206055,
#   "psnr_std": 4.8031487464904785,
#   "ssim": 0.8175548911094666,
#   "ssim_std": 0.10874251276254654,
#   "lpips": 0.2930050790309906,
#   "lpips_std": 0.13271884620189667,
#   "num_rays_per_sec": 11864583.0,
#   "num_rays_per_sec_std": 1446145.5,
#   "fps": 13.405141830444336,
#   "fps_std": 1.633920431137085

# Run the evaluation with the specified config file and output path
ns-eval --load-config "$1" --output-path "$OUTPUT_PATH" --render-output-path "$RENDER_EVAL_PATH"

# Render all images in the specified dataset split
ns-render dataset --load-config "$1" --output-path "$RENDER_PATH" --split "$2"