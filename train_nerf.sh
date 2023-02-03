#!/usr/bin/env bash
set -xe

# vis="viewer"
vis="tensorboard"

CUDA_VISIBLE_DEVICES=0 ns-train depth-nerfacto \
    --data data/nerfstudio/replica_mini/ \
    --pipeline.model.depth-loss-mult 0.01 \
    --vis $vis \
    nerfstudio-data \
    --auto-scale-poses False \
    --center-poses False \
    --train-split-percentage 0.5
