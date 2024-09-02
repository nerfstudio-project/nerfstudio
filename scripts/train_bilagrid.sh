#!/bin/bash

set -eux

ns-train splatfacto --data "$1" \
--pipeline.model.rasterize-mode antialiased \
--pipeline.model.use-scale-regularization False  \
--pipeline.model.camera-optimizer.mode off \
--pipeline.model.use_mesh_initialization False \
--pipeline.model.use_bilateral_grid True \
colmap 

