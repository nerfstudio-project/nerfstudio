#!/bin/bash

IMAGE_NAME="nerfstudio-cuda11"
CONTAINER_NAME="nerfstudio-container"

LOCAL_DIR="/home/horte/Documents/horte/GitHub/3DSplatting"
CONTAINER_DIR="/root/3DSplatting"

echo "🛠️  Construyendo la imagen '$IMAGE_NAME'..."
docker build -t $IMAGE_NAME .

echo "🚀 Iniciando contenedor '$CONTAINER_NAME'..."
docker run -it \
    --cpus=14 \
    --runtime=nvidia \
    --gpus all \
    --name $CONTAINER_NAME \
    -v "$LOCAL_DIR:$CONTAINER_DIR" \
    $IMAGE_NAME

    # --memory=11g \
    # --memory-swap=17g \
    # --shm-size=6g \
