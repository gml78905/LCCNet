#!/bin/bash

# LCCNet Docker 컨테이너 실행 스크립트

# 컨테이너 및 이미지 이름 설정
CONTAINER_NAME="lccnet_container"
IMAGE_NAME="lccnet:rtx3080"

# ------------------------------------------------------------------
# [GPU 설정 로직]
# 첫 번째 인자($1)가 없으면 "all", 있으면 "device=$1" 형식으로 설정
if [ -z "$1" ]; then
    GPU_OPTION="all"
    echo "▶ GPU 모드: 모든 GPU 사용 (Default)"
else
    GPU_OPTION="device=$1"
    echo "▶ GPU 모드: 지정된 GPU 사용 ($1)"
fi
# ------------------------------------------------------------------

# 현재 디렉토리 경로 (프로젝트 루트)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 데이터 디렉토리 경로
DATA_DIR="/media/TrainDataset/"

# 기존 컨테이너가 실행 중이면 중지 및 제거
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "기존 컨테이너를 중지하고 제거합니다..."
    docker stop $CONTAINER_NAME > /dev/null
    docker rm $CONTAINER_NAME > /dev/null
fi

# Docker 컨테이너 실행
echo "Docker 컨테이너를 실행합니다..."
echo " - 프로젝트 경로: $PROJECT_ROOT"
echo " - 컨테이너 이름: $CONTAINER_NAME"

# 데이터 디렉토리 존재 확인 및 마운트 옵션 설정
if [ -d "$DATA_DIR" ]; then
    echo " - 데이터 마운트: $DATA_DIR -> /workspace/data"
    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus "$GPU_OPTION" \
        -v "$PROJECT_ROOT:/workspace/LCCNet" \
        -v "$DATA_DIR:/workspace/data" \
        -w /workspace/LCCNet \
        $IMAGE_NAME \
        /bin/bash
else
    echo "Warning: 데이터 디렉토리를 찾을 수 없습니다 ($DATA_DIR)"
    echo " - 데이터 마운트 없이 실행합니다."
    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus "$GPU_OPTION" \
        -v "$PROJECT_ROOT:/workspace/LCCNet" \
        -w /workspace/LCCNet \
        $IMAGE_NAME \
        /bin/bash
fi
