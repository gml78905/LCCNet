#!/bin/bash

# LCCNet Docker 컨테이너 실행 스크립트

# 컨테이너 이름 설정
CONTAINER_NAME="lccnet_container"
IMAGE_NAME="lccnet:rtx3080"

# 현재 디렉토리 경로 (프로젝트 루트)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 데이터 디렉토리 경로
DATA_DIR="/media/gml78905/T7/project_data/LG_Innotek"

# 기존 컨테이너가 실행 중이면 중지 및 제거
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "기존 컨테이너를 중지하고 제거합니다..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Docker 컨테이너 실행
echo "Docker 컨테이너를 실행합니다..."
echo "프로젝트 디렉토리: $PROJECT_ROOT"
echo "컨테이너 이름: $CONTAINER_NAME"

# 데이터 디렉토리 존재 확인 및 마운트 옵션 설정
if [ -d "$DATA_DIR" ]; then
    echo "데이터 디렉토리: $DATA_DIR -> /workspace/data"
    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus all \
        -v "$PROJECT_ROOT:/workspace/LCCNet" \
        -v "$DATA_DIR:/workspace/data" \
        -w /workspace/LCCNet \
        $IMAGE_NAME \
        /bin/bash
else
    echo "경고: 데이터 디렉토리를 찾을 수 없습니다: $DATA_DIR"
    echo "데이터 마운트 없이 계속 진행합니다..."
    docker run -it --rm \
        --name $CONTAINER_NAME \
        --gpus all \
        -v "$PROJECT_ROOT:/workspace/LCCNet" \
        -w /workspace/LCCNet \
        $IMAGE_NAME \
        /bin/bash
fi

