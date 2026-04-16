#!/bin/bash

# LCCNet Docker 실행 스크립트
# 사용법:
#   ./run_docker.sh [GPU번호] [명령어...]
#
# 예시:
#   ./run_docker.sh 0
#   ./run_docker.sh 0 bash
#   ./run_docker.sh 1 python train.py
#   ./run_docker.sh python train.py        # GPU 기본값 0
#   ./run_docker.sh 0,1 python train.py    # 여러 GPU

set -euo pipefail

# --------------------------------------------------
# 기본 설정
LOCAL_IMAGE="lccnet:dev"
REMOTE_IMAGE=""

WORK_DIR="/workspace/LCCNet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${LCCNET_HOST_DATA_ROOT:-/media/TrainDataset}"
DATA_MOUNT="/workspace/data"

SHM_SIZE="${SHM_SIZE:-32g}"
DEBUGPY_PORT="${DEBUGPY_PORT:-}"

# WANDB_API_KEY는 하드코딩하지 말고 호스트 환경변수에서 받기
WANDB_API_KEY="${WANDB_API_KEY:-}"

# --------------------------------------------------
# 이미지 확인
if docker image inspect "${LOCAL_IMAGE}" &> /dev/null; then
    IMAGE_NAME="${LOCAL_IMAGE}"
    echo "Using local Docker image: ${IMAGE_NAME}"
else
    if [ -n "${REMOTE_IMAGE}" ]; then
        IMAGE_NAME="${REMOTE_IMAGE}"
        echo "Local image not found. Will use remote image: ${IMAGE_NAME}"
    else
        echo "Error: Docker image '${LOCAL_IMAGE}' not found." >&2
        exit 1
    fi
fi

# --------------------------------------------------
# 데이터 경로 확인
if [ ! -d "${DATA_DIR}" ]; then
    echo "Warning: host data path does not exist yet: ${DATA_DIR}" >&2
    echo "Docker will create an empty directory there on first run." >&2
fi

# --------------------------------------------------
# GPU / 명령어 파싱
GPU_NUM="0"
COMMAND_ARGS=()

if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    GPU_NUM="$1"
    shift
    COMMAND_ARGS=("$@")
else
    COMMAND_ARGS=("$@")
fi

CONTAINER_NAME="lccnet_${GPU_NUM//,/_}"

# --------------------------------------------------
# GPU 옵션
if command -v nvidia-smi &> /dev/null; then
    # NOTE:
    # Docker versions differ in how they parse `--gpus device=1,2`.
    # Some parse it as both DeviceIDs and Count, causing:
    # "cannot set both Count and DeviceIDs on device request."
    # Use `--gpus all` and limit visibility via NVIDIA/CUDA env vars.
    GPU_FLAG="--gpus all"
    echo "GPU detected. Using GPU(s): ${GPU_NUM}"
else
    GPU_FLAG=""
    echo "No GPU detected. Running in CPU mode"
fi

# --------------------------------------------------
# 기존 컨테이너 정리
if [ "$(docker ps -aq -f name="${CONTAINER_NAME}")" ]; then
    echo "Stopping existing container named: ${CONTAINER_NAME}..."
    docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
fi

# --------------------------------------------------
# remote image pull
if [ "${IMAGE_NAME}" = "${REMOTE_IMAGE}" ] && [ -n "${REMOTE_IMAGE}" ]; then
    echo "Pulling Docker image: ${IMAGE_NAME}"
    docker pull "${IMAGE_NAME}"
else
    echo "Using local image. Skipping pull."
fi

# --------------------------------------------------
# 출력
echo "Starting Docker container: ${CONTAINER_NAME}"
echo "Project: ${PROJECT_ROOT} -> ${WORK_DIR}"
echo "Data:    ${DATA_DIR} -> ${DATA_MOUNT}"
echo "GPU(s):  ${GPU_NUM}"
echo "shm:     ${SHM_SIZE}"
if [ -n "${DEBUGPY_PORT}" ]; then
    echo "debugpy: ${DEBUGPY_PORT}"
fi

# --------------------------------------------------
# 포트 설정
PORT_ARGS=()
if [ -n "${DEBUGPY_PORT}" ]; then
    PORT_ARGS=(-p "${DEBUGPY_PORT}:${DEBUGPY_PORT}")
fi

# --------------------------------------------------
# TTY 설정
DOCKER_TTY_ARGS=(-i)
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_TTY_ARGS=(-it)
fi

# --------------------------------------------------
# X11 설정
X11_ARGS=()

if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
    X11_ARGS+=(
        -e DISPLAY="${DISPLAY}"
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
fi

if [ -n "${XAUTHORITY:-}" ] && [ -f "${XAUTHORITY}" ]; then
    X11_ARGS+=(
        -e XAUTHORITY="${XAUTHORITY}"
        -v "${XAUTHORITY}:${XAUTHORITY}:rw"
    )
fi

# --------------------------------------------------
# 환경변수 설정
ENV_ARGS=(
    -e PYTHONUNBUFFERED=1
)

if [ -n "${GPU_FLAG}" ]; then
    ENV_ARGS+=(
        -e NVIDIA_VISIBLE_DEVICES="${GPU_NUM}"
        -e CUDA_VISIBLE_DEVICES="${GPU_NUM}"
    )
fi

if [ -n "${WANDB_API_KEY}" ]; then
    ENV_ARGS+=(-e WANDB_API_KEY="${WANDB_API_KEY}")
fi

# --------------------------------------------------
# docker run 명령 구성
DOCKER_RUN=(docker run "${DOCKER_TTY_ARGS[@]}" --rm
    ${GPU_FLAG}
    --name "${CONTAINER_NAME}"
    --ipc=host
    --shm-size="${SHM_SIZE}"
    "${PORT_ARGS[@]}"
    "${X11_ARGS[@]}"
    "${ENV_ARGS[@]}"
    -v "${PROJECT_ROOT}:${WORK_DIR}"
    -v "${DATA_DIR}:${DATA_MOUNT}"
    -w "${WORK_DIR}"
    "${IMAGE_NAME}"
)

# --------------------------------------------------
# 실행
if [ ${#COMMAND_ARGS[@]} -eq 0 ]; then
    echo "No command provided. Starting interactive bash shell..."
    "${DOCKER_RUN[@]}" /bin/bash
else
    "${DOCKER_RUN[@]}" "${COMMAND_ARGS[@]}"
fi

echo "Container stopped."
