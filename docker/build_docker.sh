#!/bin/bash

# 프로젝트 루트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "Docker 이미지를 빌드합니다..."
echo "빌드 컨텍스트: $SCRIPT_DIR"
echo "Dockerfile: docker/Dockerfile"

docker build -f docker/Dockerfile -t lccnet:rtx3080 .