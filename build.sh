#!/bin/bash

# correlation_package 빌드 및 설치 스크립트

# 프로젝트 루트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# correlation_package 디렉토리로 이동
CORRELATION_DIR="models/correlation_package"

if [ ! -d "$CORRELATION_DIR" ]; then
    echo "오류: $CORRELATION_DIR 디렉토리를 찾을 수 없습니다."
    exit 1
fi

cd "$CORRELATION_DIR"

echo "correlation_package 빌드 및 설치를 시작합니다..."
echo "작업 디렉토리: $(pwd)"

# 1. 기존 빌드 폴더 삭제
echo "기존 빌드 폴더를 삭제합니다..."
rm -rf build dist *.egg-info
echo "빌드 폴더 삭제 완료"

# 2. 설치 실행
echo "correlation_package를 설치합니다..."
python setup.py install

if [ $? -eq 0 ]; then
    echo "correlation_package 설치가 완료되었습니다."
else
    echo "오류: correlation_package 설치에 실패했습니다."
    exit 1
fi

git config --global --add safe.directory /workspace/LCCNet

