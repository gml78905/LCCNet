# 1. RTX 3080 지원 베이스 이미지 (PyTorch 1.10, CUDA 11.3)
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

LABEL maintainer="LCCNet_User"

# 2. 환경 변수 설정 (Ampere 아키텍처 sm_86 지원)
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUDA_HOME="/usr/local/cuda"
ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# 3. 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉토리 설정
WORKDIR /workspace/LCCNet

# 5. Python 라이브러리 설치
# (수정한 requirements_fixed.txt 파일을 컨테이너로 복사 후 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. 기본 쉘 설정
CMD ["/bin/bash"]