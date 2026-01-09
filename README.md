# LCCNet

Official PyTorch implementation of the paper "LCCNet: Lidar and Camera Self-Calibration Using Cost Volume Network". A video of the demonstration of the method can be found on
 https://www.youtube.com/watch?v=UAAGjYT708A

## Table of Contents

- [Quick Start](#Quick-Start)
- [Requirements](#Requirements)
- [Docker Setup](#Docker-Setup)
- [Dataset Preparation](#Dataset-Preparation)
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Pre-trained model](#Pre-trained_model)
- [Citation](#Citation)



## Quick Start

### Docker를 사용한 빠른 시작 (권장)

```bash
# 1. Docker 이미지 빌드
cd docker
bash build_docker.sh

# 2. Docker 컨테이너 실행
bash run_docker.sh

# 3. 컨테이너 내부에서 correlation_package 빌드
bash build.sh

# 4. 학습 실행
bash run_train.sh
```

## Requirements

* Python 3.6+ (recommend to use [Anaconda](https://www.anaconda.com/))
* PyTorch 1.10.0 (CUDA 11.3)
* CUDA 11.3+
* Docker (선택사항, Docker 사용 시)

## Docker Setup

### Docker 이미지 빌드

```bash
cd docker
bash build_docker.sh
```

이미지 이름: `lccnet:rtx3080`

### Docker 컨테이너 실행

```bash
cd docker
bash run_docker.sh
```

**마운트 설정:**
- 프로젝트 디렉토리: `/workspace/LCCNet`
- 데이터 디렉토리: `/media/gml78905/T7/project_data/LG_Innotek` → `/workspace/data`

## Dataset Preparation

### Hercules Camera-Radar Dataset

데이터셋 구조:
```
hercules/
├── scene1/
│   ├── calibration.yaml
│   ├── camera/
│   │   └── *.png
│   └── radar/
│       └── *.pcd
├── scene2/
│   └── ...
```

`calibration.yaml` 파일에 다음 정보가 포함되어야 합니다:
- Camera intrinsic (K matrix)
- Camera-Radar extrinsic (T_cam_radar)

## Training

### Docker를 사용한 실행 (권장)

1. **Docker 이미지 빌드**
```bash
cd docker
bash build_docker.sh
```

2. **Docker 컨테이너 실행**
```bash
bash run_docker.sh
```

3. **컨테이너 내부에서 correlation_package 빌드**
```bash
bash build.sh
```

4. **학습 실행**
```bash
bash run_train.sh
```

### 파라미터 설정 방법

`run_train.sh` 파일을 수정하여 파라미터를 설정할 수 있습니다:

```bash
# run_train.sh 파일 수정
python3 train_with_sacred.py with \
    sensor_mode='lidar' \          # 'lidar', 'radar', or 'both'
    train_scene='["SC_1", "SC_3"]' \  # 학습에 사용할 scene 리스트 (None = val_scene 제외 모두)
    val_scene='["library_1"]' \    # 검증에 사용할 scene 리스트
    checkpoint_name='my_experiment' \  # 체크포인트 저장 디렉토리 이름 (None = 자동 생성)
    epochs=120 \
    max_t=0.5 \                    # Translation 오차 최대 범위 (미터 단위)
    max_r=5.0 \                    # Rotation 오차 최대 범위 (도 단위)
    batch_size=120 \
    BASE_LEARNING_RATE=1e-4
```

또는 Command Line에서 직접 실행:

```bash
python3 train_with_sacred.py with \
    sensor_mode='both' \
    train_scene='["parking_lot_2", "parking_lot_4"]' \
    val_scene='["library_1"]' \
    epochs=120
```

#### 주요 파라미터 설명

**데이터셋 관련:**
- `sensor_mode`: 센서 모드 선택 ('lidar', 'radar', 'both')
- `train_scene`: 학습에 사용할 scene 리스트 (None = val_scene 제외한 모든 scene)
- `val_scene`: 검증에 사용할 scene 리스트
- `checkpoint_name`: 체크포인트 저장 디렉토리 이름 (None = 자동 생성: `{val_scene}_{sensor_mode}`)

**학습 관련:**
- `epochs`: 학습 epoch 수
- `BASE_LEARNING_RATE`: 학습률 (권장: 1e-4 ~ 3e-4)
- `batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `num_worker`: 데이터 로더 워커 수 (0 = 순차 처리, 4+ = 병렬 처리)

**오차 범위 설정:**
- `max_t`: Translation 오차 최대 범위 (미터 단위)
- `max_r`: Rotation 오차 최대 범위 (도 단위)

자세한 예제는 `run_train.sh` 파일의 주석을 참고하세요.


### 체크포인트 저장 위치

학습된 모델은 다음 경로에 저장됩니다:
```
./checkpoints/{dataset}/{checkpoint_name}/models/
```

예: `./checkpoints/hercules/library_1_lidar/models/checkpoint_r5.00_t0.50_e50_0.123.tar`



## Citation
 
Thank you for citing our paper if you use any of this code or datasets.
```
@article{lv2020lidar,
  title={Lidar and Camera Self-Calibration using CostVolume Network},
  author={Lv, Xudong and Wang, Boya and Ye, Dong and Wang, Shuo},
  journal={arXiv preprint arXiv:2012.13901},
  year={2020}
}
```

### Acknowledgments
 We are grateful to Daniele Cattaneo for his CMRNet [github repository](https://github.com/cattaneod/CMRNet). We use it as our initial code base.
 
<!-- [correlation_package](models/LCCNet/correlation_package) was taken from [flownet2](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)

[LCCNet.py](model/LCCNet.py) is a modified version of the original [PWC-DC network](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py) and modified version [CMRNet](https://github.com/cattaneod/CMRNet/blob/master/models/CMRNet/CMRNet.py)  -->

---
