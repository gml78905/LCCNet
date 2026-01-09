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
python3 train_with_sacred.py
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
python3 train_with_sacred.py
```

### 파라미터 설정 방법

`train_with_sacred.py` 파일의 `config()` 함수에서 파라미터를 수정할 수 있습니다:

```python
@ex.config
def config():
    # 데이터셋 설정
    dataset = 'hercules'  # 'kitti/odom', 'kitti/raw', 'hercules'
    data_folder = '/workspace/data/PublicDataset/hercules'
    
    # 검증 씬 설정 (Hercules)
    val_scene = ['parking_lot_1', 'parking_lot_2', 'parking_lot_4']
    val_scene_name = "cheakpoint_name"  # 체크포인트 저장 디렉토리 이름
    
    # 학습 설정
    epochs = 120
    BASE_LEARNING_RATE = 1e-4
    batch_size = 32
    num_worker = 0  # 0 = 순차 처리, 4+ = 병렬 처리 (속도 향상)
    
    # 오차 범위 설정
    max_t = 0.1  # Translation 오차 최대 범위 (미터 단위)
    max_r = 1.0  # Rotation 오차 최대 범위 (도 단위)
    
    # 모델 설정
    network = 'Res_f1'
    optimizer = 'adam'  # 'adam' or 'sgd'
    loss = 'combined'
    
    # 기타 설정
    use_reflectance = False
    dropout = 0.0
    max_depth = 80.0
    weight_point_cloud = 0.5
```

### 주요 파라미터 설명

#### 데이터셋 관련
- `dataset`: 사용할 데이터셋 타입
  - `'hercules'`: Camera-Radar 데이터셋
  - `'kitti/odom'`: KITTI Odometry 데이터셋
  - `'kitti/raw'`: KITTI Raw 데이터셋
- `data_folder`: 데이터셋 경로
- `val_scene`: 검증에 사용할 씬 리스트 (Hercules)
- `val_scene_name`: 체크포인트 저장 디렉토리 이름

#### 학습 관련
- `epochs`: 학습 epoch 수
- `BASE_LEARNING_RATE`: 학습률 (권장: 1e-4 ~ 3e-4)
- `batch_size`: 배치 크기 (GPU 메모리에 따라 조정)
- `num_worker`: 데이터 로더 워커 수
  - `0`: 순차 처리 (디버깅 시 유용, 느림)
  - `4+`: 병렬 처리 (속도 향상, 빠름)

#### 오차 범위 설정
- `max_t`: Translation 오차 최대 범위 (미터 단위)
  - 예: `0.1` = ±10cm 범위의 이동 오차
  - 작을수록 정밀한 보정 학습
- `max_r`: Rotation 오차 최대 범위 (도 단위)
  - 예: `1.0` = ±1도 범위의 회전 오차
  - 작을수록 정밀한 보정 학습

#### 모델 설정
- `network`: 네트워크 구조 (예: 'Res_f1')
- `optimizer`: 옵티마이저 ('adam' 또는 'sgd')
- `loss`: 손실 함수 ('combined', 'geometric', 'points_distance' 등)
- `dropout`: Dropout 비율
- `max_depth`: 최대 깊이 (미터)
- `weight_point_cloud`: 포인트 클라우드 loss 가중치

### 체크포인트 저장 위치

학습된 모델은 다음 경로에 저장됩니다:
```
./checkpoints/{dataset}/{val_scene_name}/models/
```

예: `./checkpoints/hercules/radar/models/checkpoint_r1.00_t0.10_e50_0.123.tar`


## Pre-trained model

Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1VbQV3ERDeT3QbdJviNCN71yoWIItZQnl?usp=sharing)

## Evaluation

1. Download [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. Change the path to the dataset in `evaluate_calib.py`.
```python
data_folder = '/path/to/the/KITTI/odometry_color/'
```
3. Create a folder named `pretrained` to store the pre-trained models in the root path.
4. Download pre-trained models and modify the weights path in `evaluate_calib.py`.
```python
weights = [
   './pretrained/kitti_iter1.tar',
   './pretrained/kitti_iter2.tar',
   './pretrained/kitti_iter3.tar',
   './pretrained/kitti_iter4.tar',
   './pretrained/kitti_iter5.tar',
]
```
5. Run evaluation.
```commandline
python evaluate_calib.py
```


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
