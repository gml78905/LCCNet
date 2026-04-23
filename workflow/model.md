# TriJointV3Lite Model Description

이 문서는 현재 코드베이스의 `TriJointV3Lite` 모델을 실제 구현 기준으로 설명한다.
현재 클래스 이름은 그대로 `TriModalJointCalibNetV3Lite`이지만,
내부 구조는 초기 v3-lite보다 한 단계 더 발전한 `v3.1-lite` 성격을 가진다.

핵심 차이는 다음과 같다.

- joint-centered main path는 유지한다.
- pairwise-centered internal path로 돌아가지 않는다.
- reliability는 여전히 residual 뒤에 위치한다.
- dense alignment 정보를 더 오래 유지한다.
- residual understanding 블록에 local soft alignment cue가 추가되었다.
- memory / refinement도 pooled latent 하나만 쓰지 않고 dense-aware summary를 함께 사용한다.

설명 대상 파일:

- `models/tri_joint_v3/encoders.py`
- `models/tri_joint_v3/coarse_context.py`
- `models/tri_joint_v3/joint_feature.py`
- `models/tri_joint_v3/residual_understanding.py`
- `models/tri_joint_v3/reliability.py`
- `models/tri_joint_v3/joint_fusion.py`
- `models/tri_joint_v3/memory.py`
- `models/tri_joint_v3/refinement.py`
- `models/tri_joint_v3/heads.py`
- `models/tri_joint_v3/model.py`


## 1. 목적

이 모델의 목적은 camera / LiDAR / radar 세 센서를 동시에 사용하여,
현재 입력된 perturbed extrinsic을 얼마나 되돌려야 하는지에 대한
`pairwise delta pose`를 예측하는 것이다.

모델이 직접 출력하는 것은 corrected pose가 아니라 다음 3개 delta pose이다.

- `ΔT_CL`
- `ΔT_CR`
- `ΔT_LR`

즉 모델의 최종 출력 key는 아래와 같다.

- `T_CL_t`, `T_CL_q`
- `T_CR_t`, `T_CR_q`
- `T_LR_t`, `T_LR_q`

여기서:

- `*_t`: translation delta
- `*_q`: quaternion delta

를 뜻한다.

중요:

- corrected pose 계산은 모델 내부에서 하지 않는다.
- corrected pose는 기존 코드처럼 `loss / util`에서 계산한다.
- current delta-pose formulation은 그대로 유지된다.


## 2. 핵심 철학

현재 `TriJointV3Lite`의 핵심 철학은 다음과 같다.

1. 내부 main path는 하나의 joint calibration state를 중심으로 한다.
2. 내부 추론을 explicit pairwise relation state 중심으로 두지 않는다.
3. reliability는 early gating prior가 아니라 residual-aware confidence다.
4. direct `concat -> pose regression` 경로를 사용하지 않는다.
5. dense alignment 정보를 가능한 오래 유지한다.
6. pairwise 출력은 마지막 readout에서만 분기한다.

현재 main path는 아래 순서다.

1. encoders
2. shared coarse context
3. joint alignment feature construction
4. dense alignment residual field
5. residual-aware reliability
6. alignment-preserving joint fusion
7. map-aware joint temporal memory
8. dense-aware joint refinement
9. pairwise delta readout


## 3. 입력 / 출력

## 3.1 입력

현재 모델은 기존 input contract를 그대로 유지한다.

- `rgb`: `[B, 3, H, W]`
- `lidar_proj`: `[B, 1, H, W]`
- `radar_proj`: `[B, 2, H, W]`

현재 학습 코드에서는 보통 아래 해상도를 쓴다.

- `H = 288`
- `W = 512`

즉 대개 실제 shape는 다음과 같다.

- `rgb`: `[B, 3, 288, 512]`
- `lidar_proj`: `[B, 1, 288, 512]`
- `radar_proj`: `[B, 2, 288, 512]`

sequence 입력도 지원한다.

- `rgb`: `[B, T, 3, H, W]`
- `lidar_proj`: `[B, T, 1, H, W]`
- `radar_proj`: `[B, T, 2, H, W]`


## 3.2 출력

기본 반환:

```python
pred, new_state = model(
    rgb,
    lidar_proj,
    radar_proj,
    state=None,
    return_aux=False,
)
```

디버그 반환:

```python
pred, new_state, aux = model(
    rgb,
    lidar_proj,
    radar_proj,
    state=None,
    return_aux=True,
)
```

`pred`는 아래 key를 가진다.

- `T_CL_t`: `[B, 3]` 또는 sequence면 `[B, T, 3]`
- `T_CL_q`: `[B, 4]` 또는 sequence면 `[B, T, 4]`
- `T_CR_t`: `[B, 3]` 또는 sequence면 `[B, T, 3]`
- `T_CR_q`: `[B, 4]` 또는 sequence면 `[B, T, 4]`
- `T_LR_t`: `[B, 3]` 또는 sequence면 `[B, T, 3]`
- `T_LR_q`: `[B, 4]` 또는 sequence면 `[B, T, 4]`

`new_state`는 현재 아래 형태다.

- `{"h_joint": [B, 256]}`

`aux`는 중간 디버깅 텐서를 담는다.


## 4. 전체 블록 다이어그램

```text
rgb, lidar_proj, radar_proj
    ->
multi-scale modality encoders
    ->
cam_s16, cam_s32
lid_s16, lid_s32
rad_s16, rad_s32
    ->
shared coarse context
    ->
ctx_s16, z_shared
    ->
joint alignment feature construction
    ->
F_joint_map
z_joint
z_joint_support
z_joint_conflict
z_joint_align
    ->
dense alignment residual field
    ->
E_joint_map
e_joint
e_align_summary
    ->
residual-aware reliability
    ->
R_cam, R_lid, R_rad
r_summary
invalid_penalty
    ->
alignment-preserving joint fusion
    ->
F_joint_fused
z_joint_fused
z_joint_support_weighted
z_joint_residual_weighted
fusion_map_summary
    ->
map-aware joint temporal memory
    ->
h_joint
    ->
dense-aware joint refinement
    ->
z_joint_ref
    ->
pairwise delta pose heads
    ->
T_CL, T_CR, T_LR delta pose
```


## 5. 블록별 상세 설명

## 5.1 Encoders

파일:

- `models/tri_joint_v3/encoders.py`

현재 v3 구조에서는 encoder를 새로 설계하지 않고,
기존 tri-joint encoder를 그대로 재사용한다.

이 파일은 아래 클래스를 re-export 한다.

- `CameraEncoderMS`
- `LidarEncoderMS`
- `RadarEncoderMS`

즉 encoder 출력 contract를 유지한 채,
그 이후의 main path를 새 구조로 바꾼 것이다.

출력 tensor:

- `cam_s16`: `[B, 256, H/16, W/16]`
- `cam_s32`: `[B, 512, H/32, W/32]`
- `lid_s16`: `[B, 256, H/16, W/16]`
- `lid_s32`: `[B, 512, H/32, W/32]`
- `rad_s16`: `[B, 256, H/16, W/16]`
- `rad_s32`: `[B, 512, H/32, W/32]`

예를 들어 입력이 `[B, C, 288, 512]`이면 대략:

- `s16`: `[B, 256, 18, 32]`
- `s32`: `[B, 512, 9, 16]`

이 된다.


## 5.2 Shared Coarse Context

파일:

- `models/tri_joint_v3/coarse_context.py`

클래스:

- `SharedCoarseContextV3`

역할:

- 세 센서의 `s32` feature를 하나의 coarse shared context로 합친다.
- 이 context는 joint alignment feature construction의 conditioning signal로 사용된다.

입력:

- `cam_s32`: `[B, 512, H/32, W/32]`
- `lid_s32`: `[B, 512, H/32, W/32]`
- `rad_s32`: `[B, 512, H/32, W/32]`

출력:

- `ctx_s16`: `[B, 256, H/16, W/16]`
- `z_shared`: `[B, 256]`

어떻게 만들어지는가:

1. 세 feature를 channel 방향으로 concat한다.

```text
x = cat([cam_s32, lid_s32, rad_s32], dim=1)
shape = [B, 1536, H/32, W/32]
```

2. `1x1 conv + 3x3 conv` 기반 `fuse` block을 통과시켜 `ctx_s32`를 만든다.

```text
ctx_s32 = fuse(x)
shape = [B, 256, H/32, W/32]
```

3. global average pooling으로 `z_shared`를 만든다.

```text
z_shared = AdaptiveAvgPool2d(1)(ctx_s32).flatten(1)
shape = [B, 256]
```

4. `ctx_s32`를 bilinear upsample 하여 `ctx_s16`를 만든다.

```text
ctx_s16 = interpolate(ctx_s32, size=(H/16, W/16))
shape = [B, 256, H/16, W/16]
```

즉:

- `ctx_s16`은 dense shared context map
- `z_shared`는 global shared context token

이다.


## 5.3 Joint Alignment Feature Construction

파일:

- `models/tri_joint_v3/joint_feature.py`

클래스:

- `JointAlignmentFeatureBuilder`

참고:

- backward compatibility를 위해
  `JointCalibrationFeatureBuilder = JointAlignmentFeatureBuilder`
  alias도 남겨져 있다.

역할:

- 세 센서의 `s16` feature와 shared context를 받아
  단순 fused feature가 아니라 calibration alignment 관점의 dense joint state를 만든다.
- 이 블록은 generic fusion이 아니라 아래 4종류 정보를 분리해서 구성한다.

1. support
2. common structure
3. conflict / disagreement
4. directional alignment precursor

입력:

- `cam_s16`: `[B, 256, H/16, W/16]`
- `lid_s16`: `[B, 256, H/16, W/16]`
- `rad_s16`: `[B, 256, H/16, W/16]`
- `ctx_s16`: `[B, 256, H/16, W/16]`

출력:

- `F_joint_map`: `[B, 384, H/16, W/16]`
- `z_joint`: `[B, 384]`
- `z_joint_support`: `[B, 384]`
- `z_joint_conflict`: `[B, 384]`
- `z_joint_align`: `[B, 384]`

중간 텐서:

- `support_branch`: `[B, 96, H/16, W/16]`
- `common_structure`: `[B, 96, H/16, W/16]`
- `conflict_branch`: `[B, 96, H/16, W/16]`
- `align_precursor`: `[B, 96, H/16, W/16]`
- `diff_cl`: `[B, 128, H/16, W/16]`
- `diff_cr`: `[B, 128, H/16, W/16]`
- `diff_lr`: `[B, 128, H/16, W/16]`

어떻게 만들어지는가:

### 5.3.1 modality projection

각 feature를 common space로 projection 한다.

```text
cam_j = proj_cam(cam_s16) -> [B,128,H/16,W/16]
lid_j = proj_lid(lid_s16) -> [B,128,H/16,W/16]
rad_j = proj_rad(rad_s16) -> [B,128,H/16,W/16]
ctx_j = proj_ctx(ctx_s16) -> [B,128,H/16,W/16]
```

또한 modality 평균 feature를 만든다.

```text
mean_mod = (cam_j + lid_j + rad_j) / 3
shape = [B,128,H/16,W/16]
```

### 5.3.2 conflict seeds

pairwise absolute difference를 계산한다.

```text
diff_cl = |cam_j - lid_j|
diff_cr = |cam_j - rad_j|
diff_lr = |lid_j - rad_j|
shape = [B,128,H/16,W/16]
```

그리고 평균 conflict seed를 만든다.

```text
conflict_seed = (diff_cl + diff_cr + diff_lr) / 3
shape = [B,128,H/16,W/16]
```

### 5.3.3 support branch

```text
support = support_branch(cat([cam_j, lid_j, rad_j, ctx_j], dim=1))
shape = [B,96,H/16,W/16]
```

이 branch는
"현재 위치에서 tri-modal joint evidence를 얼마나 참고할 수 있는가"
를 나타내는 support-like feature다.

### 5.3.4 common structure branch

```text
common_structure = common_branch(
    cat([mean_mod, min(cam_j,lid_j), min(cam_j,rad_j), ctx_j], dim=1)
)
shape = [B,96,H/16,W/16]
```

이 branch는 여러 modality가 공유하는 구조적 common pattern을 강조하려는 branch다.

### 5.3.5 conflict branch

```text
conflict = conflict_branch(cat([diff_cl, diff_cr, diff_lr, conflict_seed], dim=1))
shape = [B,96,H/16,W/16]
```

이 branch는 어디서 modality disagreement가 강한지 표현한다.

### 5.3.6 alignment precursor branch

```text
align_precursor = align_precursor_branch(
    cat([cam_j, lid_j, rad_j, mean_mod, ctx_j], dim=1)
)
shape = [B,96,H/16,W/16]
```

이 branch는 이후 residual understanding에서 alignment를 해석하기 전에,
방향성 있는 dense alignment clue의 precursor 역할을 한다.

### 5.3.7 joint fusion

위 네 branch를 concat하여 `F_joint_map`을 만든다.

```text
cat([support, common_structure, conflict, align_precursor], dim=1)
shape = [B,384,H/16,W/16]

F_joint_map = fusion(...)
shape = [B,384,H/16,W/16]
```

그리고 global average pooling:

```text
z_joint = AdaptiveAvgPool2d(1)(F_joint_map).flatten(1)
shape = [B,384]
```

### 5.3.8 weighted pooled joint summaries

단일 GAP token만 쓰지 않기 위해,
branch별 score를 이용한 weighted pooling도 추가한다.

예를 들면:

```text
support_score = score_head(support) -> [B,1,H/16,W/16]
conflict_score = score_head(conflict) -> [B,1,H/16,W/16]
align_score = score_head(align_precursor) -> [B,1,H/16,W/16]
```

이를 `softmax` 기반 weighted pooling에 사용한다.

출력:

- `z_joint_support [B,384]`
- `z_joint_conflict [B,384]`
- `z_joint_align [B,384]`

즉 이 블록은 이제 단순 `F_joint_map + GAP`가 아니라,
alignment 관점의 여러 dense state와 map-aware summary를 동시에 만든다.


## 5.4 Support / Validity Priors

이 부분은 `models/tri_joint_v3/model.py` 내부 helper에서 만든다.

현재 모델은 residual / reliability 계산 전에 다음 prior를 만든다.

- `support_cam`
- `valid_lid`
- `valid_rad`

이들은 최종 reliability가 아니라,
residual / reliability 계산에 들어가는 support / validity prior다.


### 5.4.1 support_cam

생성 함수:

- `TriModalJointCalibNetV3Lite._support_cam`

입력:

- `rgb [B,3,H,W]`

생성 방식:

1. RGB 절대값 평균

```text
support = rgb.abs().mean(dim=1, keepdim=True)
shape = [B,1,H,W]
```

2. bilinear downsample

```text
support = interpolate(support, size=(H/16,W/16))
shape = [B,1,H/16,W/16]
```

3. spatial max normalize

```text
support_cam = support / support.amax(dim=(2,3), keepdim=True)
shape = [B,1,H/16,W/16]
```

의미:

- 현재 `support_cam`은 camera structural support의 매우 단순한 proxy다.
- 아직 full structural support estimator는 아니다.


### 5.4.2 valid_lid / valid_rad

생성 함수:

- `TriModalJointCalibNetV3Lite._valid_from_proj`

입력:

- `lidar_proj [B,1,H,W]`
- `radar_proj[:, :1] [B,1,H,W]`

생성 방식:

```text
valid = (proj > 0).float()
valid = interpolate(valid, size=(H/16,W/16), mode="nearest")
```

출력:

- `valid_lid [B,1,H/16,W/16]`
- `valid_rad [B,1,H/16,W/16]`

의미:

- 해당 셀에 실제 LiDAR / Radar support가 존재하는지를 나타낸다.


## 5.5 Dense Alignment Residual Field

파일:

- `models/tri_joint_v3/residual_understanding.py`

클래스:

- `JointResidualUnderstanding`

이 블록은 현재 모델에서 가장 중요한 변화 중 하나다.

역할:

- joint feature가 형성된 뒤,
  "현재 image/grid 상에서 얼마나 안 맞는가"를 더 직접적으로 나타내는 dense residual field를 만든다.

입력:

- `cam_s16`: `[B,256,H/16,W/16]`
- `lid_s16`: `[B,256,H/16,W/16]`
- `rad_s16`: `[B,256,H/16,W/16]`
- `F_joint_map`: `[B,384,H/16,W/16]`
- `support_cam`: `[B,1,H/16,W/16]`
- `valid_lid`: `[B,1,H/16,W/16]`
- `valid_rad`: `[B,1,H/16,W/16]`

출력:

- `E_joint_map`: `[B,192,H/16,W/16]`
- `e_joint`: `[B,192]`
- `e_align_summary`: `[B,64]`

중간 텐서:

- `feat_disagree`: `[B,128,H/16,W/16]`
- `support_mismatch`: `[B,32,H/16,W/16]`
- `local_align`: `[B,64,H/16,W/16]`
- `geom_residual`: `[B,32,H/16,W/16]`
- `align_cl`: `[B,5,H/16,W/16]`
- `align_cr`: `[B,5,H/16,W/16]`
- `align_lr`: `[B,5,H/16,W/16]`


### 5.5.1 feature disagreement

기존처럼 encoder feature 차이를 계산하지만,
이제 이건 residual field의 한 구성요소일 뿐이다.

```text
cat([
  |cam_s16 - lid_s16|,
  |cam_s16 - rad_s16|,
  |lid_s16 - rad_s16|
], dim=1)
shape = [B,768,H/16,W/16]

feat_disagree = disagree_proj(...)
shape = [B,128,H/16,W/16]
```


### 5.5.2 support mismatch

각 modality support / validity 차이를 이용한다.

```text
cat([
  |support_cam - valid_lid|,
  |support_cam - valid_rad|,
  |valid_lid - valid_rad|
], dim=1)
shape = [B,3,H/16,W/16]

support_mismatch = support_proj(...)
shape = [B,32,H/16,W/16]
```


### 5.5.3 local soft alignment cue

이 부분이 v3.1-lite의 핵심이다.

기존처럼 exact cell만 비교하지 않고,
작은 neighborhood 안에서 soft best local match를 본다.

구현 함수:

- `_local_soft_alignment(src, ref)`

입력:

- `src [B,C,H,W]`
- `ref [B,C,H,W]`

과정:

1. `src`, `ref`를 channel-wise normalize 한다.
2. `ref`에서 `(2r+1) x (2r+1)` neighborhood를 `unfold`로 꺼낸다.
3. 현재 cell의 `src` feature와 주변 patch feature 간 similarity를 계산한다.
4. similarity에 `softmax`를 적용하여 neighborhood 내 soft match 분포를 만든다.
5. 이를 이용해 expected displacement를 계산한다.

현재 출력 5채널은 다음 의미를 가진다.

- `best_sim`
- `expected_sim`
- `disp_norm`
- `disp_x`
- `disp_y`

즉:

```text
align_cl = _local_soft_alignment(cam_s16, lid_s16) -> [B,5,H/16,W/16]
align_cr = _local_soft_alignment(cam_s16, rad_s16) -> [B,5,H/16,W/16]
align_lr = _local_soft_alignment(lid_s16, rad_s16) -> [B,5,H/16,W/16]
```

그 후 현재 구현은 아래 6채널 입력을 만든다.

- `align_cl`의 best / expected sim
- `align_cr`의 best / expected sim
- `align_lr`의 displacement norm
- `(align_cl disp_norm + align_cr disp_norm) / 2`

```text
local_align_input shape = [B,6,H/16,W/16]
local_align = align_proj(local_align_input)
shape = [B,64,H/16,W/16]
```

이 `local_align`는 이제
"어디가 주변 위치 기준으로도 잘 안 맞는가"
를 나타내는 dense alignment cue다.


### 5.5.4 lightweight geometric residual

현재는 full geometry residual까지는 아니지만,
LiDAR / Radar support overlap과 asymmetry를 이용한 가벼운 geometric cue를 둔다.

```text
overlap = valid_lid * valid_rad
asym = |valid_lid - valid_rad|
shape = [B,2,H/16,W/16]
```

이를 projection 하면:

```text
geom_residual = geom_proj(...)
shape = [B,32,H/16,W/16]
```


### 5.5.5 residual map formation

최종적으로 아래를 concat한다.

- `F_joint_map [B,384,H/16,W/16]`
- `feat_disagree [B,128,H/16,W/16]`
- `support_mismatch [B,32,H/16,W/16]`
- `local_align [B,64,H/16,W/16]`
- `geom_residual [B,32,H/16,W/16]`

총:

```text
[B,640,H/16,W/16]
```

이를 `fuse`에 통과시켜:

```text
E_joint_map = fuse(...)
shape = [B,192,H/16,W/16]
```

이후 global average pooling:

```text
e_joint = AdaptiveAvgPool2d(1)(E_joint_map).flatten(1)
shape = [B,192]
```


### 5.5.6 alignment summary

refinement과 reliability에서 alignment-aware summary를 쓰기 위해
`e_align_summary`도 추가로 만든다.

현재 구현은 다음 alignment statistics를 모은다.

- `align_cl disp_norm mean`
- `align_cr disp_norm mean`
- `align_lr disp_norm mean`
- `align_cl best_sim mean`
- `align_cr best_sim mean`
- `align_lr best_sim mean`

즉:

```text
align_stats = [B,6]
```

그리고:

```text
cat([e_joint, align_stats], dim=1)
shape = [B,198]

e_align_summary = summary_mlp(...)
shape = [B,64]
```


## 5.6 Residual-Aware Reliability

파일:

- `models/tri_joint_v3/reliability.py`

클래스:

- `ResidualAwareReliabilityEstimator`

역할:

- reliability를 joint feature와 residual field를 본 뒤 계산한다.
- 즉 reliability는 early prior가 아니라 alignment-aware confidence다.

입력:

- `F_joint_map`: `[B,384,H/16,W/16]`
- `E_joint_map`: `[B,192,H/16,W/16]`
- `support_cam`: `[B,1,H/16,W/16]`
- `valid_lid`: `[B,1,H/16,W/16]`
- `valid_rad`: `[B,1,H/16,W/16]`
- `e_joint`: `[B,192]`
- `e_align_summary`: `[B,64]`

출력:

- `R_cam`: `[B,1,H/16,W/16]`
- `R_lid`: `[B,1,H/16,W/16]`
- `R_rad`: `[B,1,H/16,W/16]`
- `r_summary`: `[B,64]`
- `invalid_penalty`: `[B,3]`


### 5.6.1 modality-specific inputs

camera:

```text
cam_in = cat([F_joint_map, E_joint_map, support_cam, 1 - support_cam], dim=1)
shape = [B,578,H/16,W/16]
```

lidar:

```text
lid_in = cat([F_joint_map, E_joint_map, valid_lid, 1 - valid_lid], dim=1)
shape = [B,578,H/16,W/16]
```

radar:

```text
rad_in = cat([F_joint_map, E_joint_map, valid_rad, 1 - valid_rad], dim=1)
shape = [B,578,H/16,W/16]
```

이 구조는 validity / non-validity 정보를 모두 head가 볼 수 있게 하려는 것이다.


### 5.6.2 raw reliability maps

각 input은 `_ResidualAwareReliabilityHead`를 통과한다.

출력:

- `R_cam_raw [B,1,H/16,W/16]`
- `R_lid_raw [B,1,H/16,W/16]`
- `R_rad_raw [B,1,H/16,W/16]`


### 5.6.3 camera / lidar / radar reliability

camera는 dense modality이므로 support-aware floor를 둔다.

```text
camera_floor = 0.15 + 0.85 * support_cam
R_cam = sigmoid(R_cam_raw) * camera_floor
```

즉 camera reliability는 완전 free mask가 아니라 support prior를 가진다.

LiDAR / Radar는 validity-aware로 유지한다.

```text
R_lid = sigmoid(R_lid_raw) * valid_lid
R_rad = sigmoid(R_rad_raw) * valid_rad
```

따라서 point가 없는 위치에서 high reliability가 뜨는 것을 구조적으로 제한한다.


### 5.6.4 invalid_penalty hook

후속 loss에서 활용할 수 있도록 invalid-zone high reliability penalty hook를 모델에서 계산해둔다.

현재 정의:

```text
invalid_penalty = mean(R_map * (1 - valid))
```

세 modality에 대해:

```text
invalid_penalty shape = [B,3]
```

순서는:

- camera penalty
- lidar penalty
- radar penalty

의미:

- 지금은 loss에 직접 쓰지 않지만,
  invalid zone high confidence를 제어할 수 있는 명확한 hook가 생긴 것이다.


### 5.6.5 r_summary

reliability summary는 이제 단순 mean/std만이 아니라
alignment summary도 함께 본다.

현재 `summary_in` 구성:

- `e_joint [B,192]`
- `e_align_summary [B,64]`
- `stats(R_cam) [B,2]`
- `stats(R_lid) [B,2]`
- `stats(R_rad) [B,2]`
- `invalid_penalty [B,3]`
- `support_cam mean [B,1]`
- `valid_lid mean [B,1]`
- `valid_rad mean [B,1]`

총:

```text
[B,268]
```

이를 `summary_mlp`에 넣어:

```text
r_summary [B,64]
```

를 만든다.


## 5.7 Alignment-Preserving Joint Fusion

파일:

- `models/tri_joint_v3/joint_fusion.py`

클래스:

- `JointEvidenceFusion`

역할:

- joint feature, residual field, reliability map을 합쳐
  실제 readout에 가까운 fused joint state를 만든다.
- 단, 여기서 곧바로 하나의 GAP token만 남기지 않고,
  dense map 정보를 더 오래 유지하기 위한 map-aware summary를 함께 만든다.

입력:

- `F_joint_map`: `[B,384,H/16,W/16]`
- `E_joint_map`: `[B,192,H/16,W/16]`
- `R_cam`: `[B,1,H/16,W/16]`
- `R_lid`: `[B,1,H/16,W/16]`
- `R_rad`: `[B,1,H/16,W/16]`

출력:

- `F_joint_fused`: `[B,384,H/16,W/16]`
- `z_joint_fused`: `[B,384]`
- `z_joint_support_weighted`: `[B,384]`
- `z_joint_residual_weighted`: `[B,384]`
- `fusion_map_summary`: `[B,64]`
- `support_strength`: `[B,1,H/16,W/16]`
- `residual_strength`: `[B,1,H/16,W/16]`


### 5.7.1 dense fusion map

먼저:

- `E_joint_map`를 `128`채널로 projection
- `R_cam/R_lid/R_rad`를 concat 후 `32`채널로 projection

한다.

```text
res_proj(E_joint_map) -> [B,128,H/16,W/16]
reli_proj(cat([R_cam,R_lid,R_rad], dim=1)) -> [B,32,H/16,W/16]
```

그 다음 아래를 concat한다.

- `F_joint_map [B,384,H/16,W/16]`
- `residual_proj [B,128,H/16,W/16]`
- `reliability_proj [B,32,H/16,W/16]`

총:

```text
[B,544,H/16,W/16]
```

이를 fuse block에 넣어:

```text
F_joint_fused = fuse(...)
shape = [B,384,H/16,W/16]
```


### 5.7.2 map-aware summaries

기존처럼 GAP token도 만든다.

```text
z_joint_fused = AdaptiveAvgPool2d(1)(F_joint_fused).flatten(1)
shape = [B,384]
```

하지만 여기서 끝내지 않는다.

support-based summary:

```text
support_strength = mean(cat([R_cam,R_lid,R_rad]), dim=1, keepdim=True)
shape = [B,1,H/16,W/16]
```

residual-based summary:

```text
residual_strength = mean(E_joint_map, dim=1, keepdim=True)
shape = [B,1,H/16,W/16]
```

그리고 weighted token:

```text
z_joint_support_weighted = weighted_token(F_joint_fused, support_strength)
shape = [B,384]

z_joint_residual_weighted = weighted_token(F_joint_fused, residual_strength)
shape = [B,384]
```

마지막으로 두 summary를 concat하여 `fusion_map_summary`를 만든다.

```text
cat([z_joint_support_weighted, z_joint_residual_weighted], dim=1)
shape = [B,768]

fusion_map_summary = summary_mlp(...)
shape = [B,64]
```

즉 이 블록은 이제:

- dense fused map
- global GAP token
- support-weighted token
- residual-weighted token
- map summary

를 동시에 유지한다.


## 5.8 Map-Aware Joint Temporal Memory

파일:

- `models/tri_joint_v3/memory.py`

클래스:

- `JointTemporalMemory`

역할:

- main temporal state는 계속 joint 하나만 유지한다.
- 다만 memory 입력을 pooled vector 하나가 아니라 map-aware summary까지 포함하도록 확장한다.

입력:

- `z_joint_fused`: `[B,384]`
- `z_joint_support_weighted`: `[B,384]`
- `z_joint_residual_weighted`: `[B,384]`
- `fusion_map_summary`: `[B,64]`
- `state["h_joint"]`: `[B,256]` 또는 `None`

출력:

- `h_joint`: `[B,256]`
- `new_state = {"h_joint": h_joint}`

어떻게 만들어지는가:

1. 네 개 입력을 concat한다.

```text
cat([
  z_joint_fused,
  z_joint_support_weighted,
  z_joint_residual_weighted,
  fusion_map_summary
], dim=1)
shape = [B,1216]
```

2. `pre` MLP로 다시 `[B,384]` joint input으로 압축한다.

```text
joint_in = pre(...)
shape = [B,384]
```

3. `GRUCell`에 넣어 `h_joint`를 만든다.

```text
h_joint = GRUCell(joint_in, prev_h_joint)
shape = [B,256]
```

즉 memory는 여전히 joint-centered이지만,
이제 support-aware / residual-aware / map-aware summary를 함께 반영한다.


## 5.9 Dense-Aware Joint Refinement

파일:

- `models/tri_joint_v3/refinement.py`

클래스:

- `JointResidualRefinement`

역할:

- fused joint latent를 residual하게 보정한다.
- 현재는 memory state뿐 아니라 alignment summary와 dense summary도 함께 본다.

입력:

- `z_joint_fused`: `[B,384]`
- `h_joint`: `[B,256]`
- `r_summary`: `[B,64]`
- `e_align_summary`: `[B,64]`
- `z_joint_support_weighted`: `[B,384]`
- `z_joint_residual_weighted`: `[B,384]`

출력:

- `z_joint_ref`: `[B,384]`
- `delta_z`: `[B,384]`
- `gate_z`: `[B,384]`
- `dense_summary`: `[B,64]`


### 5.9.1 dense summary

먼저 support-weighted / residual-weighted token을 concat하여
`dense_summary`를 만든다.

```text
cat([z_joint_support_weighted, z_joint_residual_weighted], dim=1)
shape = [B,768]

dense_summary = map_summary_proj(...)
shape = [B,64]
```


### 5.9.2 refinement trunk

그 다음 아래를 concat한다.

- `z_joint_fused [B,384]`
- `h_joint [B,256]`
- `r_summary [B,64]`
- `e_align_summary [B,64]`
- `dense_summary [B,64]`

총:

```text
[B,832]
```

이를 trunk에 통과시켜:

```text
trunk_feat [B,384]
```


### 5.9.3 residual refinement

```text
delta_z = delta(trunk_feat) -> [B,384]
gate_z = sigmoid(gate(trunk_feat)) -> [B,384]
z_joint_ref = z_joint_fused + gate_z * delta_z
```

즉 refinement는 여전히

```text
z_joint_ref = z_joint_fused + gate_z * delta_z
```

공식을 유지하지만,
그 입력이 훨씬 더 dense-aware / alignment-aware 해진 것이다.


## 5.10 Pairwise Delta Pose Readout

파일:

- `models/tri_joint_v3/heads.py`

클래스:

- `JointToPairDeltaPoseHeads`

역할:

- internal joint latent에서 마지막 단계에서만 pairwise delta pose를 읽는다.
- internal reasoning은 joint 중심이고, output만 pairwise다.

입력:

- `z_joint_ref`: `[B,384]`

출력:

- `T_CL_t [B,3]`
- `T_CL_q [B,4]`
- `T_CR_t [B,3]`
- `T_CR_q [B,4]`
- `T_LR_t [B,3]`
- `T_LR_q [B,4]`

어떻게 만들어지는가:

1. shared MLP:

```text
shared = self.shared(z_joint_ref)
shape = [B,256]
```

2. pair adapters:

```text
f_cl = adapt_cl(shared) -> [B,256]
f_cr = adapt_cr(shared) -> [B,256]
f_lr = adapt_lr(shared) -> [B,256]
```

3. 각 pair head:

```text
delta_t = fc_t(x) -> [B,3]
delta_q = normalize(fc_q(x)) -> [B,4]
```

quaternion은 `F.normalize(..., dim=1)`로 정규화된다.


## 6. 전체 forward 흐름

single-frame 입력 기준 현재 forward는 개념적으로 아래와 같다.

```python
cam = camera_encoder(rgb)
lid = lidar_encoder(lidar_proj)
rad = radar_encoder(radar_proj)

coarse = shared_context(cam["s32"], lid["s32"], rad["s32"])

joint = joint_feature(
    cam["s16"],
    lid["s16"],
    rad["s16"],
    coarse["ctx_s16"],
)

support_cam = _support_cam(rgb)
valid_lid = _valid_from_proj(lidar_proj)
valid_rad = _valid_from_proj(radar_proj[:, :1])

residual = residual_understanding(
    cam["s16"],
    lid["s16"],
    rad["s16"],
    joint["F_joint_map"],
    support_cam,
    valid_lid,
    valid_rad,
)

reli = reliability(
    joint["F_joint_map"],
    residual["E_joint_map"],
    support_cam,
    valid_lid,
    valid_rad,
    residual["e_joint"],
    residual["e_align_summary"],
)

fused = joint_fusion(
    joint["F_joint_map"],
    residual["E_joint_map"],
    reli["R_cam"],
    reli["R_lid"],
    reli["R_rad"],
)

h_joint, new_state = memory(
    fused["z_joint_fused"],
    fused["z_joint_support_weighted"],
    fused["z_joint_residual_weighted"],
    fused["fusion_map_summary"],
    state,
)

refined = refinement(
    fused["z_joint_fused"],
    h_joint,
    reli["r_summary"],
    residual["e_align_summary"],
    fused["z_joint_support_weighted"],
    fused["z_joint_residual_weighted"],
)

pred = pose_heads(refined["z_joint_ref"])
```

sequence 입력일 때는 time step마다 위 과정을 반복하며 `h_joint`를 갱신한다.


## 7. aux에 들어가는 주요 텐서

`return_aux=True`일 때 현재 모델은 아래 텐서를 제공한다.

coarse context:

- `ctx_s16`
- `z_shared`

joint alignment feature:

- `F_joint_map`
- `z_joint`
- `z_joint_support`
- `z_joint_conflict`
- `z_joint_align`
- `support_branch`
- `common_structure`
- `conflict_branch`
- `align_precursor`
- `diff_cl`
- `diff_cr`
- `diff_lr`

residual field:

- `E_joint_map`
- `e_joint`
- `e_align_summary`
- `feat_disagree`
- `support_mismatch`
- `local_align`
- `align_cl`
- `align_cr`
- `align_lr`
- `geom_residual`

support / validity:

- `support_cam`
- `valid_lid`
- `valid_rad`

reliability:

- `R_cam`
- `R_lid`
- `R_rad`
- `r_summary`
- `invalid_penalty`

fusion:

- `F_joint_fused`
- `z_joint_fused`
- `z_joint_support_weighted`
- `z_joint_residual_weighted`
- `fusion_map_summary`
- `support_strength`
- `residual_strength`

memory / refinement:

- `h_joint`
- `delta_z`
- `gate_z`
- `dense_summary`
- `z_joint_ref`


## 8. 현재 구조에서 output은 어떻게 만들어지는가

최종 output `T_CL / T_CR / T_LR`는 다음 흐름으로 만들어진다.

1. 세 센서 입력이 encoder를 거쳐 multi-scale feature가 된다.
2. `s32`로부터 shared coarse context를 만든다.
3. `s16 + ctx_s16`로부터 dense joint alignment state `F_joint_map`을 만든다.
4. `F_joint_map`과 encoder feature, support/validity를 이용해 dense residual field `E_joint_map`을 만든다.
5. `F_joint_map + E_joint_map + support/validity`를 이용해 reliability map을 만든다.
6. `F_joint_map + E_joint_map + reliability`를 다시 fuse하여 `F_joint_fused`를 만든다.
7. `F_joint_fused`에서 여러 map-aware summary를 뽑아 memory와 refinement에 사용한다.
8. refinement를 거쳐 최종 joint latent `z_joint_ref`를 만든다.
9. `z_joint_ref`에서만 pairwise delta pose를 읽는다.

즉 output은 처음부터 pair별로 직접 회귀되는 것이 아니라,
joint dense state -> residual field -> reliability -> fused joint latent -> refinement
를 거친 뒤 마지막에 pairwise head에서만 분기된다.


## 9. V3Lite 초기 버전 대비 변경점

현재 버전은 초기 `TriJointV3Lite`보다 다음이 달라졌다.

### 9.1 joint feature가 더 alignment-specific 해졌다

기존:

- support + agreement 중심

현재:

- support
- common structure
- conflict
- directional alignment precursor

를 분리해서 구성한다.


### 9.2 residual field가 더 직접적인 alignment cue를 본다

기존:

- feature disagreement
- support mismatch

현재:

- feature disagreement
- support mismatch
- local soft alignment cue
- lightweight geometric residual


### 9.3 dense map 정보를 더 오래 유지한다

기존:

- `F_joint_fused -> z_joint_fused`

현재:

- `z_joint_fused`
- `z_joint_support_weighted`
- `z_joint_residual_weighted`
- `fusion_map_summary`

를 함께 유지한다.


### 9.4 memory / refinement가 더 map-aware 해졌다

기존:

- pooled latent 중심

현재:

- support-weighted summary
- residual-weighted summary
- alignment summary
- dense summary

를 함께 사용한다.


## 10. 현재 한계와 TODO

현재 구조는 여전히 `v3.1-lite` 수준의 minimum implementation이다.

### 10.1 camera support

- `support_cam`은 아직 `rgb.abs().mean()` 기반의 단순 proxy다.
- 더 강한 structural / photometric support estimator가 필요할 수 있다.

### 10.2 geometry residual

- 현재 `geom_residual`은 LiDAR/Radar support overlap / asymmetry 기반의 lightweight cue다.
- full geometric residual은 아니다.

### 10.3 reliability regularization

- `invalid_penalty` hook는 만들어졌지만,
  현재 loss에서 직접 사용하는 구조는 아직 아니다.

### 10.4 local alignment cue 범위

- 현재 local alignment는 작은 neighborhood soft matching 수준이다.
- full cost volume main path는 아니다.


## 11. 요약

현재 `TriJointV3Lite`는 다음과 같이 이해하면 된다.

- 세 센서 feature를 joint-centered로 받아
- shared context를 만들고
- dense joint alignment feature를 구성하고
- 그 위에서 dense residual field를 만들고
- residual을 본 뒤 modality reliability를 계산하고
- joint fused state를 만들되 dense map 정보를 더 오래 유지하고
- joint memory와 dense-aware refinement를 거쳐
- 마지막에만 CL / CR / LR delta pose를 읽는 구조

즉 이 모델의 핵심은:

- joint-centered internal reasoning
- residual 뒤의 reliability
- dense alignment clue의 유지
- pairwise output only at the end

에 있다.
