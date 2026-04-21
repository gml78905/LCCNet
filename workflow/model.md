# Current Model Explanation

이 문서는 현재 코드베이스의 tri-modal calibration 모델 구조를 실제 구현 기준으로 정리한 설명 문서다.

현재 패키지는 `TriModalJointCalibNetV2` 중심으로 정리되어 있고,
학습 경로도 `network='TriJointV2'` 기준으로 사용한다.

핵심적으로, 현재 우리가 집중해서 봐야 하는 모델은 `TriModalJointCalibNetV2`다.  
이 모델은 direct concat -> pose regression 구조에서 벗어나, 다음 순서를 메인 경로로 가진다.

`encoders -> shared coarse context -> dense reliability maps -> reliability-aware pair evidence aggregation -> temporal memory -> residual refinement -> delta pose`


## 1. 문제 정의

현재 task는 absolute extrinsic을 직접 맞추는 것이 아니라, **현재 입력 extrinsic을 얼마나 보정해야 하는지**를 delta pose로 예측하는 문제다.

즉 모델은 corrected pose를 직접 출력하지 않는다.  
모델이 내는 것은 각 pair에 대한 delta pose다.

- `Delta T_CL`
- `Delta T_CR`
- `Delta T_LR`

여기서 의미는 다음과 같다.

- `CL`: camera -> lidar
- `CR`: camera -> radar
- `LR`: lidar -> radar

학습 시에는 입력으로 들어온 perturbed extrinsic `T_input`과 GT absolute extrinsic `T_gt`를 이용해,
loss 쪽에서 GT delta target을 계산한다.

공식은 loss 기준으로 다음과 같다.

`Delta T_gt = T_gt * inv(T_input)`

모델이 예측한 delta를 다시 input pose에 곱해서 corrected pose를 만드는 것도 모델 안이 아니라 loss/util 쪽에서 한다.

`T_corrected = Delta T_pred * T_input`

이 설계는 매우 중요하다.

- 모델은 오직 “얼마나 되돌릴 것인가”를 학습한다
- corrected pose 계산은 공용 util/loss 경로에 남겨둔다
- baseline / joint model 모두 동일한 delta-pose formulation을 공유할 수 있다


## 2. 입력 계약

현재 v2 minimum implementation은 기존 input contract를 유지한다.

- `rgb`: `[B, 3, 288, 512]`
- `lidar_proj`: `[B, 1, 288, 512]`
- `radar_proj`: `[B, 2, 288, 512]`

sequence 모드에서는 time 차원이 앞에 하나 더 붙는다.

- `rgb`: `[B, T, 3, 288, 512]`
- `lidar_proj`: `[B, T, 1, 288, 512]`
- `radar_proj`: `[B, T, 2, 288, 512]`

여기서 lidar/radar는 아직 equirectangular representation이 아니라 projection/grid representation을 쓴다.
즉, v2의 핵심 변화는 입력 포맷이 아니라 **내부 reasoning 구조**에 있다.


## 3. 큰 그림

현재 tri-joint 계열의 철학은 다음과 같다.

1. 세 센서를 모두 evidence source로 본다
2. 어떤 센서도 항상 정답이라고 가정하지 않는다
3. relation을 먼저 만든 뒤 pose를 예측한다
4. sequence/state path를 유지해서 temporal memory를 넣을 수 있게 한다
5. corrected pose는 모델 밖에서 계산한다

현재 핵심 구조는 TriJointV2이며, 메인 경로는 다음과 같다.

V2의 메인 경로는 다음과 같다.

`encoders -> shared coarse context -> dense reliability maps -> reliability-aware pair aggregation -> memory -> refinement -> delta pose`

즉 reliability가 relation 뒤에서 보조적으로 붙는 것이 아니라,
**relation을 만들기 전에 어떤 observation을 더 믿을지 고르는 역할**로 올라왔다.

이게 현재 코드에서 가장 중요한 구조 변화다.


## 4. Encoder Block

파일:

- `models/tri_joint/encoders.py`

세 개의 modality-specific multi-scale encoder를 쓴다.

- `CameraEncoderMS`
- `LidarEncoderMS`
- `RadarEncoderMS`

출력은 공통적으로 multi-scale feature dict다.

- `s16`: `[B, 256, H/16, W/16]`
- `s32`: `[B, 512, H/32, W/32]`

입력 해상도 `288 x 512` 기준이면 대략 다음 shape가 된다.

- `s16`: `[B, 256, 18, 32]`
- `s32`: `[B, 512, 9, 16]`

### Camera encoder

camera는 `torchvision`의 `resnet18` backbone을 사용한다.

- 입력: RGB 3채널
- 내부 normalize: `(x - 0.45) / 0.225`
- `layer3` 출력을 `s16`
- `layer4` 출력을 `s32`

### LiDAR / Radar encoder

LiDAR와 Radar는 `RangeEncoderMS`를 공유 구조로 사용한다.

- LiDAR 입력 채널: 1
- Radar 입력 채널: 2

둘 다 conv stem + residual block stack을 사용해서 `s16`, `s32`를 만든다.

핵심 의미는 이렇다.

- `s16`은 관계 형성에 직접 쓰는 비교적 dense한 feature
- `s32`는 coarse global context와 prior token 생성에 쓰는 더 추상적인 feature

### encoder output이 실제로 어떻게 만들어지는가

여기서 중요한 것은 `s16`, `s32`가 단순 이름이 아니라 backbone의 서로 다른 stage output이라는 점이다.

camera의 경우:

1. 입력 `rgb`가 정규화된다
2. `conv1 -> bn1 -> relu -> maxpool`을 지난다
3. `layer1`, `layer2`를 지난다
4. `layer3`의 출력을 `s16`으로 사용한다
5. `layer3` output을 다시 `layer4`에 넣고, 그 출력을 `s32`로 사용한다

즉:

- `cam["s16"] = resnet.layer3(...)`
- `cam["s32"] = resnet.layer4(cam["s16"])`

LiDAR / Radar의 경우도 같은 철학이다.

1. 입력 projection을 conv stem으로 처리한다
2. residual layer stack을 통과시킨다
3. 세 번째 stage 출력을 `s16`
4. 네 번째 stage 출력을 `s32`

즉 모든 modality에서

- `s16`은 relation formation용 dense mid-level feature
- `s32`는 coarse context / prior token용 high-level feature

로 역할이 분리되어 있다.


## 5. Shared Coarse Context

파일:

- `models/tri_joint/coarse_context.py`

V2에서는 `SharedCoarseContextV2`를 사용한다.

입력:

- `cam_s32`: `[B, 512, H/32, W/32]`
- `lid_s32`: `[B, 512, H/32, W/32]`
- `rad_s32`: `[B, 512, H/32, W/32]`

출력:

- `z_shared`: `[B, 256]`
- `ctx_s16`: `[B, 256, H/16, W/16]`
- `p_cam`: `[B, 512]`
- `p_lid`: `[B, 512]`
- `p_rad`: `[B, 512]`

### 역할

이 블록은 세 센서의 `s32` feature를 합쳐서,

- 전체 장면에 대한 shared coarse understanding을 만들고
- dense reliability head가 참고할 context map을 만들고
- 각 modality별 global prior token도 만든다

### 왜 `ctx_s16`이 필요한가

V2에서는 reliability를 modality별 dense map으로 예측한다.
그런데 reliability를 오직 자기 modality feature만 보고 예측하면, cross-modal evidence selection이라는 철학이 약해진다.

그래서 세 센서의 `s32`를 fuse한 뒤 그 결과를 `s16` 해상도로 upsample한 `ctx_s16`을 reliability head에 같이 넣는다.

즉 camera reliability도 camera feature만 보지 않고,  
LiDAR/Radar가 반영된 coarse shared context를 간접적으로 condition으로 받는다.

### output이 실제로 어떻게 만들어지는가

이 블록의 출력은 다음 순서로 만들어진다.

#### 1. `z_shared`는 어떻게 만들어지는가

먼저 세 modality의 `s32` feature를 channel 방향으로 concat한다.

- `x = cat([cam_s32, lid_s32, rad_s32], dim=1)`
- shape: `[B, 1536, H/32, W/32]`

그 다음 `self.fuse` conv stack을 통과시켜 fused coarse feature map을 만든다.

- `ctx_s32 = fuse(x)`
- shape: `[B, 256, H/32, W/32]`

마지막으로 global average pooling을 해서 1개의 shared token으로 압축한다.

- `z_shared = AdaptiveAvgPool2d(1)(ctx_s32).flatten(1)`
- shape: `[B, 256]`

즉 `z_shared`는
“세 센서의 s32 정보를 합친 coarse scene-level summary vector”다.

#### 2. `ctx_s16`은 어떻게 만들어지는가

`ctx_s16`은 `ctx_s32`를 spatially 보존한 채 `s16` 해상도로 업샘플한 결과다.

- `ctx_s16 = interpolate(ctx_s32, size=(H/16, W/16), mode='bilinear')`

shape는 다음과 같다.

- `ctx_s16`: `[B, 256, H/16, W/16]`

즉 `ctx_s16`은
“shared coarse understanding을 dense reliability head가 참조할 수 있도록 s16 grid로 올린 context map”이다.

#### 3. `p_cam`, `p_lid`, `p_rad`는 어떻게 만들어지는가

각 modality의 `s32`를 각각 별도로 global average pooling해서 만든다.

- `p_cam = pool(cam_s32).flatten(1)`
- `p_lid = pool(lid_s32).flatten(1)`
- `p_rad = pool(rad_s32).flatten(1)`

shape는 모두 `[B, 512]`다.

즉 이 값들은
각 센서의 고수준 global prior token이다.

정리하면:

- `z_shared`: 세 센서를 fuse한 공통 global token
- `ctx_s16`: fuse된 공통 context의 dense map
- `p_cam/p_lid/p_rad`: 각 modality 고유의 pooled prior token


## 6. Dense Modality Reliability Maps

파일:

- `models/tri_joint/reliability_maps.py`

V2의 핵심 변화 중 하나다.

각 modality마다 dense reliability map을 예측한다.

- `R_cam`: `[B, 1, H/16, W/16]`
- `R_lid`: `[B, 1, H/16, W/16]`
- `R_rad`: `[B, 1, H/16, W/16]`

입력:

- modality의 `s16` feature: `[B, 256, H/16, W/16]`
- shared coarse context map `ctx_s16`: `[B, 256, H/16, W/16]`

각 reliability head는 두 feature를 concat해서 conv stack으로 map을 만든다.

### 의미

이 map은 단순 confidence visualization이 아니라,
**어떤 grid cell의 observation을 relation formation에 더 강하게 사용할지**를 결정하는 weight다.

예를 들어 radar에서 ghost-like response가 많은 구간은,
shared context와 자신의 feature를 보고 상대적으로 낮은 reliability를 줄 수 있다.

### Scalar reliability도 남아 있는 이유

현재 구현에서는 map 평균으로 modality scalar도 만든다.

- `w_c`: `[B, 1]`
- `w_l`: `[B, 1]`
- `w_r`: `[B, 1]`

이 scalar는 V2의 메인 의미는 아니다.  
현재는 refinement block이 scalar conditioning을 받도록 설계되어 있어서, dense map에서 summary statistics처럼 파생해 쓰고 있다.

즉,

- 메인 evidence selection은 dense map이 담당
- refinement conditioning용 scalar는 보조 신호로 유지

### output이 실제로 어떻게 만들어지는가

각 modality reliability map은 같은 방식으로 생성된다.

예를 들어 camera 쪽은 다음과 같다.

1. `cam_s16`와 `ctx_s16`을 channel 방향으로 concat한다
2. 3x3 conv + BN + LeakyReLU를 두 번 통과한다
3. 마지막 1x1 conv로 1채널 map으로 줄인다
4. `Sigmoid`를 적용해 0~1 범위의 reliability map을 만든다

수식적으로 쓰면:

- `x_cam = cat([cam_s16, ctx_s16], dim=1)`  -> `[B, 512, H/16, W/16]`
- `R_cam = sigmoid(conv_stack(x_cam))` -> `[B, 1, H/16, W/16]`

LiDAR와 Radar도 완전히 같은 구조다.

- `R_lid = head_lid(cat([lid_s16, ctx_s16]))`
- `R_rad = head_rad(cat([rad_s16, ctx_s16]))`

그 다음 scalar summary는 spatial mean으로 만든다.

- `w_c = mean(R_cam, dim=(2,3))` -> `[B, 1]`
- `w_l = mean(R_lid, dim=(2,3))` -> `[B, 1]`
- `w_r = mean(R_rad, dim=(2,3))` -> `[B, 1]`

즉:

- `R_*`는 dense cell-wise evidence weight
- `w_*`는 그 dense map을 요약한 modality-level scalar


## 7. Reliability-Aware Pairwise Evidence Aggregation

파일:

- `models/tri_joint/evidence_aggregation.py`

이 블록이 V2의 핵심이다.

입력:

- `cam_s16`, `lid_s16`, `rad_s16`: `[B, 256, H/16, W/16]`
- `R_cam`, `R_lid`, `R_rad`: `[B, 1, H/16, W/16]`
- `p_cam`, `p_lid`, `p_rad`: `[B, 512]`
- `z_shared`: `[B, 256]`

출력:

- `F_cam_w`, `F_lid_w`, `F_rad_w`: `[B, 256, H/16, W/16]`
- `r_cl0`, `r_cr0`, `r_lr0`: `[B, 384]`
- `w_cl`, `w_cr`, `w_lr`: `[B, 1]`

### 실제 동작

먼저 modality별 weighted feature를 만든다.

- `F_cam_w = R_cam * cam_s16`
- `F_lid_w = R_lid * lid_s16`
- `F_rad_w = R_rad * rad_s16`

그 다음 reliability-weighted average pooling으로 token을 만든다.

- `tok_cam`: `[B, 256]`
- `tok_lid`: `[B, 256]`
- `tok_rad`: `[B, 256]`

수식으로 쓰면 대략 다음과 같다.

`tok = sum(F * R) / sum(R)`

즉 단순 global average pooling이 아니라, reliability가 높은 위치에 더 큰 비중을 준 pooled evidence token이다.

### Pair relation 생성

이 token들을 `s32` prior token, `z_shared`와 concat해서 각 pair coarse relation state를 만든다.

- `r_cl0 = f(tok_cam, tok_lid, p_cam, p_lid, z_shared)`
- `r_cr0 = f(tok_cam, tok_rad, p_cam, p_rad, z_shared)`
- `r_lr0 = f(tok_lid, tok_rad, p_lid, p_rad, z_shared)`

shape는 모두 `[B, 384]`다.

여기서 중요한 점은,
V2는 더 이상 raw pooled feature만으로 relation을 만들지 않는다는 것이다.  
**“무엇을 믿을지”를 먼저 고른 뒤, 그 evidence로 relation을 형성한다.**

이게 V1 대비 가장 본질적인 차이다.

### output이 실제로 어떻게 만들어지는가

이 블록의 output은 크게 세 종류다.

- weighted dense feature
- weighted pooled token
- pair relation state

#### 1. `F_cam_w`, `F_lid_w`, `F_rad_w`

가장 먼저 dense feature를 reliability map으로 element-wise weighting한다.

- `F_cam_w = cam_s16 * R_cam`
- `F_lid_w = lid_s16 * R_lid`
- `F_rad_w = rad_s16 * R_rad`

shape는 모두 `[B, 256, H/16, W/16]`다.

이 값은 “신뢰도 높은 위치가 강조된 modality feature map”이다.

#### 2. `tok_cam`, `tok_lid`, `tok_rad`

그 다음 weighted feature를 weighted average pooling해서 token으로 만든다.

- `num = sum(feat * reli, dim=(2,3))`
- `den = sum(reli, dim=(2,3))`
- `tok = num / den`

예를 들어 camera는:

- `tok_cam = sum(cam_s16 * R_cam) / sum(R_cam)`

shape는 `[B, 256]`이다.

즉 global average pooling과 달리,
reliability가 높은 위치의 feature가 더 크게 반영된다.

#### 3. `r_cl0`, `r_cr0`, `r_lr0`

이제 pair별로 필요한 token과 prior를 concat한다.

- `rel_cl_in = cat([tok_cam, tok_lid, p_cam, p_lid, z_shared], dim=1)`
- `rel_cr_in = cat([tok_cam, tok_rad, p_cam, p_rad, z_shared], dim=1)`
- `rel_lr_in = cat([tok_lid, tok_rad, p_lid, p_rad, z_shared], dim=1)`

각 shape는 `[B, 1792]`이다.

이유는 다음과 같다.

- `256 + 256 + 512 + 512 + 256 = 1792`

이 concat vector를 pair-specific MLP에 넣어 coarse relation state를 만든다.

- `r_cl0 = rel_cl(rel_cl_in)` -> `[B, 384]`
- `r_cr0 = rel_cr(rel_cr_in)` -> `[B, 384]`
- `r_lr0 = rel_lr(rel_lr_in)` -> `[B, 384]`

즉 `r_*0`는
“reliability-aware evidence를 요약한 pairwise coarse relation embedding”이다.

#### 4. `w_cl`, `w_cr`, `w_lr`

pair scalar reliability는 해당 두 modality map 평균으로 만든다.

- `w_cl = mean((R_cam + R_lid) / 2)` -> `[B, 1]`
- `w_cr = mean((R_cam + R_rad) / 2)` -> `[B, 1]`
- `w_lr = mean((R_lid + R_rad) / 2)` -> `[B, 1]`

이 값은 relation formation의 중심은 아니고,
현재 refinement block에 넣기 위한 pair-level scalar summary다.


## 8. Temporal Memory

파일:

- `models/tri_joint/memory.py`

메모리는 V1과 V2가 같은 구조를 공유한다.

구성:

- shared memory: `h_shared` `[B, 256]`
- pair memory: `h_cl`, `h_cr`, `h_lr` 각각 `[B, 256]`

구현은 `GRUCell` 기반이다.

### 입력

- shared memory input: `z_shared` `[B, 256]`
- pair memory input: `r_cl0`, `r_cr0`, `r_lr0` `[B, 384]`

### 출력

- `h_shared`: `[B, 256]`
- `h_cl`: `[B, 256]`
- `h_cr`: `[B, 256]`
- `h_lr`: `[B, 256]`
- `new_state` dict

### 의미

이 메모리는 단일 프레임 예측만 하는 구조가 아니라,
시간 축을 따라 filtered evidence를 누적할 수 있게 한다.

현재 one-batch debug에서는 `state=None`으로 시작해 zero-init이 되지만,
forward API 자체는 sequence/state path를 지원한다.

single frame에서는 state 없이 돌고,
sequence 모드에서는 time step마다 `new_state`를 다음 step에 넘기는 방식이다.

### output이 실제로 어떻게 만들어지는가

memory는 GRUCell로 업데이트된다.

#### 1. `h_shared`

shared memory는 shared coarse token을 입력으로 받는다.

- `h_shared = GRUCell(z_shared, h_shared_prev)`

shape는 `[B, 256]`이다.

즉 이전 시점까지의 공통 문맥과 현재 `z_shared`를 합쳐 새 shared memory를 만든다.

#### 2. `h_cl`, `h_cr`, `h_lr`

각 pair memory는 해당 pair coarse relation을 입력으로 받는다.

- `h_cl = GRUCell(r_cl0, h_cl_prev)`
- `h_cr = GRUCell(r_cr0, h_cr_prev)`
- `h_lr = GRUCell(r_lr0, h_lr_prev)`

shape는 모두 `[B, 256]`이다.

즉 각 pair는 자기 relation history를 따로 기억한다.

#### 3. `new_state`

최종적으로 현재 step의 memory를 dict로 묶어 반환한다.

- `new_state["h_shared"] = h_shared`
- `new_state["h_cl"] = h_cl`
- `new_state["h_cr"] = h_cr`
- `new_state["h_lr"] = h_lr`

이 `new_state`가 다음 time step의 `state`로 다시 들어간다.


## 9. Reliability in V2

파일:

- `models/tri_joint/reliability_maps.py`

V2의 중심 reliability는 dense map이다.

- `R_cam`
- `R_lid`
- `R_rad`

이 map이 relation 형성 전에 evidence weighting에 직접 들어간다.

현재 V2도 refinement block conditioning을 위해 scalar summary를 함께 쓰지만,
핵심은 dense reliability map이 relation 형성 이전 evidence weighting에 직접 들어간다는 점이다.


## 10. Residual Refinement Core

파일:

- `models/tri_joint/calibration_core.py`

이 블록은 V1과 V2가 공유한다.

입력:

- coarse pair relation: `[B, 384]`
- pair memory: `[B, 256]`
- shared memory: `[B, 256]`
- pair scalar reliability: `[B, 1]`
- 두 modality scalar reliability: `[B, 1]`, `[B, 1]`

예를 들어 CL pair는 다음 입력을 pack한다.

- `r_cl0`
- `h_cl`
- `h_shared`
- `w_cl`
- `w_c`
- `w_l`

이를 concat하면 899차원 벡터가 된다.

`384 + 256 + 256 + 1 + 1 + 1 = 899`

### shared refinement philosophy

구조는 fully independent pair block이 아니라,

- 공통 `shared_trunk`
- pair-specific `delta` head
- pair-specific `gate` head

로 이루어진다.

즉 refinement 철학은 공유하되, 최종 보정량은 pair마다 따로 낸다.

### 핵심 공식

반드시 residual refinement를 쓴다.

`r_ref = r_coarse + gate * delta_r`

여기서

- `delta_r`: 새로운 보정 제안
- `gate`: 각 relation channel별 보정 강도

다시 말해 coarse relation을 완전히 덮어쓰지 않고,
memory와 reliability를 참고해 점진적으로 다듬는 구조다.

### output이 실제로 어떻게 만들어지는가

각 pair에 대해 먼저 coarse relation, memory, reliability scalar를 하나의 벡터로 concat한다.

예를 들어 CL pair는:

- `x_cl = cat([r_cl0, h_cl, h_shared, w_cl, w_c, w_l], dim=1)`

shape는 `[B, 899]`다.

같은 방식으로 `x_cr`, `x_lr`를 만든 뒤,
공통 `shared_trunk`에 넣는다.

- `f_cl = shared_trunk(x_cl)` -> `[B, 384]`
- `f_cr = shared_trunk(x_cr)` -> `[B, 384]`
- `f_lr = shared_trunk(x_lr)` -> `[B, 384]`

그 다음 pair별 delta head와 gate head를 적용한다.

- `delta_r_cl = delta_cl(f_cl)` -> `[B, 384]`
- `gate_cl = sigmoid(gate_cl(f_cl))` -> `[B, 384]`

최종 refined relation은 residual rule로 만든다.

- `r_cl_ref = r_cl0 + gate_cl * delta_r_cl`

CR, LR도 완전히 동일하다.

즉 refinement block의 output은
“새 relation을 처음부터 다시 만든 값”이 아니라,
기존 coarse relation을 얼마나, 어느 channel에서, 얼마나 강하게 수정할지를 반영한 결과다.


## 11. Pairwise Delta Pose Heads

파일:

- `models/tri_joint/heads.py`

refined relation state에서 최종 delta pose를 예측한다.

입력:

- `r_cl_ref`, `r_cr_ref`, `r_lr_ref`: 각 `[B, 384]`

출력:

- `T_CL_t`: `[B, 3]`
- `T_CL_q`: `[B, 4]`
- `T_CR_t`: `[B, 3]`
- `T_CR_q`: `[B, 4]`
- `T_LR_t`: `[B, 3]`
- `T_LR_q`: `[B, 4]`

각 pair head는 독립 MLP를 가지고,
quaternion 출력은 `F.normalize`로 unit quaternion으로 정규화한다.

중요한 점은 여기서도 absolute pose가 아니라 delta pose만 낸다는 점이다.

### output이 실제로 만들어지는 순서

최종 output은 갑자기 바로 나오는 것이 아니라, 아래 순서로 만들어진다.

1. encoder가 modality별 multi-scale feature를 만든다
2. shared coarse context가 `z_shared`, `ctx_s16`, `p_*`를 만든다
3. dense reliability head가 `R_cam`, `R_lid`, `R_rad`를 만든다
4. reliability-aware aggregation이 weighted evidence로부터 `r_cl0`, `r_cr0`, `r_lr0`를 만든다
5. temporal memory가 `h_shared`, `h_cl`, `h_cr`, `h_lr`를 업데이트한다
6. residual refinement가 coarse relation을 `r_*_ref`로 보정한다
7. pairwise delta pose head가 각 `r_*_ref`에서 `t`와 `q`를 예측한다

즉 각 pair output은 다음 흐름을 가진다.

- `CL`: `cam/radar/lidar feature`가 아니라 최종적으로 `r_cl_ref -> head_cl -> (T_CL_t, T_CL_q)`
- `CR`: `r_cr_ref -> head_cr -> (T_CR_t, T_CR_q)`
- `LR`: `r_lr_ref -> head_lr -> (T_LR_t, T_LR_q)`

이 말은 곧, output은 단순 pooled token concat의 직접 회귀값이 아니라
**relation state를 memory와 reliability로 refinement한 뒤 읽어낸 delta pose**라는 뜻이다.

### head 내부에서 무엇이 일어나는가

각 pair head는 동일한 형태의 MLP를 가진다.

- 입력: `r_pair_ref` `[B, 384]`
- hidden MLP: `384 -> 256 -> 256`
- 출력 branch 1: translation head `256 -> 3`
- 출력 branch 2: quaternion head `256 -> 4`

즉 수식적으로 쓰면 대략 다음과 같다.

- `x_pair = MLP(r_pair_ref)`
- `delta_t = fc_t(x_pair)`
- `delta_q_raw = fc_q(x_pair)`
- `delta_q = normalize(delta_q_raw)`

translation은 3차원 벡터로 바로 나오고,
rotation은 4차원 quaternion으로 나온 뒤 정규화된다.

### 왜 quaternion을 정규화하는가

quaternion은 회전을 나타내기 위해 unit norm이어야 한다.
그래서 head가 4차원 값을 낸 뒤 `F.normalize(..., dim=1)`를 적용해 valid rotation representation이 되도록 만든다.

이 덕분에 loss에서 quaternion distance를 안정적으로 계산할 수 있다.

### output이 corrected pose가 아닌 이유

모델 output인 `T_CL_t`, `T_CL_q` 등은 이름만 보면 absolute pose처럼 보일 수 있지만,
실제로는 모두 **delta pose prediction**이다.

예를 들어 `T_CL_t`, `T_CL_q`는

- “camera-lidar의 최종 절대 extrinsic”

이 아니라

- “현재 입력 `T_CL_input`을 얼마나 보정해야 하는가”

를 뜻한다.

즉 모델의 직접 출력은 항상 `Delta T`이며,
absolute corrected pose는 model 밖에서 아래처럼 계산된다.

- `Delta T_pred = pose_to_matrix(T_*_t, T_*_q)`
- `T_corrected = Delta T_pred * T_input`

이 설계를 유지함으로써,

- 모델은 correction amount 예측에 집중하고
- geometry composition은 util/loss 경로에서 일관되게 처리할 수 있다

### pair별 output 의미

현재 pred dict의 6개 값은 다음 의미를 가진다.

- `T_CL_t`: input `T_CL_input`에 곱할 delta translation
- `T_CL_q`: input `T_CL_input`에 곱할 delta rotation quaternion
- `T_CR_t`: input `T_CR_input`에 곱할 delta translation
- `T_CR_q`: input `T_CR_input`에 곱할 delta rotation quaternion
- `T_LR_t`: input `T_LR_input`에 곱할 delta translation
- `T_LR_q`: input `T_LR_input`에 곱할 delta rotation quaternion

따라서 output을 해석할 때는
“모델이 최종 extrinsic을 반환한다”가 아니라
“모델이 입력 extrinsic을 되돌리는 correction transform을 반환한다”로 이해하는 것이 정확하다.


## 12. Forward API

현재 tri-joint 모델의 forward contract는 다음과 같다.

### Single-frame 입력

- 입력:
  - `rgb`: `[B, 3, H, W]`
  - `lidar_proj`: `[B, 1, H, W]`
  - `radar_proj`: `[B, 2, H, W]`

- 출력:
  - 기본: `(pred, new_state)`
  - `return_aux=True`: `(pred, new_state, aux)`

### Sequence 입력

- 입력:
  - `rgb`: `[B, T, 3, H, W]`
  - `lidar_proj`: `[B, T, 1, H, W]`
  - `radar_proj`: `[B, T, 2, H, W]`

- 출력:
  - 각 pred/aux 항목을 time 축으로 stack한 tensor dict
  - state는 마지막 step의 `new_state`

### pred dict

현재 loss와 연결되는 핵심 output key는 아래 6개다.

- `T_CL_t`
- `T_CL_q`
- `T_CR_t`
- `T_CR_q`
- `T_LR_t`
- `T_LR_q`

이 값들은 모두 모델의 **직접 출력값**이며, 의미는 “각 pair input extrinsic에 적용할 delta pose”다.
즉 pred dict만으로 corrected absolute pose가 완성되는 것은 아니고,
loss/util 쪽에서 input pose와 compose해야 최종 corrected pose가 된다.

예를 들어 CL pair는 다음 순서로 해석한다.

1. 모델이 `T_CL_t`, `T_CL_q`를 예측한다
2. 이것을 `Delta T_CL_pred` 행렬로 변환한다
3. `T_CL_corrected = Delta T_CL_pred * T_CL_input` 으로 corrected pose를 만든다

CR, LR도 완전히 같은 방식이다.

### new_state dict

- `h_shared`
- `h_cl`
- `h_cr`
- `h_lr`

### aux dict

V2 기준으로 다음과 같은 내부 디버그 신호를 볼 수 있다.

- context: `z_shared`, `ctx_s16`, `p_cam`, `p_lid`, `p_rad`
- dense reliability: `R_cam`, `R_lid`, `R_rad`
- weighted features: `F_cam_w`, `F_lid_w`, `F_rad_w`
- coarse relation: `r_cl0`, `r_cr0`, `r_lr0`
- scalar summary: `w_c`, `w_l`, `w_r`, `w_cl`, `w_cr`, `w_lr`
- memory: `h_shared`, `h_cl`, `h_cr`, `h_lr`
- refinement internals: `delta_r_*`, `gate_*`, `r_*_ref`


## 13. Loss 구조

파일:

- `losses_tri.py`

현재 tri 계열 loss는 `TriModalPairwiseLoss`를 사용한다.

### Pairwise delta loss

각 pair에 대해 다음 loss를 쓴다.

`L_pair = w_t * SmoothL1(delta_t_pred, delta_t_gt) + w_q * quat_distance(delta_q_pred, delta_q_gt)`

세 pair를 모두 더한다.

- `L_CL`
- `L_CR`
- `L_LR`

### GT delta target 계산

입력 pose가 absolute extrinsic이고, GT도 absolute extrinsic이므로
loss 안에서 delta target을 계산한다.

`Delta T_gt = T_gt * inv(T_input)`

즉 모델은 absolute target을 직접 맞추는 게 아니라 delta target을 학습한다.

### Loop consistency

loop consistency도 그대로 유지된다.

corrected pose들을 만든 뒤,

- `T_CL * T_LR` 이 `T_CR`와 일치해야 하고
- `T_CR * T_RL` 이 `T_CL`과 일치해야 한다

는 제약을 loss로 건다.

이 loop loss는 모델 내부가 아니라 loss에서 계산된다.

이 점도 설계적으로 중요하다.

- 모델은 delta 예측에 집중
- pose composition/consistency는 geometry-aware loss가 담당


## 14. Train Path 연결 방식

파일:

- `train_with_sacred.py`

학습 코드에서는 `network='TriJointV2'`일 때 `TriModalJointCalibNetV2`를 생성한다.

forward adapter `_tri_model_forward(...)`가 있어서,

`(pred, state)` 또는 `(pred, state, aux)` 반환을 공통적으로 처리한다.

즉 현재 train path는 크게 깨지지 않고,
joint model이 별도 network option으로 추가된 상태다.


## 15. One-Batch Debug에서 실제로 보이는 Shape

현재 one-batch debug 기준으로 확인된 대표 shape는 다음과 같다.

- `rgb`: `(2, 3, 288, 512)`
- `lidar_proj`: `(2, 1, 288, 512)`
- `radar_proj`: `(2, 2, 288, 512)`
- `T_CL_t`: `(2, 3)`
- `T_CL_q`: `(2, 4)`
- `T_CR_t`: `(2, 3)`
- `T_CR_q`: `(2, 4)`
- `T_LR_t`: `(2, 3)`
- `T_LR_q`: `(2, 4)`
- `new_state['h_shared']`: `(2, 256)`
- `new_state['h_cl']`: `(2, 256)`
- `new_state['h_cr']`: `(2, 256)`
- `new_state['h_lr']`: `(2, 256)`
- `aux['R_cam']`: `(2, 1, 18, 32)`
- `aux['r_cl0']`: `(2, 384)`

이 shape들은 현재 V2 구현이 의도한 블록 흐름과 실제로 맞아떨어진다는 강한 증거다.


## 16. 현재 코드에서 무엇이 좋아졌는가

V2 기준으로 보면, baseline 대비 좋아진 점은 다음과 같다.

### 1. direct concat -> pose regression에서 벗어남

이제 pose는 encoder pooled token의 단순 concat에서 바로 나오지 않는다.

중간에 반드시 다음 reasoning state가 들어간다.

- shared context
- dense reliability
- pair relation
- memory
- refinement

### 2. reliability가 relation 형성 전에 작동함

이전에는 reliability가 relation 뒤의 보조 신호에 가까웠다면,
V2에서는 실제 relation evidence aggregation 단계에 직접 들어간다.

즉 “무엇을 믿을지”가 “무엇을 예측할지”보다 먼저 결정된다.

### 3. delta formulation과 geometry loss를 유지함

absolute pose regression보다 calibration task에 더 자연스러운 delta formulation을 사용한다.

### 4. sequence/state 확장이 가능함

현재 memory는 단순 GRUCell 기반이지만, 구조적으로는 temporal evidence accumulation 방향이 열려 있다.


## 17. 현재 코드의 한계

현재 구현이 연구 방향과 완전히 동일한 것은 아니다. 중요한 한계도 있다.

### 1. dense reliability는 아직 modality-local head 성격이 강함

`ctx_s16` shared context를 condition으로 받긴 하지만,
아직 camera reliability를 LiDAR/Radar observation과 더 직접적으로 interaction시키는 cross-attention 수준 구조는 아니다.

### 2. pair relation은 아직 token pooling 기반

V2가 raw pooling보다는 훨씬 낫지만,
여전히 최종 pair relation은 reliability-weighted pooled token을 concat해서 MLP로 만드는 형태다.

즉 relation field 전체를 spatially reason하는 graph/attention 구조는 아직 아니다.

### 3. pair scalar reliability는 아직 mean map 기반 summary

현재 `w_cl`, `w_cr`, `w_lr`는 dense map에서 평균을 내서 만든다.
이건 v2 minimum으로는 충분하지만, 이후 더 정교하게 바뀔 수 있다.

### 4. memory는 아직 단순 GRU

현재는 구조를 세우는 단계라서 충분하지만,
긴 시계열과 observation reliability propagation을 더 잘 다루려면 이후 개선 여지가 크다.


## 18. 지금 코드 기준 한 줄 요약

현재 `TriModalJointCalibNetV2`는  
**camera / lidar / radar의 multi-scale feature를 뽑고, shared coarse context를 이용해 modality별 dense reliability map을 만든 뒤, 그 reliability로 가중된 pairwise evidence에서 coarse relation state를 형성하고, temporal memory와 residual refinement를 거쳐 pairwise delta extrinsic을 예측하는 tri-modal joint calibration 모델**이다.

그리고 corrected pose는 모델 내부가 아니라 loss/util 쪽에서 계산하며,
loop consistency loss까지 포함해 학습된다.


## 19. 실무적으로 어떻게 읽으면 되는가

현재 코드를 빠르게 이해하려면 아래 순서로 보는 것이 가장 좋다.

1. `models/tri_joint/model.py`
   - 전체 forward 흐름 확인
2. `models/tri_joint/encoders.py`
   - modality별 feature shape 확인
3. `models/tri_joint/coarse_context.py`
   - shared context와 prior token 생성 확인
4. `models/tri_joint/reliability_maps.py`
   - dense reliability map 생성 확인
5. `models/tri_joint/evidence_aggregation.py`
   - reliability-aware relation formation 확인
6. `models/tri_joint/memory.py`
   - temporal state 구조 확인
7. `models/tri_joint/calibration_core.py`
   - residual refinement rule 확인
8. `models/tri_joint/heads.py`
   - delta pose head 확인
9. `losses_tri.py`
   - delta target과 loop consistency 계산 확인
10. `train_with_sacred.py`
   - network selection과 train loop 연결 확인


## 20. 결론

현재 코드에서 가장 중요한 포인트는 다음 세 가지다.

- 모델 출력은 absolute pose가 아니라 pairwise delta pose다
- TriJointV2의 핵심 contribution은 dense reliability map을 relation 형성 이전 evidence weighting에 사용한다는 점이다
- temporal memory와 residual refinement를 통해 coarse relation을 점진적으로 보정한 뒤 최종 pose를 예측한다

즉 현재 코드는 더 이상 단순한 tri-modal direct regression baseline이 아니라,
적어도 구조적으로는 **joint estimation + evidence selection + temporal refinement** 방향으로 옮겨와 있다.
