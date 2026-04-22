# experiments.md

> 이 파일은 프로젝트의 실험 workflow를 관리하는 인덱스 파일이다.  
> 각 실험의 자세한 결과, 로그, 시각화, 실패 원인 분석은 개별 실험 문서에 기록한다.  
> 이 파일에는 각 실험의 핵심 성능, 속도, 상태, 짧은 요약만 남긴다.  
> 모든 세션에서 direction.md를 먼저 읽고, 그 다음 이 파일을 읽는다.

---

## 1. 목적

이 파일의 목적은 다음과 같다.

- 지금까지 수행한 실험들을 한눈에 관리한다
- 현재 workflow에서 어떤 단계까지 왔는지 빠르게 파악한다
- 각 실험의 핵심 성능과 상태를 요약한다
- 다음에 어떤 실험을 해야 할지 우선순위를 정한다
- 이미 해본 실험을 다시 제안하지 않도록 한다

즉 이 파일은 **상세 실험 보고서**가 아니라,  
**실험 workflow 관리 파일**이다.

---

## 2. 기록 원칙

### 2.1 이 파일에 적는 것
각 실험에 대해 아래만 적는다.

- 실험 이름
- 상태
- 목적
- 핵심 변경점
- 핵심 성능
- 속도 / 자원 사용량
- 짧은 분석 요약
- 상세 문서 링크 또는 파일명
- 다음 액션

### 2.2 이 파일에 적지 않는 것
아래는 개별 실험 문서에 기록한다.

- epoch별 세부 로그
- train / val curve 상세
- 정량 결과 표 전체
- 정성 시각화
- 실패 원인 상세 분석
- 각 pair별 세부 해석
- ablation의 자세한 비교표
- 긴 실험 메모

즉 이 파일은 **요약만**,  
자세한 내용은 **각 실험 파일**에 둔다.

---

## 3. 실험 문서 구조

각 실험은 별도 문서로 관리한다. 예를 들면:

- `exp_000_baseline_triple_input.md`
- `exp_001_loop_consistency.md`
- `exp_002_recurrent_context.md`

각 실험 문서에는 다음을 자세히 기록한다.

- 설정
- 데이터
- 학습 로그
- metric
- 속도
- GPU 사용량
- 정성 결과
- 분석
- 실패 원인
- 결론

즉:
- `experiments.md` = 실험 인덱스 / workflow 요약
- `exp_xxx_*.md` = 실험 상세 보고서

---

## 4. workflow 구조

실험은 workflow 단위로 관리한다.

예시:

1. 최소 tri-modal baseline 구축
2. baseline 안정화
3. tri-modal consistency 추가
4. temporal context 추가
5. reliability 추가
6. calibration-conditioned reliability refinement

각 workflow 단계 아래에 해당 실험들을 요약해서 기록한다.

---

## 5. 기록 형식

각 실험은 아래 형식으로 요약한다.

---

### [실험 이름]

- 상태: KEEP / MAYBE / FAIL / DO NOT REPEAT
- 목적:
- 핵심 변경점:
- 핵심 성능:
- 속도 / 자원:
- 요약 분석:
- 상세 기록:
- 다음 액션:

---

## 6. Workflow

### Stage 0. 최소 tri-modal baseline 구축

#### [Baseline-0: TriBaseline one-batch]
- 상태: KEEP
- 목적: dataset → dataloader → model → loss → backward 최소 경로 검증
- 핵심 변경점: tri-modal input contract 연결, pairwise head 3개, tri-modal loss 추가
- 핵심 성능: one-batch forward/backward 성공
- 속도 / 자원: debug run, batch size 1
- 요약 분석: 최소 파이프라인 연결에는 성공했으며, 모델 입출력 계약이 유효함
- 상세 기록: `exp_000_tribaseline_onebatch.md`
- 다음 액션: full train/val loop로 확장

#### [Baseline-1: TriBaseline short training]
- 상태: KEEP
- 목적: 실제 epoch 단위 train/val loop 검증
- 핵심 변경점: tri_run_one_batch=False, full train/val loop 연결
- 핵심 성능: train/val loss 정상 감소, checkpoint 저장 성공
- 속도 / 자원: batch size 80 기준 epoch당 약 500초 수준
- 요약 분석: baseline 모델은 실제 학습이 가능하며, 최소한의 tri-modal calibration 모델로 동작함
- 상세 기록: `exp_001_tribaseline_training.md`
- 다음 액션: pairwise loss 분해 및 baseline 안정성 분석

---

### Stage 1. Baseline 안정화

#### [Baseline-2: Pairwise loss decomposition]
- 상태: TODO
- 목적: CL / CR / LR 각 pair의 난이도와 학습 추세 분리 분석
- 핵심 변경점: total loss 외에 pairwise loss logging 추가
- 핵심 성능: 미실행
- 속도 / 자원: 미기록
- 요약 분석: total loss만으로는 어떤 pair가 문제인지 알 수 없으므로 우선 수행 필요
- 상세 기록: `exp_002_pairwise_loss_breakdown.md`
- 다음 액션: quaternion norm 및 각 pair 성능 기록

---

### Stage 2. Tri-modal consistency 추가

#### [Loop-0: Loop consistency loss]
- 상태: TODO
- 목적: pairwise-only baseline 대비 tri-modal consistency 효과 확인
- 핵심 변경점: loop consistency loss 추가
- 핵심 성능: 미실행
- 속도 / 자원: 미기록
- 요약 분석: 구현 난도가 낮고 tri-modal 구조 의미를 가장 직접적으로 살릴 수 있는 다음 실험
- 상세 기록: `exp_003_loop_consistency.md`
- 다음 액션: baseline 대비 비교

---

### Stage 3. Temporal context 추가

#### [Temporal-0: Recurrent calibration context]
- 상태: TODO
- 목적: framewise prediction 대비 memory 기반 calibration context의 효과 확인
- 핵심 변경점: shared calibration memory 추가
- 핵심 성능: 미실행
- 속도 / 자원: 미기록
- 요약 분석: reliability 이전에 temporal context를 먼저 안정화하는 것이 우선
- 상세 기록: `exp_004_recurrent_context.md`
- 다음 액션: loop consistency와의 조합 실험

---

### Stage 4. Reliability 추가

#### [TriJointV2-0: Hercules full training]
- 상태: keep
- 목적: TriJointV2 full architecture를 Hercules tri-modal sequence setting에서 end-to-end로 학습하고 pairwise calibration 성능 및 안정성을 확인
- 핵심 변경점: TriJointV2 main path(`coarse context -> dense reliability -> reliability-aware evidence aggregation -> memory -> refinement -> delta pose`)를 full training loop에 적용
- 핵심 성능: best `val loss = 0.387` at epoch `112`; best corrected error `CL 18.106 cm / 0.916 deg`, `CR 40.785 cm / 3.700 deg`, `LR 44.604 cm / 3.932 deg`
- 속도 / 자원: `120 epoch`, total `12.19 hr`, late epoch time `351-358 sec`, `batch size 36`, `seq_len 4`, input `288x512`, AMP on
- 요약 분석: TriJointV2는 안정적으로 학습되며 CL 개선은 뚜렷하지만 CR/LR은 여전히 약하다. `epoch 118-119`에서 validation이 크게 흔들려 final checkpoint보다 best checkpoint 사용이 필수다.
- 상세 기록: [`results/tri_joint_v2_hercules_training_20260422/summary.md`](/home/gml78905/Project/LG/LCCNet/results/tri_joint_v2_hercules_training_20260422/summary.md)
- 다음 액션: baseline/V1 대비 정량 비교, CR/LR 약세 원인 분석, late-epoch instability 완화

#### [Reliability-0: Pair-level confidence]
- 상태: TODO
- 목적: 모든 pair를 동일하게 신뢰하지 않는 구조의 효과 확인
- 핵심 변경점: pair-level scalar confidence 추가
- 핵심 성능: 미실행
- 속도 / 자원: 미기록
- 요약 분석: 첫 reliability 실험은 spatial map보다 단순한 scalar confidence로 시작
- 상세 기록: `exp_005_pair_confidence.md`
- 다음 액션: calibration-conditioned refinement로 확장

---

### Stage 5. Calibration-conditioned reliability refinement

#### [Reliability-1: Context-conditioned reliability]
- 상태: TODO
- 목적: coarse calibration context를 기반으로 reliability를 refinement하는 구조 검증
- 핵심 변경점: self proposal + coarse-conditioned refinement
- 핵심 성능: 미실행
- 속도 / 자원: 미기록
- 요약 분석: 본 프로젝트의 핵심 아이디어에 해당하므로 충분히 안정화된 baseline 이후에 수행
- 상세 기록: `exp_006_context_conditioned_reliability.md`
- 다음 액션: observation-level reliability map으로 확장 여부 검토

---

## 7. 상태 태그 기준

- `planned`: 아직 실행 전
- `running`: 현재 실행 중이거나 결과를 정리 중인 상태
- `keep`: 의미 있는 결과가 있었고, 이후 방향에 반영할 가치가 있음
- `discard`: 현재 방향에서는 다시 수행할 필요가 낮음
- `revisit_later`: 지금은 우선순위가 낮지만, 나중에 다시 검토할 가치가 있음
- `inconclusive`: 결과가 애매해서 명확한 결론을 내리기 어려움

---

### 8. 상태 태그 사용 원칙

- 하나의 실험에는 기본적으로 하나의 대표 상태 태그를 둔다.
- 실험이 아직 끝나지 않았으면 `running`을 사용한다.
- 실험이 끝났지만 해석이 불충분하면 `inconclusive`를 사용한다.
- 성능이 낮더라도 교훈이 크고 이후 설계에 반영할 가치가 있으면 `keep`이다.
- 지금 기준으로 가치가 낮아도, 특정 조건이 바뀌면 다시 볼 수 있으면 `revisit_later`를 사용한다.
- 현재 방향성과 맞지 않거나 충분히 반복되어 더 이상 수행할 필요가 없으면 `discard`를 사용한다.

---

## 9. 실험 제안 규칙

새 실험을 제안할 때는 반드시 아래를 확인한다.

1. workflow 상 어느 stage에 속하는가?
2. 이미 같은 목적의 실험을 한 적이 있는가?
3. 상세 분석은 별도 실험 문서에 기록할 것인가?
4. 이 파일에는 요약만 남길 수 있는가?

즉 새로운 실험을 제안할 때는  
`experiments.md`에는 **요약 엔트리만 추가**하고,  
실제 분석은 새로운 실험 문서를 만들어 기록한다.

---

## 10. 세션 규칙

모든 실험 세션에서:

1. direction.md를 먼저 읽는다
2. experiments.md를 읽어 현재 workflow 상태를 확인한다
3. 상세 결과는 반드시 각 실험 문서에 남긴다
4. experiments.md에는 성능, 속도, 요약 분석만 업데이트한다

---

## 11. 한 줄 요약

이 파일은 **실험 상세 보고서를 모아두는 곳이 아니라, 각 실험의 핵심 결과와 상태를 workflow 관점에서 관리하는 실험 인덱스 파일**이다.
