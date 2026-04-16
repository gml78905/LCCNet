# direction.md

> 이 파일은 본 프로젝트의 장기적인 연구 방향을 정의한다.  
> 모든 세션 시작 시 가장 먼저 읽어야 한다.  
> 이 파일은 자주 수정하지 않는다.  
> 방법론, 모듈, 실험 설계는 바뀔 수 있지만, 이 파일의 방향성 자체는 쉽게 바뀌지 않아야 한다.

---

## 1. 연구 목표

본 연구의 목표는 **camera–LiDAR–radar 삼종 센서 시스템**에 대해,  
**실제 주행 환경에서 동작 가능한 online targetless extrinsic calibration framework**를 구축하는 것이다.

즉, 별도의 calibration target이나 수동 개입 없이도,  
플랫폼이 동작하는 동안 세 센서 사이의 extrinsic calibration을 지속적으로 추정하고 유지할 수 있어야 한다.

이 연구는 단순히 특정 네트워크 구조를 개선하는 것이 목적이 아니다.  
본질적으로는 calibration 문제를 **불완전한 관측 하에서의 joint estimation 문제**로 다시 정의하는 데 목적이 있다.

---

## 2. 핵심 문제 설정

본 연구는 다음과 같은 설정을 다룬다.

- 센서 구성: **camera, LiDAR, radar**
- calibration 종류: **extrinsic calibration**
- 동작 방식: **online**
- supervision 스타일: **targetless**
- 환경: **실제 야외 주행 환경**
- 현재 가정:
  - timestamp는 우선 맞는다고 가정한다
  - 관측은 noisy하고 imperfect하다
  - 각 센서는 서로 보완적이지만 동시에 불완전한 evidence를 제공한다

핵심 어려움은, calibration이 모든 관측을 동일하게 신뢰하는 방식으로는 풀릴 수 없다는 점이다.

---

## 3. 본 연구의 기본 관점

본 연구는 삼종 센서 calibration을 다음과 같이 본다.

> **관측 신뢰도 추정을 포함한 joint extrinsic estimation 문제**

즉, 이를 단순히

> 단일 프레임에서의 직접 회귀 문제  
> 혹은 완전히 신뢰 가능한 입력 사이의 단순 matching 문제

로 보지 않는다.

camera, LiDAR, radar는 모두 calibration에 유용한 evidence를 제공하지만,  
각 센서는 서로 다른 방식으로 실패할 수 있다.  
따라서 모델은 단순히 센서를 결합하는 것이 아니라,

- **무엇을 믿을지**
- **언제 믿을지**
- **시간에 따라 어떻게 누적할지**

를 함께 학습해야 한다.

---

## 4. 변하지 않아야 하는 연구 방향

아래 원칙들은 본 연구의 안정적인 방향성을 정의한다.

### 4.1 어떤 센서도 항상 정답이라고 가정하지 않는다

어떤 센서도 모든 상황에서 완전히 신뢰할 수 있는 센서로 두지 않는다.

- camera는 저조도, glare, blur, appearance 변화에 취약할 수 있다.
- LiDAR는 dynamic object, sparse한 먼 구조, 불안정한 return에서 약해질 수 있다.
- radar는 sparse하고 noisy하며, ghost나 unstable detection이 발생할 수 있다.

따라서 모델은 sensor identity 자체보다,  
**observation quality**를 중심으로 판단해야 한다.

---

### 4.2 세 센서를 모두 보완적인 evidence source로 사용한다

이 연구는 한 센서를 메인으로 두고 나머지를 단순 보조로 두는 방향을 지향하지 않는다.

대신 다음과 같이 본다.

- **camera**는 dense한 visual structure와 semantic cue를 준다
- **LiDAR**는 안정적인 geometric structure를 준다
- **radar**는 motion-related cue와 환경 강인성을 제공한다

즉 calibration framework는 세 센서의 강점을 공동으로 활용해야 한다.

---

### 4.3 Calibration은 단일 프레임만으로 결정되지 않아야 한다

단일 프레임만으로는 다음을 안정적으로 판단하기 어렵다.

- 어떤 관측이 static인지 dynamic인지
- 어떤 radar return이 stable한지 transient한지
- 어떤 영역이 calibration에 유효한지
- mismatch가 extrinsic 때문인지 bad evidence 때문인지

따라서 calibration은 framewise independent prediction이 아니라,  
**temporal evidence accumulation**을 사용해야 한다.

---

### 4.4 Temporal information은 raw frame stacking이 아니라 memory로 다뤄야 한다

과거 프레임들을 단순히 input 채널처럼 쌓아 넣는 구조를 지향하지 않는다.

대신 모델은 내부 memory를 유지하며 다음을 누적해야 한다.

- persistent한 calibration evidence
- observation reliability의 추세
- static / unstable structure에 대한 힌트
- 최근의 tri-modal consistency context

즉 online하고 효율적인 구조를 지향한다.

---

### 4.5 Reliability는 refined calibration 이전에 추정되어야 한다

모델은 먼저 coarse calibration context를 만들고,  
그 context 아래에서 어떤 관측이 신뢰할 만한지 추정한 뒤,  
그 reliability를 반영하여 calibration을 refinement해야 한다.

즉, reasonable한 calibration hypothesis가 생기기 전부터  
정밀한 cross-modal agreement를 강요하지 않는다.

---

### 4.6 Framework는 targetless하고 practical해야 한다

방법은 다음에 의존하지 않아야 한다.

- calibration board
- 특수한 calibration trajectory
- 강한 manual initialization
- 완벽한 측정을 전제로 한 비현실적 가정

즉 실제 차량 시스템에서 운영 가능한 방법을 지향한다.

---

## 5. 이 연구가 궁극적으로 풀고자 하는 질문

본 연구가 답하고자 하는 핵심 질문은 다음과 같다.

> 모든 센서가 불완전하고, 서로 다른 failure mode를 가지며, calibration evidence 또한 시간에 따라 선택되어야 하는 상황에서, camera–LiDAR–radar extrinsic calibration을 online으로 어떻게 안정적으로 유지할 수 있는가?

성공적인 결과는 단순히 한 benchmark에서 낮은 error를 얻는 것이 아니다.  
성공적인 결과는 다음을 만족하는 framework이다.

- online으로 동작하고
- tri-modal evidence를 공동으로 사용하며
- unreliable observation에 강인하고
- fragile한 single-frame formulation을 넘어서며
- 구조적으로 확장 가능하다

---

## 6. 모델 수준에서의 방향성

모델 수준에서는 아래 구조적 방향을 유지한다.

1. **modality-specific encoders**
2. **coarse tri-modal calibration context formation**
3. **calibration-conditioned reliability estimation**
4. **reliability-weighted refined calibration**
5. **pairwise extrinsic outputs**
6. **temporal memory를 통한 online evidence accumulation**

이 구조는 특정 middle block 구현을 강제하지는 않는다.  
cost volume, relation embedding, graph interaction, recurrent refinement 등은 바뀔 수 있다.

하지만 아래 원칙은 유지한다.

> coarse context를 먼저 만들고, reliability를 그 다음에 추정하며,  
> 그 reliability를 바탕으로 refined calibration을 수행하고,  
> 이 모든 과정은 시간에 따라 memory를 통해 누적된다.

---

## 7. 바뀔 수 있는 것

다음은 구현 수준의 선택이므로 얼마든지 바뀔 수 있다.

- encoder의 정확한 architecture
- ResNet, ConvNeXt 등 backbone 선택
- relation module의 구체적 설계
- graph interaction의 세부 구조
- reliability parameterization
- loss 구성
- pairwise latent를 쓸지 shared latent를 쓸지
- dataset-specific preprocessing
- projection 방식의 세부 구현
- training strategy
- ablation setup

즉 이것들은 연구 방향 자체가 아니라 구현 선택이다.

---

## 8. 쉽게 바뀌면 안 되는 것

다음은 강한 근거 없이 바꾸지 않는다.

- 본 연구는 **online targetless tri-modal extrinsic calibration**을 다룬다
- 문제를 **unreliable observation 하의 joint estimation**으로 본다
- temporal information은 필수적이다
- observation reliability는 필수적이다
- 세 센서는 모두 보완적인 evidence source이다
- 실제 환경에서 deploy 가능한 구조를 지향한다

---

## 9. 하지 않으려는 것

본 연구의 주목적은 아래가 아니다.

- camera-only 또는 LiDAR-only calibration을 가장 잘 만드는 것
- 완벽한 sensor alignment를 미리 가정하는 것
- handcrafted rule만으로 문제를 푸는 것
- practical하지 않은 방식으로 benchmark score만 높이는 것
- 하나의 dataset에만 강하게 결합된 일회성 architecture를 만드는 것

---

## 10. 평가 철학

방법의 가치는 단순히 최종 calibration error만으로 판단하지 않는다.

다음도 함께 중요하다.

- unreliable observation에 대한 robustness
- 시간에 따른 안정성
- dynamic scene에서의 거동
- sensor-specific degradation 상황에서의 거동
- sensor pair 간 consistency
- 실제 online deployment 가능성

---

## 11. 장기 비전

장기적으로 본 연구는 다음 방향으로 나아가야 한다.

> ideal input 하에서의 framewise sensor-pair calibration

에서

> imperfect real-world evidence 하에서의 persistent tri-modal calibration

으로 이동하는 것.

즉 궁극적으로는, 실제 주행 환경에서 calibration을 **감지하고, 업데이트하고, 유지하는 시스템**으로 발전하는 것이 목표다.

---

## 12. 세션 규칙

모든 연구 세션 시작 시:

1. 이 파일을 가장 먼저 읽는다
2. 이 파일을 프로젝트의 최상위 direction으로 간주한다
3. 장기 목표 자체가 바뀌지 않는 한 이 파일은 수정하지 않는다

---

## 13. 한 줄 요약

본 프로젝트는 **unreliable한 multi-sensor observation 하에서, temporal memory를 활용해 camera–LiDAR–radar extrinsic calibration을 online으로 유지하는 targetless tri-modal calibration framework**를 연구한다.