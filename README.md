# Universal Random Picker - Deep AI Arena

이 프로젝트는 `신경망과 딥러닝` 과목 발표를 위한 Unity 기반 AI 아레나 실험 프로젝트입니다.

현재 방향성은 단순히 "DQN으로 사용자의 약점을 찾는 것"이 아니라, **복잡한 게임 환경에서 신경망 구조와 학습 초기화 방식이 강화학습 성능에 어떤 영향을 주는지 분석하는 것**입니다.

## Project Direction

중간 발표 피드백을 반영해 프로젝트의 중심 질문을 다음처럼 재정의했습니다.

- DQN을 사용하는 이유는 강화학습 알고리즘 자체의 우열 비교가 아니라, Q-Learning에 결합되는 신경망 구조를 실험하기 위해서입니다.
- 모방학습은 초기 탐색 시간을 줄이고 초반 학습 안정성을 높이는지 확인하는 비교 요소입니다.
- 인과 데이터는 행동 결과 예측이 아니라, 문, 스위치, 장애물 같은 환경 요인이 행동 실패와 성공에 어떤 원인으로 작용했는지 설명하기 위해 사용합니다.

따라서 최종 목표는 다음 세 가지를 조합하는 것입니다.

1. DQN 계열 모델의 신경망 구조 비교
2. 모방학습 기반 초기화 효과 분석
3. 인과 태그 기반 행동 원인 분석

## Arena v2 Environment

기존 v1 아레나는 단순 추적만으로도 어느 정도 동작할 수 있는 환경이었습니다.
v2 아레나는 학습 의미를 높이기 위해 다음 요소를 추가했습니다.

- 스위치로 열리는 지름길 문
- 문이 닫혀도 이동 가능한 우회 경로
- 상하/중앙 이동 경로를 방해하는 동적 장애물
- 문, 스위치, 장애물 상태를 포함한 38개 관측값
- 닫힌 문 열기, 스위치 상태 관측, 장애물 회피와 관련된 보상 shaping

이제 AI는 단순히 아이템을 향해 직선 이동하는 것이 아니라, 문이 닫혔는지, 스위치를 밟아야 하는지, 우회해야 하는지, 동적 장애물을 기다리거나 피해야 하는지를 함께 판단해야 합니다.

## Research Questions

발표에서 사용할 핵심 연구 질문은 다음과 같습니다.

- 복잡한 v2 환경에서 DQN의 신경망 구조에 따라 성능 차이가 발생하는가?
- Conv1D처럼 시간축 패턴을 보는 구조가 단순 MLP보다 유리한가?
- Behavior Cloning으로 초기화한 DQN은 scratch DQN보다 초반 성능과 안정성이 좋아지는가?
- `door_closed`, `switch_available`, `moving_obstacle_ahead`, `shortcut_blocked`, `detour_needed` 같은 인과 태그로 실패 원인을 설명할 수 있는가?

## Model Comparison Plan

v2 데이터 기준으로 다음 모델을 비교합니다.

- `mlp_scratch`: 기본 MLP DQN
- `mlp_bc_initialized`: Behavior Cloning 가중치로 초기화한 MLP DQN
- `dueling`: value/advantage를 분리한 Dueling DQN
- `conv1d`: `(sequence=4, features=38)` 입력에서 시간축 패턴을 보는 Conv1D DQN

v2 BC/DQN 입력 크기는 `38 features * sequence length 4 = 152`입니다.

## ML-Agents Reward Summary

현재 `ArenaMlAgent`는 다음 보상 체계를 사용합니다.

- 목표에 가까워짐: 거리 감소량 기반 보상, `-0.05 ~ +0.05` 범위로 제한
- 매 decision 시간 패널티: `-0.001`
- 벽을 향해 이동: `-0.02`
- 아이템 획득: `+2`
- 아이템 배달 성공: `+5`
- 아이템을 빼앗김: `-2`
- 상대 아이템을 밀어서 떨어뜨림: `+2`
- 스위치 활성화: 직접 보상 없음
- 닫힌 문을 스위치로 열기 성공: `+0.3`
- 이미 열린 문에 연결된 스위치를 다시 누름: `-0.05`
- 닫힌 지름길 앞에서 계속 이동: `-0.015 * Time.deltaTime`
- 동적 장애물 앞에서 계속 이동: `-0.01 * Time.deltaTime`
- 라운드 타임아웃 시 아이템 미소지: `-0.5`

## Causal Tags

로그에는 다음과 같은 인과 태그를 기록해 행동 원인을 분석합니다.

- `door_closed`
- `door_open`
- `switch_available`
- `switch_active`
- `moving_obstacle_ahead`
- `shortcut_blocked`
- `detour_needed`

이 태그들은 단순 성공률뿐 아니라 "왜 실패했는가", "문이 닫혀 있을 때 우회 또는 스위치 선택을 했는가"를 설명하는 데 사용합니다.

## Usage

Unity에서 `Assets/Scenes/DQNScene.unity`를 열고 실행합니다.

ML-Agents v2 학습:

```cmd
mlagents-learn Training\mlagents\arena_rl_only_v2.yaml --run-id arena_rl_only_v2
```

그 다음 Unity Editor에서 Play를 누릅니다.

DQN/BC v2 실험:

```cmd
Training\run_dqn_v2_experiments.bat
```

자세한 데이터 전처리, BC/DQN 학습, ONNX 내보내기 방법은 [Training/README.md](Training/README.md)를 참고합니다.

## Current Status

- v2 아레나 환경 구현 완료
- 문/스위치/동적 장애물 컴포넌트 구현 완료
- v2 관측값 38개 확장 완료
- RuleBased 기준선 v2 맵 대응 완료
- ML-Agents 관측 크기와 보상 체계 갱신 완료
- BC/DQN v2 전처리 및 Conv1D 입력 크기 갱신 완료

남은 작업은 v2 로그 수집, 모델별 학습 실행, 실험 결과 표 정리, 인과 태그 기반 실패 분석입니다.
