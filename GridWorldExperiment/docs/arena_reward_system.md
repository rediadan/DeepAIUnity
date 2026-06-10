# Arena Reward System

이 문서는 Unity ML-Agents PPO와 Live CNN-DQN이 같은 기준으로 비교되도록 보상 체계를 정리한다.

## Event Rewards

이벤트 보상은 게임에서 실제 의미 있는 사건이 발생했을 때 한 번만 지급된다.

| Event | Reward | Meaning |
| --- | ---: | --- |
| Item collected | `+1.0` | 아이템을 먼저 획득한 행동을 강화한다. |
| Item delivered | `+5.0` | 아이템을 베이스에 전달한 승리 행동을 가장 크게 강화한다. |
| Item lost | `-2.0` | 아이템을 들고 있다가 잃는 행동을 억제한다. |
| Forced opponent drop | `+2.0` | 상대가 아이템을 들고 있을 때 밀치기로 떨어뜨리는 행동을 강화한다. |
| Opponent delivered | `-5.0` in Python mirror | 상대가 먼저 전달한 실패 상황을 강하게 억제한다. |
| Door opened | `+0.3` | 우회 경로가 막힌 상황에서 스위치를 활용하도록 유도한다. |
| Press opened-door switch | `-0.15` | 이미 열린 문 스위치를 계속 누르는 행동을 억제한다. |
| Timeout without item | `-0.5` | 아무것도 하지 못하고 라운드를 끝내는 행동을 억제한다. |
| Invalid shove | `-0.03` | 상대가 아이템을 들고 가까이 있지 않은 상황에서 밀치기를 남발하지 못하게 한다. |

## Shared Shaping Rewards

Shaping 보상은 매 decision 또는 transition마다 계산된다. Unity의 DQN 로그, Live CNN-DQN, ML-Agents PPO가 같은 `ArenaGameManager.ComputeSharedAgentShapingReward()` 기준을 사용한다.

| Condition | Reward |
| --- | ---: |
| Step penalty | `-0.01` |
| Item target progress | `(previousDistance - currentDistance) * 0.08`, clamped to `[-0.15, +0.15]` |
| Base target progress | `(previousDistance - currentDistance) * 0.20`, clamped to `[-0.25, +0.25]` |
| Opponent target progress | `(previousDistance - currentDistance) * 0.05`, clamped to `[-0.15, +0.15]` |
| Move into wall | `-0.1` |
| Shove without useful target | `-0.03` |
| Move while shortcut is blocked | `-0.015` |
| Move toward moving obstacle danger | `-0.03` |

## Notes

- `AccumulateFrameStats()`는 이제 보상을 직접 넣지 않고 통계만 기록한다.
- 벽, 닫힌 문, 이동 장애물 벌점은 shared shaping에서 한 번만 적용된다.
- Live CNN-DQN의 frame stack 입력은 보상 체계를 바꾸지 않는다. 입력만 `[3,32,48]` single frame에서 `[12,32,48]` four-frame stack으로 바뀐다.
- RGB raster 입력은 현재 목표를 작은 마커로 강조한다. 아이템을 들면 베이스가 초록 target으로 강조되어, CNN-DQN이 `self_has_item -> return_to_base` 전환을 더 쉽게 볼 수 있다.
- RGB raster 입력의 y축은 Unity 화면과 맞춘다. 즉 Unity 월드 위쪽은 raster 이미지에서도 위쪽 행에 저장된다.
- y축 정렬 이전에 만든 checkpoint와 `.npz` dataset은 입력 의미가 다르므로 새 y축 기준 학습에 섞지 않는다.
- `ArenaGameManager`의 Return-To-Base Curriculum을 켜면 라운드 시작 시 일정 확률로 한쪽 Agent가 아이템을 든 상태에서 시작한다. 이 커리큘럼은 아이템 획득 이후 베이스 복귀 하위 과제를 따로 학습시키기 위한 것이다.
