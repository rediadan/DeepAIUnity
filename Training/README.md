# Deep AI Arena 학습 파이프라인

이 폴더에는 Unity 프로토타입을 위한 최소한의 행동 복제(Behavior Cloning) 학습 파이프라인이 들어 있습니다.

## 전체 흐름

1. Unity에서 게임을 플레이하고 `arena_training_log.jsonl` 로그를 생성합니다.
2. JSONL 로그를 학습용 데이터셋으로 변환합니다.
3. 행동 복제 모델을 학습합니다.
4. 학습된 체크포인트를 ONNX와 정규화 통계 JSON으로 내보냅니다.

## 예상 로그 위치

런타임 로거는 아래 위치에 파일을 생성합니다.

- `Application.persistentDataPath/arena_training_log.jsonl`

이 파일을 `Training/data/`로 복사하거나, 전체 경로를 직접 넘겨도 됩니다.

## 1. 로그 전처리

```bash
python preprocess_logs.py --input "C:/path/to/arena_training_log.jsonl" --output "data/arena_dataset.npz"
```

참고:

- 보상 이벤트만 있는 줄은 자동으로 무시됩니다.
- 현재는 플레이어 step 로그만 저장되도록 되어 있습니다.
- 데이터 불균형을 줄이기 위해 아무 행동도 없는 idle 프레임은 기본적으로 일부만 유지합니다.

## 2. 행동 복제 학습

```bash
python train_behavior_cloning.py --dataset "data/arena_dataset.npz" --output "models/arena_bc.pt"
```

선택 가능한 인자:

- `--epochs`
- `--batch-size`
- `--lr`
- `--hidden-size`

## 3. ONNX 내보내기

```bash
python export_onnx.py --checkpoint "models/arena_bc.pt" --output "models/arena_bc.onnx"
```

이 명령은 기본적으로 `models/arena_bc.stats.json`도 함께 생성합니다.

## Python 의존성

권장 패키지:

```bash
pip install numpy torch
```

ONNX 내보내기까지 필요하면:

```bash
pip install onnx
```

## 모델 출력

모델은 아래 출력을 예측합니다.

- 이동 logits: `Idle / MoveLeft / MoveRight`
- 점프 확률
- 드롭다운 확률
- 밀치기 확률

## Unity에서 다음 단계

ONNX를 내보낸 뒤 Unity에서는 다음 순서로 진행하면 됩니다.

- `.onnx` 파일을 `Assets/` 안으로 가져옵니다.
- 생성된 `.stats.json` 파일을 `TextAsset`으로 가져옵니다.
- Unity AI Inference / Sentis 패키지가 없다면 설치합니다.
- 두 파일을 `ArenaGhostOnnxPolicy`에 연결합니다.
- `ArenaGhostController`의 정책 모드를 `RuleBased`에서 `OnnxInference`로 변경합니다.
