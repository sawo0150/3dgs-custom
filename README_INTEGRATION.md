# 3DGS Keyframe/Incremental Scaffold

이 스캐폴드는 다음 목적을 위해 만든 골격입니다.

1. `manifest -> keyframe selection -> 3DGS train` 파이프라인 분리
2. Hydra 설정 기반 실험 관리
3. W&B 로깅/아티팩트 관리 연결
4. 나중에 incremental/replay 파이프라인으로 확장 가능한 구조 확보

## 현재 상태
- `app/select_keyframes.py`: 실행 가능
- `app/export_manifest.py`: 실행 가능
- `app/train_hydra.py`: 실행 가능(단, 기존 3DGS `train.py`와 `scene` 패치 필요)
- `selection/*`: 기본 baseline 구현 완료
- `runtime/*`, `eval/*`: skeleton 위주

## 최소 통합 포인트

### 1) `scene/__init__.py` 패치
`Scene(...)` 내부에서 `scene_info`를 얻은 직후 train/test camera list를 만들기 전에 아래 후처리를 넣으면 됩니다.

```python
from data.dataset_adapter import (
    maybe_apply_keyframe_filter_to_scene_info,
    maybe_apply_pose_override_to_scene_info,
)

scene_info = maybe_apply_pose_override_to_scene_info(scene_info, args)
scene_info = maybe_apply_keyframe_filter_to_scene_info(scene_info, args)
```

그리고 `args`에 아래 필드가 있도록 합니다.
- `keyframe_list_path`
- `pose_override_path`
- `selection_keep_test`

### 2) `scene/dataset_readers.py` 패치
카메라 메타에 아래 속성을 달아주면 selection/debug가 쉬워집니다.
- `image_name`
- `frame_idx` (없으면 `-1`)
- `timestamp_ns` (없으면 `None`)
- `pose_source` (없으면 `"unknown"`)

### 3) `train.py`는 유지 가능
이 스캐폴드는 `core/trainer.py`에서 기존 `train.py`의 `training()` 함수를 import해서 래핑합니다.
즉 원본 학습 수학 로직은 유지하고, 엔트리/설정/selection만 새 레이어로 분리합니다.

## 추천 시작 순서
1. `python app/export_manifest.py ...`
2. `python app/select_keyframes.py ...`
3. `python app/train_hydra.py ... selection=pose_threshold pose=gt logging=wandb`

