from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _build_all_stage_ids() -> List[str]:
    """1-1 ~ 8-4 까지 32개 개별 스테이지 env id 목록."""
    ids = []
    for world in range(1, 9):
        for stage in range(1, 5):
            ids.append(f"SuperMarioBros-{world}-{stage}-v0")
    return ids


ALL_STAGE_IDS: List[str] = _build_all_stage_ids()
DEFAULT_TRAIN_ENV_ID: str = "SuperMarioBrosRandomStages-v0"


@dataclass(frozen=True)
class EvalConfig:
    """평가 실행 조건.

    Attributes:
        env_id: 단일 환경 생성에 사용할 gym-super-mario-bros 환경 ID.
            개별 스테이지 평가나 학습 시 make_env 에 전달된다.
        eval_seeds: 단일 스테이지 반복 평가 seed 목록.
        eval_env_ids: 전체 스테이지 평가 시 사용할 32개 스테이지 목록.
        stage_seeds: 전체 스테이지 평가에서 각 스테이지마다 사용할 seed 목록.
        max_steps_per_episode: 한 에피소드 최대 스텝(무한 루프 방지).
        frame_skip: 프레임 스킵(행동 반복) 수.
        frame_stack: 누적할 프레임 수(시간 정보 제공).
        resize: 관측 이미지 리사이즈 크기 (H, W).
        grayscale: 흑백 변환 여부.
        per_stage_time_limit_seconds: 스테이지 1개 평가의 wall-clock 제한(초).
        time_limit_seconds: 단일 스테이지 평가의 wall-clock 제한(초). 초과 시 중단.
        video_fps: 평가 영상 저장 시 사용할 FPS.
    """

    env_id: str = DEFAULT_TRAIN_ENV_ID
    eval_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    eval_env_ids: List[str] = field(default_factory=lambda: list(ALL_STAGE_IDS))
    stage_seeds: List[int] = field(default_factory=lambda: [0])
    max_steps_per_episode: int = 4000
    frame_skip: int = 4
    frame_stack: int = 4
    resize: tuple = (84, 84)
    grayscale: bool = True
    per_stage_time_limit_seconds: float = 120.0
    time_limit_seconds: float = 600.0
    video_fps: int = 15


DEFAULT_EVAL_CONFIG = EvalConfig()
