"""
mario_rl - 2026 강화학습 프로젝트 공통 패키지

- interface.py : 모든 제출물이 따라야 하는 Agent 공통 인터페이스
- actions.py   : 평가에 사용하는 행동 번호/의미의 단일 기준
- env.py       : 실제 gym-super-mario-bros 환경 팩토리와 전처리
- config.py    : 평가에 사용되는 고정 설정값(평가 반복 수, 에피소드 길이 등)

학생은 이 패키지를 절대 수정하지 않는다. (수정 시 평가 환경과 불일치하여 0점 처리될 수 있음)
"""

from .actions import ACTION_MEANINGS, MARIO_MOVEMENT, describe_actions
from .config import ALL_STAGE_IDS, DEFAULT_EVAL_CONFIG, DEFAULT_TRAIN_ENV_ID, EvalConfig
from .interface import BaseAgent, AgentMetadata

__all__ = [
    "ACTION_MEANINGS",
    "MARIO_MOVEMENT",
    "describe_actions",
    "BaseAgent",
    "AgentMetadata",
    "ALL_STAGE_IDS",
    "DEFAULT_TRAIN_ENV_ID",
    "EvalConfig",
    "DEFAULT_EVAL_CONFIG",
]

__version__ = "1.0.0"
