"""
모든 제출물이 반드시 따라야 하는 공통 인터페이스.

핵심 아이디어
-------------
기본 학생 키트는 PPO 예시를 제공함
하지만 평가자가 자동으로 점수를 매기려면, 평가 시 호출하는 입력/출력 규격은
모두 동일해야 하므로, 아래 두 가지만 고정함

    1) act(observation) -> action         (필수)
    2) load(path, ...) -> BaseAgent        (필수, classmethod)

각 조에서는 BaseAgent 를 상속한 Agent 클래스를 student_template/agent.py 에 구현
내부 구현(신경망 구조, 학습 알고리즘)은 전혀 제약이 없음

평가 시 다음과 같이만 사용:

    from agent import Agent
    agent = Agent.load("model.pt", observation_space, action_space, device)
    action = agent.act(obs)           # obs -> 정수 행동(action)

이 규격만 지키면 어떤 알고리즘이든 동일하게 평가 됨
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AgentMetadata:
    """제출물 식별/검증을 위한 메타데이터.

    치팅 검사(동일 모델 공유 여부 등)와 재현성 확인에 사용

    """

    team_id: str  # 예: "team03"
    members: list  # 예: ["홍길동", "김철수"]
    method: str  # 예: "PPO"
    backbone: str = "cnn"  # 예: "cnn", "transformer", "mlp"
    extra_libraries: Optional[list] = None  # 사용한 외부 라이브러리
    notes: str = ""  # 자유 메모

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAgent(abc.ABC):
    """에이전트가 상속해야 하는 추상 클래스.

    평가자는 act() 와 load() 만 호출한다. 나머지는 자유.
    """

    def __init__(self, observation_space, action_space, device: str = "cpu"):
        """
        Args:
            observation_space: gymnasium observation space.
                기본 래퍼 기준: shape = (frame_stack, H, W), dtype=uint8, 값 범위 0~255.
            action_space: gymnasium Discrete action space.
                action 은 0 ~ (action_space.n - 1) 사이의 정수여야 한다.
            device: "cpu" 또는 "cuda".
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    # ------------------------------------------------------------------ #
    # 필수 구현 (평가 시 호출함)
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def act(self, observation: np.ndarray) -> int:
        """관측을 받아 하나의 이산 행동(정수)을 반환한다.

        Args:
            observation: np.ndarray, shape = observation_space.shape.
                평가 시 단일 관측(배치 차원 없음)이 들어온다.

        Returns:
            int: 0 ~ action_space.n - 1 범위의 행동.

        주의:
            - 행동은 반드시 정수여야 한다. (예: argmax 결과)
            - 이 함수 안에서 환경을 직접 만들거나 파일을 추가로 읽으면 안 된다.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(
        cls,
        path: str,
        observation_space,
        action_space,
        device: str = "cpu",
    ) -> "BaseAgent":
        """저장된 모델 파일을 읽어 평가 가능한 에이전트를 복원한다.

        Args:
            path: 학생이 제출한 모델 가중치 파일 경로(예: model.pt).
            observation_space, action_space, device: __init__ 과 동일.

        Returns:
            BaseAgent: 즉시 act() 호출이 가능한 에이전트.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 선택 구현 (있으면 치팅 검사/리포트에 활용)
    # ------------------------------------------------------------------ #
    def metadata(self) -> Optional[AgentMetadata]:
        """제출물 메타데이터를 반환(선택)."""
        return None

    def reset(self) -> None:
        """에피소드 시작 시 호출(선택). RNN/시퀀스 모델의 내부 상태 초기화 등."""
        return None
