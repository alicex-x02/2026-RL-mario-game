from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# Allows this file to import mario_rl when run from student_template/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mario_rl.interface import AgentMetadata, BaseAgent


# 원래 12개 action 중 자주 쓸 것만 사용
# 1: RIGHT
# 2: RIGHT_JUMP
# 3: RIGHT_RUN
# 4: RIGHT_RUN_JUMP
# 5: JUMP
# 10: DOWN
DEFAULT_REDUCED_ACTIONS = [1, 2, 3, 4, 5, 10]


class NatureCNN(nn.Module):
    """
    PPO용 CNN policy/value network.
    입력: (4, 84, 84)
    출력: reduced action logits, value
    """

    def __init__(self, in_channels: int, n_policy_actions: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.features(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
        )

        self.policy = nn.Linear(512, n_policy_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        h = self.fc(self.features(x))
        logits = self.policy(h)
        value = self.value(h)
        return logits, value


class Agent(BaseAgent):
    """
    제출용 Agent.

    평가 서버는 대략 이렇게 호출함:
        agent = Agent.load("model.pt", observation_space, action_space, device)
        action = agent.act(obs)

    내부 policy는 reduced action index를 출력하지만,
    최종 act()는 원래 12-action id를 반환한다.
    """

    TEAM_ID = "teamXX"
    MEMBERS = ["name1", "name2"]
    METHOD = "PPO_easy_stage_curriculum"
    BACKBONE = "cnn"

    def __init__(
        self,
        observation_space,
        action_space,
        device: str = "cpu",
        reduced_actions: Optional[Sequence[int]] = None,
    ):
        super().__init__(observation_space, action_space, device)

        self.reduced_actions = list(reduced_actions or DEFAULT_REDUCED_ACTIONS)

        in_channels = int(observation_space.shape[0])
        self.net = NatureCNN(
            in_channels=in_channels,
            n_policy_actions=len(self.reduced_actions),
        ).to(device)

        self.net.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        """
        평가 시 호출됨.
        observation: np.ndarray, shape=(4, 84, 84)
        return: 원래 action space의 정수 action id
        """
        obs = torch.as_tensor(np.asarray(observation), device=self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        logits, _ = self.net(obs)
        reduced_idx = int(torch.argmax(logits, dim=1).item())

        # reduced index -> original 12-action id
        action = int(self.reduced_actions[reduced_idx])

        # 혹시 모를 안전장치
        if action < 0 or action >= int(self.action_space.n):
            action = 3  # RIGHT_RUN

        return action

    @classmethod
    def load(
        cls,
        path,
        observation_space,
        action_space,
        device: str = "cpu",
    ) -> "BaseAgent":
        ckpt = torch.load(path, map_location=device)

        if isinstance(ckpt, dict):
            reduced_actions = ckpt.get("reduced_actions", DEFAULT_REDUCED_ACTIONS)
            state_dict = ckpt.get("state_dict", ckpt)
        else:
            reduced_actions = DEFAULT_REDUCED_ACTIONS
            state_dict = ckpt

        agent = cls(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            reduced_actions=reduced_actions,
        )

        agent.net.load_state_dict(state_dict)
        agent.net.eval()
        return agent

    def metadata(self) -> Optional[AgentMetadata]:
        return AgentMetadata(
            team_id=self.TEAM_ID,
            members=self.MEMBERS,
            method=self.METHOD,
            backbone=self.BACKBONE,
            extra_libraries=[],
            notes="PPO with reduced actions and easy-stage curriculum.",
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "method": self.METHOD,
                "team_id": self.TEAM_ID,
                "members": self.MEMBERS,
                "backbone": self.BACKBONE,
                "reduced_actions": self.reduced_actions,
            },
            path,
        )