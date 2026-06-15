from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Allows this file to import mario_rl when run from student_template/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mario_rl.interface import AgentMetadata, BaseAgent


class NatureCNN(nn.Module):
    """Small CNN policy/value network used by the PPO example."""

    def __init__(self, in_channels: int, n_actions: int):
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
        self.fc = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255.0
        h = self.fc(self.features(x))
        return self.policy(h), self.value(h)


class Agent(BaseAgent):
    """Submission entry point.

    The evaluator imports this class and calls:
        Agent.load("model.pt", obs_space, act_space, device)
        agent.act(obs)

    Students may freely replace the internals, but the class name and method
    signatures must remain compatible.
    """

    TEAM_ID = "teamXX"
    MEMBERS = ["name1", "name2"]
    METHOD = "PPO"
    BACKBONE = "cnn"

    def __init__(self, observation_space, action_space, device: str = "cpu"):
        super().__init__(observation_space, action_space, device)
        in_channels = int(observation_space.shape[0])
        self.net = NatureCNN(in_channels, int(action_space.n)).to(device)
        self.net.eval()

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> int:
        obs = torch.as_tensor(np.asarray(observation), device=self.device)
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        logits, _ = self.net(obs)
        return int(torch.argmax(logits, dim=1).item())

    @classmethod
    def load(
        cls, path, observation_space, action_space, device: str = "cpu"
    ) -> "BaseAgent":
        ckpt = torch.load(path, map_location=device)
        agent = cls(observation_space, action_space, device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        agent.net.load_state_dict(state_dict)
        agent.net.eval()
        return agent

    def metadata(self) -> Optional[AgentMetadata]:
        return AgentMetadata(
            team_id=self.TEAM_ID,
            members=self.MEMBERS,
            method=self.METHOD,
            backbone=self.BACKBONE,
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "method": self.METHOD,
                "team_id": self.TEAM_ID,
                "members": self.MEMBERS,
                "backbone": self.BACKBONE,
            },
            path,
        )
