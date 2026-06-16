# preset: full curriculum PPO | device=cuda|auto | rollout=512 | lr=2.5e-4 | progress_coef=0.04 | clear_bonus=50 | total_steps=30000 default
"""
Curriculum PPO trainer for the Super Mario RL student kit.

The default plan is:
1. Start from a weighted stage sampler that mixes early stages with the
   random-stage environment.
2. Shape reward by forward progress and a clear bonus.
3. Periodically evaluate greedy performance on a small stage set.

The evaluator only cares about agent.py + model.pt, but this script is kept
in the final submission for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent
from mario_rl.config import ALL_STAGE_IDS, DEFAULT_TRAIN_ENV_ID, EvalConfig
from mario_rl.env import make_env


@dataclass(frozen=True)
class StageSpec:
    env_id: str
    weight: float


@dataclass
class EpisodeStats:
    shaped_return: float = 0.0
    max_x: int = 0
    cleared: bool = False
    length: int = 0


class WeightedStageSampler:
    def __init__(self, specs: Sequence[StageSpec], seed: int):
        env_ids = [spec.env_id for spec in specs if spec.weight > 0]
        weights = np.asarray([spec.weight for spec in specs if spec.weight > 0], dtype=np.float64)
        if not env_ids:
            raise ValueError("curriculum has no stages")
        self.env_ids = env_ids
        self.probs = weights / weights.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self) -> str:
        return str(self.rng.choice(self.env_ids, p=self.probs))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--curriculum",
        choices=["warmup", "full", "random"],
        default="full",
        help="Stage sampling policy used during training.",
    )
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--rollout", type=int, default=512, help="Steps collected per PPO update")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--progress-coef", type=float, default=0.04)
    parser.add_argument("--clear-bonus", type=float, default=50.0)
    parser.add_argument("--death-penalty", type=float, default=-15.0)
    parser.add_argument("--timeout-penalty", type=float, default=-2.0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="model.pt")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def build_curriculum(name: str) -> list[StageSpec]:
    if name == "random":
        return [StageSpec(DEFAULT_TRAIN_ENV_ID, 1.0)]

    specs: list[StageSpec] = []
    for env_id in ALL_STAGE_IDS:
        world = int(env_id.split("-")[1])
        if name == "warmup":
            if world == 1:
                weight = 8.0
            elif world == 2:
                weight = 4.0
            else:
                weight = 2.0
        else:
            if world == 1:
                weight = 6.0
            elif world == 2:
                weight = 5.0
            elif world == 3:
                weight = 4.0
            elif world in (4, 5):
                weight = 3.0
            else:
                weight = 2.0
        specs.append(StageSpec(env_id, weight))

    random_weight = 8.0 if name == "warmup" else 12.0
    specs.append(StageSpec(DEFAULT_TRAIN_ENV_ID, random_weight))
    return specs


def make_episode_env(env_id: str, seed: int):
    env = make_env(EvalConfig(env_id=env_id))
    obs, _ = env.reset(seed=seed)
    return env, obs


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def shape_reward(
    raw_reward: float,
    info: dict,
    prev_x: int,
    terminated: bool,
    truncated: bool,
    args,
):
    x_pos = int(info.get("x_pos", prev_x))
    progress = max(0, x_pos - prev_x)
    shaped = float(raw_reward) + args.progress_coef * float(progress)
    cleared = bool(info.get("flag_get", False)) or info.get("status") == "flag_get"
    if cleared:
        shaped += args.clear_bonus
    elif terminated:
        shaped += args.death_penalty
    elif truncated:
        shaped += args.timeout_penalty
    return shaped, x_pos, cleared


@torch.no_grad()
def evaluate(agent: Agent, stage_ids: Sequence[str], episodes_per_stage: int, seed: int):
    was_training = agent.net.training
    agent.net.eval()

    results = []
    for stage_index, env_id in enumerate(stage_ids):
        for episode in range(episodes_per_stage):
            env, obs = make_episode_env(env_id, seed + stage_index * 100 + episode)
            try:
                done = False
                max_x = 0
                cleared = False
                while not done:
                    action = agent.act(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    max_x = max(max_x, int(info.get("x_pos", 0)))
                    cleared = cleared or bool(info.get("flag_get", False)) or info.get("status") == "flag_get"
                results.append((cleared, max_x))
            finally:
                env.close()

    if was_training:
        agent.net.train()

    clear_count = sum(1 for cleared, _ in results if cleared)
    progress_values = [x for _, x in results] if results else [0]
    mean_progress = float(np.mean(progress_values))
    score = clear_count * 100000.0 + mean_progress
    return {
        "clear_count": clear_count,
        "mean_progress": mean_progress,
        "score": score,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    curriculum = WeightedStageSampler(build_curriculum(args.curriculum), args.seed)
    train_env_id = curriculum.sample()
    env, obs = make_episode_env(train_env_id, args.seed)
    obs_space, act_space = env.observation_space, env.action_space
    device = resolve_device(args.device)

    print(
        f"[PPO] curriculum={args.curriculum} start_env={train_env_id} "
        f"obs={obs_space.shape} n_act={act_space.n} device={device}"
    )

    agent = Agent(obs_space, act_space, device)
    agent.net.train()
    optimizer = optim.Adam(agent.net.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    episode_index = 0
    current_env_id = train_env_id
    current_episode = EpisodeStats()
    prev_x = 0
    returns_log: list[float] = []
    best_eval_score = float("-inf")
    start = time.time()

    n_updates = max(1, args.total_steps // args.rollout)
    eval_stage_ids = [
        ALL_STAGE_IDS[0],
        ALL_STAGE_IDS[3],
        ALL_STAGE_IDS[15],
        ALL_STAGE_IDS[31],
        DEFAULT_TRAIN_ENV_ID,
    ]

    try:
        for update in range(1, n_updates + 1):
            b_obs = np.zeros((args.rollout, *obs_space.shape), dtype=np.uint8)
            b_actions = np.zeros(args.rollout, dtype=np.int64)
            b_logprobs = np.zeros(args.rollout, dtype=np.float32)
            b_rewards = np.zeros(args.rollout, dtype=np.float32)
            b_dones = np.zeros(args.rollout, dtype=np.float32)
            b_values = np.zeros(args.rollout, dtype=np.float32)

            for t in range(args.rollout):
                global_step += 1
                b_obs[t] = obs

                with torch.no_grad():
                    ot = torch.as_tensor(obs, device=device).unsqueeze(0)
                    logits, value = agent.net(ot)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                a = int(action.item())
                b_actions[t] = a
                b_logprobs[t] = float(logprob.item())
                b_values[t] = float(value.item())

                obs, reward, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                shaped_reward, prev_x, cleared = shape_reward(
                    reward,
                    info,
                    prev_x,
                    terminated,
                    truncated,
                    args,
                )

                b_rewards[t] = shaped_reward
                b_dones[t] = float(done)
                current_episode.shaped_return += shaped_reward
                current_episode.max_x = max(current_episode.max_x, int(info.get("x_pos", 0)))
                current_episode.cleared = current_episode.cleared or cleared
                current_episode.length += 1

                if done:
                    returns_log.append(current_episode.shaped_return)
                    episode_index += 1
                    env.close()
                    current_env_id = curriculum.sample()
                    env, obs = make_episode_env(current_env_id, args.seed + episode_index)
                    prev_x = 0
                    current_episode = EpisodeStats()

            with torch.no_grad():
                ot = torch.as_tensor(obs, device=device).unsqueeze(0)
                _, last_value = agent.net(ot)
                last_value = float(last_value.item())

            advantages = np.zeros(args.rollout, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(args.rollout)):
                next_nonterminal = 1.0 - b_dones[t]
                next_value = last_value if t == args.rollout - 1 else b_values[t + 1]
                delta = (
                    b_rewards[t] + args.gamma * next_value * next_nonterminal - b_values[t]
                )
                last_gae = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                )
                advantages[t] = last_gae
            returns = advantages + b_values

            t_obs = torch.as_tensor(b_obs, device=device)
            t_actions = torch.as_tensor(b_actions, device=device)
            t_logprobs = torch.as_tensor(b_logprobs, device=device)
            t_adv = torch.as_tensor(advantages, device=device)
            t_returns = torch.as_tensor(returns, device=device)
            t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

            idx = np.arange(args.rollout)
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for start_i in range(0, args.rollout, args.minibatch):
                    mb = idx[start_i : start_i + args.minibatch]
                    logits, values = agent.net(t_obs[mb])
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = dist.log_prob(t_actions[mb])
                    entropy = dist.entropy().mean()
                    ratio = (new_logprob - t_logprobs[mb]).exp()

                    pg1 = -t_adv[mb] * ratio
                    pg2 = -t_adv[mb] * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                    policy_loss = torch.max(pg1, pg2).mean()
                    value_loss = 0.5 * ((values.squeeze(-1) - t_returns[mb]) ** 2).mean()
                    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.net.parameters(), args.max_grad_norm)
                    optimizer.step()

            if update % 5 == 0 or update == n_updates:
                recent = returns_log[-10:] if returns_log else [0.0]
                sps = int(global_step / max(1.0, time.time() - start))
                print(
                    f"upd {update}/{n_updates} step {global_step} "
                    f"avg_return {np.mean(recent):7.2f} env {current_env_id} sps {sps}"
                )

            if args.eval_every > 0 and (update % args.eval_every == 0 or update == n_updates):
                metrics = evaluate(agent, eval_stage_ids, args.eval_episodes, args.seed + 10_000)
                print(
                    "[eval] "
                    f"clear {metrics['clear_count']}/"
                    f"{len(eval_stage_ids) * args.eval_episodes} "
                    f"mean_x {metrics['mean_progress']:.1f} "
                    f"score {metrics['score']:.1f}"
                )
                if metrics["score"] > best_eval_score:
                    best_eval_score = metrics["score"]
                    agent.save(args.out)
                    print(f"[PPO] best checkpoint -> {args.out}")

        if best_eval_score == float("-inf"):
            agent.save(args.out)
            print(f"[PPO] checkpoint -> {args.out}")
    finally:
        env.close()

    print(f"[PPO] final checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
