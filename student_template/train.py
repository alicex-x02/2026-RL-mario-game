"""
Easy-stage curriculum PPO for Super Mario RL.

핵심 전략:
1. 전체 32개 랜덤 스테이지 학습 X
2. 쉬운 스테이지 subset만 단계적으로 학습
3. 내부 action space 축소
4. x_pos 진행 보상 + flag_get 대형 보상
5. 최종 제출은 agent.py + model.pt 하나로 가능
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mario_rl.env import make_env
from mario_rl.config import EvalConfig
from agent import Agent


DEFAULT_EASY_STAGES = [
    "SuperMarioBros-1-1-v0",
    "SuperMarioBros-1-2-v0",
    "SuperMarioBros-2-1-v0",
    "SuperMarioBros-3-1-v0",
    "SuperMarioBros-4-1-v0",
]


class MultiStageEnv:
    """
    여러 개의 개별 Mario stage env를 만들어두고,
    episode마다 하나를 골라 학습하는 wrapper.

    gym API와 완전히 동일하지는 않지만,
    train.py 내부에서 쓰기에는 충분함.
    """

    def __init__(self, stage_ids, frame_seed: int = 0):
        self.stage_ids = list(stage_ids)
        if not self.stage_ids:
            raise ValueError("stage_ids must not be empty")

        self.envs = {}
        for sid in self.stage_ids:
            cfg = EvalConfig(env_id=sid)
            self.envs[sid] = make_env(cfg)

        self.current_stage = self.stage_ids[0]
        self.current_env = self.envs[self.current_stage]

        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space
        self.frame_seed = int(frame_seed)

    def reset_to_stage(self, stage_id: str, seed=None):
        self.current_stage = stage_id
        self.current_env = self.envs[stage_id]
        return self.current_env.reset(seed=seed)

    def step(self, action: int):
        return self.current_env.step(action)

    def close(self):
        for env in self.envs.values():
            env.close()


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--stages",
        default=",".join(DEFAULT_EASY_STAGES),
        help="학습에 사용할 스테이지 목록. comma-separated.",
    )

    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--rollout", type=int, default=1024, help="업데이트당 수집 스텝")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Reward shaping
    p.add_argument("--x-coef", type=float, default=0.10)
    p.add_argument("--base-reward-coef", type=float, default=0.05)
    p.add_argument("--flag-bonus", type=float, default=10000.0)
    p.add_argument("--death-penalty", type=float, default=300.0)
    p.add_argument("--stuck-penalty", type=float, default=0.5)
    p.add_argument("--stuck-threshold", type=int, default=25)

    # Curriculum
    p.add_argument(
        "--phase-steps",
        type=int,
        default=0,
        help="몇 step마다 학습 stage를 하나씩 추가할지. 0이면 total_steps / stage_count.",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="model.pt")
    p.add_argument("--save-every", type=int, default=200_000)

    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return p.parse_args()


def get_stage_pool(stage_ids, global_step: int, phase_steps: int):
    """
    curriculum:
    처음에는 1개 stage만,
    시간이 지나면 2개,
    그다음 3개...
    """
    if phase_steps <= 0:
        return stage_ids

    n = min(len(stage_ids), 1 + global_step // phase_steps)
    return stage_ids[:n]


def choose_stage(stage_pool, rng: random.Random):
    return rng.choice(stage_pool)


def compute_shaped_reward(
    raw_reward: float,
    info: dict,
    done: bool,
    prev_x: int,
    stuck_count: int,
    args,
):
    """
    x 진행도 + flag_get 보너스 중심 reward shaping.
    """
    x = int(info.get("x_pos", 0))
    delta_x = max(0, x - prev_x)
    new_prev_x = max(prev_x, x)

    flag_get = bool(info.get("flag_get", False))

    shaped = 0.0
    shaped += args.base_reward_coef * float(raw_reward)
    shaped += args.x_coef * float(delta_x)

    if flag_get:
        shaped += args.flag_bonus

    # done인데 flag_get이 아니면 죽었거나 시간초과라고 보고 패널티
    if done and not flag_get:
        shaped -= args.death_penalty

    # 너무 오래 제자리면 약한 패널티
    if delta_x <= 0:
        stuck_count += 1
    else:
        stuck_count = 0

    if stuck_count >= args.stuck_threshold:
        shaped -= args.stuck_penalty

    return float(shaped), new_prev_x, stuck_count, flag_get, x


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    stage_ids = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stage_ids:
        stage_ids = list(DEFAULT_EASY_STAGES)

    phase_steps = args.phase_steps
    if phase_steps <= 0:
        phase_steps = max(1, args.total_steps // max(1, len(stage_ids)))

    print("[PPO] Easy-stage curriculum")
    print(f"[PPO] stages={stage_ids}")
    print(f"[PPO] phase_steps={phase_steps}")
    print(f"[PPO] total_steps={args.total_steps}")
    print(f"[PPO] device={args.device}")

    env = MultiStageEnv(stage_ids, frame_seed=args.seed)
    obs_space = env.observation_space
    act_space = env.action_space
    device = args.device

    agent = Agent(obs_space, act_space, device)
    agent.net.train()

    optimizer = optim.Adam(agent.net.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    update = 0

    returns_log = []
    clears_log = []
    stage_return_log = defaultdict(list)
    stage_clear_log = defaultdict(int)
    stage_episode_log = defaultdict(int)
    best_x_by_stage = defaultdict(int)

    # 첫 episode 시작
    pool = get_stage_pool(stage_ids, global_step, phase_steps)
    current_stage = choose_stage(pool, rng)
    obs, _ = env.reset_to_stage(current_stage, seed=args.seed)

    ep_return = 0.0
    ep_len = 0
    prev_x = 0
    stuck_count = 0

    start_time = time.time()

    n_updates = max(1, args.total_steps // args.rollout)

    for update in range(1, n_updates + 1):
        b_obs = np.zeros((args.rollout, *obs_space.shape), dtype=np.uint8)

        # b_actions는 reduced action index를 저장한다.
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

                reduced_action = dist.sample()
                logprob = dist.log_prob(reduced_action)

            reduced_idx = int(reduced_action.item())
            real_action = int(agent.reduced_actions[reduced_idx])

            b_actions[t] = reduced_idx
            b_logprobs[t] = float(logprob.item())
            b_values[t] = float(value.item())

            next_obs, raw_reward, terminated, truncated, info = env.step(real_action)
            done = bool(terminated or truncated)

            shaped_reward, prev_x, stuck_count, flag_get, x = compute_shaped_reward(
                raw_reward=raw_reward,
                info=info,
                done=done,
                prev_x=prev_x,
                stuck_count=stuck_count,
                args=args,
            )

            b_rewards[t] = shaped_reward
            b_dones[t] = float(done)

            ep_return += shaped_reward
            ep_len += 1
            best_x_by_stage[current_stage] = max(best_x_by_stage[current_stage], x)

            obs = next_obs

            if done:
                returns_log.append(ep_return)
                clears_log.append(1 if flag_get else 0)

                stage_return_log[current_stage].append(ep_return)
                stage_episode_log[current_stage] += 1
                if flag_get:
                    stage_clear_log[current_stage] += 1

                # 새 episode: 현재 curriculum pool에서 stage 랜덤 선택
                pool = get_stage_pool(stage_ids, global_step, phase_steps)
                current_stage = choose_stage(pool, rng)
                obs, _ = env.reset_to_stage(current_stage)

                ep_return = 0.0
                ep_len = 0
                prev_x = 0
                stuck_count = 0

        # bootstrap value
        with torch.no_grad():
            ot = torch.as_tensor(obs, device=device).unsqueeze(0)
            _, last_value = agent.net(ot)
            last_value = float(last_value.item())

        # GAE
        advantages = np.zeros(args.rollout, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(args.rollout)):
            next_nonterminal = 1.0 - b_dones[t]
            next_value = last_value if t == args.rollout - 1 else b_values[t + 1]

            delta = (
                b_rewards[t]
                + args.gamma * next_value * next_nonterminal
                - b_values[t]
            )

            last_gae = (
                delta
                + args.gamma
                * args.gae_lambda
                * next_nonterminal
                * last_gae
            )

            advantages[t] = last_gae

        returns = advantages + b_values

        # tensor conversion
        t_obs = torch.as_tensor(b_obs, device=device)
        t_actions = torch.as_tensor(b_actions, device=device)
        t_logprobs = torch.as_tensor(b_logprobs, device=device)
        t_adv = torch.as_tensor(advantages, device=device)
        t_returns = torch.as_tensor(returns, device=device)

        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        # PPO update
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
                pg2 = -t_adv[mb] * torch.clamp(
                    ratio,
                    1.0 - args.clip,
                    1.0 + args.clip,
                )

                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * (
                    (values.squeeze(-1) - t_returns[mb]) ** 2
                ).mean()

                loss = (
                    policy_loss
                    + args.vf_coef * value_loss
                    - args.ent_coef * entropy
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.net.parameters(),
                    args.max_grad_norm,
                )
                optimizer.step()

        # log
        if update % 5 == 0 or update == n_updates:
            elapsed = time.time() - start_time
            sps = int(global_step / max(1e-6, elapsed))

            recent_returns = returns_log[-20:] if returns_log else [0.0]
            recent_clears = clears_log[-20:] if clears_log else [0]

            pool = get_stage_pool(stage_ids, global_step, phase_steps)

            print(
                f"upd {update:4d}/{n_updates} "
                f"step {global_step:8d} "
                f"pool={len(pool)}/{len(stage_ids)} "
                f"avg_ret={np.mean(recent_returns):8.2f} "
                f"recent_clear={sum(recent_clears):2d}/{len(recent_clears):2d} "
                f"sps={sps}"
            )

            for sid in stage_ids:
                eps = stage_episode_log[sid]
                clears = stage_clear_log[sid]
                bx = best_x_by_stage[sid]

                if eps > 0:
                    cr = clears / eps
                    recent_stage_ret = np.mean(stage_return_log[sid][-10:])
                    print(
                        f"  {sid}: episodes={eps:4d} "
                        f"clears={clears:3d} "
                        f"clear_rate={cr:5.2%} "
                        f"best_x={bx:4d} "
                        f"avg_ret10={recent_stage_ret:8.2f}"
                    )
                else:
                    print(f"  {sid}: episodes=0")

        # checkpoint
        if args.save_every > 0 and global_step % args.save_every < args.rollout:
            ckpt_path = args.out.replace(".pt", f"_step{global_step}.pt")
            agent.save(ckpt_path)
            print(f"[PPO] checkpoint saved -> {ckpt_path}")

    agent.save(args.out)
    env.close()

    print(f"[PPO] final model saved -> {args.out}")


if __name__ == "__main__":
    main()