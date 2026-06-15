"""
[학습 예시 1] 경량 PPO (from scratch, gymnasium-native)

- 외부 RL 라이브러리 없이 PyTorch 만으로 구현한 최소 PPO.
- agent.py 의 NatureCNN 정책을 그대로 학습하고, Agent.save 규격으로 model.pt 저장.
- `Agent.load("model.pt", ...)` 로 그대로 불러올 수 있음.

실행 예:
    # 기본값은 매 에피소드 무작위 스테이지
    python train.py --total-steps 1000000

학생은 이 파일을 자유롭게 수정/대체해도 됨. (제출 시 학습 코드도 함께 제출)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mario_rl.env import make_env
from mario_rl.config import DEFAULT_TRAIN_ENV_ID, EvalConfig
from agent import Agent


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--env",
        default=DEFAULT_TRAIN_ENV_ID,
        help="학습 환경 ID. 기본값은 매 에피소드 무작위 스테이지.",
    )
    p.add_argument("--total-steps", type=int, default=30000)
    p.add_argument("--rollout", type=int, default=512, help="업데이트당 수집 스텝")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="model.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = EvalConfig(env_id=args.env)
    env = make_env(cfg)
    obs_space, act_space = env.observation_space, env.action_space
    device = args.device
    print(
        f"[PPO] env={args.env} obs={obs_space.shape} n_act={act_space.n} device={device}"
    )

    agent = Agent(obs_space, act_space, device)
    agent.net.train()
    optimizer = optim.Adam(agent.net.parameters(), lr=args.lr, eps=1e-5)

    obs, _ = env.reset(seed=args.seed)
    global_step = 0
    ep_return, ep_len = 0.0, 0
    returns_log = []
    best_x = 0
    start = time.time()

    n_updates = args.total_steps // args.rollout
    for update in range(1, n_updates + 1):
        # 롤아웃 버퍼
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
            b_rewards[t] = reward
            b_dones[t] = float(done)
            ep_return += reward
            ep_len += 1
            best_x = max(best_x, int(info.get("x_pos", 0)))

            if done:
                returns_log.append(ep_return)
                obs, _ = env.reset()
                ep_return, ep_len = 0.0, 0

        # 부트스트랩 value
        with torch.no_grad():
            ot = torch.as_tensor(obs, device=device).unsqueeze(0)
            _, last_value = agent.net(ot)
            last_value = float(last_value.item())

        # GAE 계산
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

        # 텐서 변환
        t_obs = torch.as_tensor(b_obs, device=device)
        t_actions = torch.as_tensor(b_actions, device=device)
        t_logprobs = torch.as_tensor(b_logprobs, device=device)
        t_adv = torch.as_tensor(advantages, device=device)
        t_returns = torch.as_tensor(returns, device=device)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

        # PPO 업데이트
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
            sps = int(global_step / (time.time() - start))
            print(
                f"upd {update}/{n_updates} step {global_step} "
                f"avg_return {np.mean(recent):7.2f} best_x {best_x} sps {sps}"
            )

    agent.save(args.out)
    env.close()
    print(f"[PPO] 저장 완료 -> {args.out}")


if __name__ == "__main__":
    main()
