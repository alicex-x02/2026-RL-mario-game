from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mario_rl.config import ALL_STAGE_IDS, EvalConfig
from mario_rl.env import make_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Mario RL submission locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", default="student_template/agent.py")
    parser.add_argument("--model", default="student_template/model.pt")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=["SuperMarioBros-1-1-v0"],
        help="Stage ids to evaluate, or 'all' for all 32 final stages.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-stage", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_agent_class(agent_path: Path):
    if not agent_path.exists():
        raise FileNotFoundError(f"agent file not found: {agent_path}")

    root = str(Path.cwd())
    agent_dir = str(agent_path.parent.resolve())
    for path in (agent_dir, root):
        if path not in sys.path:
            sys.path.insert(0, path)

    spec = importlib.util.spec_from_file_location("submission_agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Agent"):
        raise AttributeError(f"{agent_path} does not define Agent")
    return module.Agent


def normalize_stages(raw_stages: list[str]) -> list[str]:
    if len(raw_stages) == 1 and raw_stages[0].lower() == "all":
        return list(ALL_STAGE_IDS)
    return raw_stages


def evaluate_stage(Agent, model_path: Path, env_id: str, seed: int, max_steps: int, device: str):
    env = make_env(EvalConfig(env_id=env_id, max_steps_per_episode=max_steps))
    try:
        agent = Agent.load(str(model_path), env.observation_space, env.action_space, device)
        if hasattr(agent, "reset"):
            agent.reset()

        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        total_reward = 0.0
        max_x = 0
        cleared = False
        started = time.time()

        while not done and steps < max_steps:
            action = int(agent.act(obs))
            if not 0 <= action < env.action_space.n:
                raise ValueError(
                    f"invalid action {action}; expected 0..{env.action_space.n - 1}"
                )

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total_reward += float(reward)
            max_x = max(max_x, int(info.get("x_pos", 0)))
            cleared = cleared or bool(info.get("flag_get", False))
            cleared = cleared or info.get("status") == "flag_get"

        return {
            "stage": env_id,
            "seed": seed,
            "cleared": bool(cleared),
            "max_x": int(max_x),
            "steps": int(steps),
            "reward": float(total_reward),
            "seconds": round(time.time() - started, 3),
        }
    finally:
        env.close()


def main():
    args = parse_args()
    agent_path = Path(args.agent)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    stages = normalize_stages(args.stages)
    Agent = load_agent_class(agent_path)
    device = resolve_device(args.device)

    results = []
    for stage_index, stage in enumerate(stages):
        for episode in range(args.episodes_per_stage):
            seed = args.seed + stage_index * 100 + episode
            result = evaluate_stage(Agent, model_path, stage, seed, args.max_steps, device)
            results.append(result)
            flag = "CLEAR" if result["cleared"] else "MISS"
            print(
                f"[{flag}] {stage} seed={seed} "
                f"x={result['max_x']} steps={result['steps']} "
                f"reward={result['reward']:.1f} time={result['seconds']:.1f}s",
                flush=True,
            )

    clear_count = sum(1 for result in results if result["cleared"])
    mean_progress = float(np.mean([result["max_x"] for result in results])) if results else 0.0
    score = clear_count * 100000.0 + mean_progress
    summary = {
        "agent": str(agent_path),
        "model": str(model_path),
        "device": device,
        "episodes": len(results),
        "clear_count": int(clear_count),
        "mean_progress": mean_progress,
        "score": score,
        "results": results,
    }

    print(
        "\n"
        f"summary: clear={clear_count}/{len(results)} "
        f"mean_progress={mean_progress:.1f} score={score:.1f}"
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
