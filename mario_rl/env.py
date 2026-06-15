from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .actions import MARIO_MOVEMENT, describe_actions
from .config import DEFAULT_EVAL_CONFIG, EvalConfig


class MaxAndSkipWrapper(gym.Wrapper):

    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = max(1, int(skip))

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}

        frames = deque(maxlen=2)

        obs = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            frames.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        if len(frames) == 2:
            obs = np.maximum(frames[0], frames[1])
        return obs, float(total_reward), terminated, truncated, info


class GrayResizeWrapper(gym.ObservationWrapper):

    def __init__(self, env, out_hw=(84, 84), grayscale: bool = True):
        super().__init__(env)
        self.out_h, self.out_w = out_hw
        self.grayscale = bool(grayscale)
        shape = (
            (self.out_h, self.out_w) if self.grayscale else (self.out_h, self.out_w, 3)
        )
        self.observation_space = spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observation(self, obs):
        obs = np.asarray(obs)
        if obs.ndim == 2:
            obs = obs[..., None]
        if self.grayscale and obs.shape[-1] == 3:

            obs = (
                0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]
            ).astype(np.uint8)

        else:
            obs = obs.astype(np.uint8)

        out = _resize_nn(obs, self.out_h, self.out_w)

        return out.astype(np.uint8)


def _resize_nn(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:

    if img.ndim == 2:
        h, w = img.shape
        yi = np.linspace(0, h - 1, out_h).astype(np.int64)

        xi = np.linspace(0, w - 1, out_w).astype(np.int64)

        return img[yi][:, xi]

    h, w, _ = img.shape
    yi = np.linspace(0, h - 1, out_h).astype(np.int64)
    xi = np.linspace(0, w - 1, out_w).astype(np.int64)
    return img[yi][:, xi, :]


class FrameStackWrapper(gym.Wrapper):

    def __init__(self, env, k: int = 4):
        super().__init__(env)

        self.k = int(k)

        self.frames = deque(maxlen=self.k)
        base = env.observation_space
        if len(base.shape) != 2:
            raise ValueError("FrameStackWrapper expects grayscale (H, W) observations.")
        h, w = base.shape
        self.observation_space = spaces.Box(
            0, 255, shape=(self.k, h, w), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.frames.append(obs)
        return self._get(), reward, terminated, truncated, info

    def _get(self):
        return np.stack(self.frames, axis=0).astype(np.uint8)


class _LegacyMarioToGymnasium(gym.Env):

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, legacy_env):
        self.env = legacy_env
        self.action_space = spaces.Discrete(len(MARIO_MOVEMENT))
        try:
            shp = legacy_env.observation_space.shape
        except Exception:
            shp = (240, 256, 3)
        self.observation_space = spaces.Box(0, 255, shape=shp, dtype=np.uint8)
        self.render_mode = "rgb_array"
        self._last_rgb = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self.env.seed(int(seed))
            except Exception:
                pass
            try:
                self.env.action_space.seed(int(seed))

            except Exception:
                pass
        out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out

        self._last_rgb = np.asarray(obs, dtype=np.uint8)
        return self._last_rgb, {}

    def step(self, action):
        out = self.env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        self._last_rgb = np.asarray(obs, dtype=np.uint8)
        return self._last_rgb, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self._last_rgb is not None:
            return self._last_rgb.copy()
        try:
            return np.asarray(self.env.render(mode="rgb_array"), dtype=np.uint8)
        except Exception:
            return None

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


class _IntRAM(np.ndarray):

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        if isinstance(out, np.generic):
            return int(out)
        return out


def _patch_nes_py_numpy2():

    try:
        for _name, _alias in [
            ("bool8", np.bool_),
            ("float_", np.float64),
            ("complex_", np.complex128),
            ("int0", np.intp),
            ("uint0", np.uintp),
            ("object0", np.object_),
            ("str0", np.str_),
            ("bytes0", np.bytes_),
        ]:
            if not hasattr(np, _name):
                setattr(np, _name, _alias)

    except Exception:
        pass

    try:
        import nes_py._rom as _rom_mod

        ROM = _rom_mod.ROM
        if not getattr(ROM, "_numpy2_patched", False):

            ROM.prg_rom_size = property(lambda self: 16 * int(self.header[4]))
            ROM.chr_rom_size = property(lambda self: 8 * int(self.header[5]))
            ROM._numpy2_patched = True

        import nes_py.nes_env as _ne

        NESEv = _ne.NESEnv
        if not getattr(NESEv, "_numpy2_ram_patched", False):
            _orig_ram_buffer = NESEv._ram_buffer

            def _ram_buffer_int(self):
                return _orig_ram_buffer(self).view(_IntRAM)

            NESEv._ram_buffer = _ram_buffer_int
            NESEv._numpy2_ram_patched = True
    except Exception:
        pass


def _make_super_mario_bros(
    env_id: str,
):
    _patch_nes_py_numpy2()

    import gym_super_mario_bros
    from nes_py.wrappers import (
        JoypadSpace,
    )

    try:
        base = gym_super_mario_bros.make(
            env_id,
            apply_api_compatibility=False,
            disable_env_checker=True,
        ).unwrapped
    except TypeError:
        base = gym_super_mario_bros.make(env_id).unwrapped
    legacy = JoypadSpace(base, MARIO_MOVEMENT)
    return _LegacyMarioToGymnasium(legacy)


def make_env(config: Optional[EvalConfig] = None, render: bool = False):

    config = config or DEFAULT_EVAL_CONFIG

    # 허용: 개별 스테이지(SuperMarioBros-1-1-v0 ...), 전체(SuperMarioBros-v0),
    #       랜덤 스테이지(SuperMarioBrosRandomStages-v0) 등 모든 마리오 환경.
    if not config.env_id.startswith("SuperMarioBros"):
        raise ValueError(
            f"Only real gym-super-mario-bros envs are supported, got {config.env_id!r}."
        )

    env = _make_super_mario_bros(config.env_id)
    env = MaxAndSkipWrapper(env, skip=config.frame_skip)
    env = GrayResizeWrapper(env, out_hw=config.resize, grayscale=config.grayscale)
    env = FrameStackWrapper(env, k=config.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_steps_per_episode)
    return env


def describe_spaces(config: Optional[EvalConfig] = None):

    env = make_env(config)
    try:
        return {
            "env_id": (config or DEFAULT_EVAL_CONFIG).env_id,
            "observation_shape": tuple(env.observation_space.shape),
            "observation_dtype": str(env.observation_space.dtype),
            "n_actions": int(env.action_space.n),
            "actions": describe_actions(),
        }
    finally:
        env.close()
