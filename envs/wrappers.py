from typing import Any, Dict, SupportsFloat, Tuple

import numpy as np
import gymnasium as gym


class RoboticsDictToFlatObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = self.observation_space['observation']

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs['observation']
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs = obs['observation']
        info = {}
        return obs, info