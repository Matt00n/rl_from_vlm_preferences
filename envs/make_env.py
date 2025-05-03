from typing import Any, Dict, List, Optional, Union

import gymnasium as gym

from envs.atari_wrappers import ( 
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from envs.batched_env import SequencedBatchedEnv, ParallelBatchedEnv
from envs.evaluate import RecordScores, Evaluator
from envs.normalize import VecNormalize
from envs.pixel_observation import PixelObservationWrapper
from envs.wrappers import RoboticsDictToFlatObsWrapper


def is_gym_robotics(env_id: Union[str, Any]):
    if not isinstance(env_id, str):
        return False
    out = env_id in [
        'FrankaKitchen-v1',
    ]
    return out



def make_env(
    env_id: Union[str, Any],
    num_envs: int,
    parallel: bool = True,
    clip_actions: bool = False,
    norm_obs: bool = False,
    norm_reward: bool = False,
    clip_obs: float = 10.,
    clip_rewards: float = 10.,
    gamma: float = 0.99,
    epsilon: float = 1e-8,
    norm_obs_keys: Optional[List[str]] = None,
    evaluate: bool = False,
    is_atari: bool = False,
    add_pixel_observation_to_info: bool = False,
    capture_video: bool = False,
    video_path: str = './videos',
    env_kwargs: Dict[str, Any] = {},
) -> Union[SequencedBatchedEnv, ParallelBatchedEnv]:
    
    # TODO seed env
    # TODO handle atari
    def f():
        if is_atari:
            env_kwargs["repeat_action_probability"] = 0.
            env_kwargs["full_action_space"] = False
            env_kwargs["frameskip"] = 1
        if capture_video or add_pixel_observation_to_info:
            env_kwargs['render_mode'] = 'rgb_array'
        if isinstance(env_id, str):
            env = gym.make(env_id, **env_kwargs) # TODO env kwards
        else:
            env = env_id
        if capture_video:
            env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda x: x%10==1)
        if is_gym_robotics(env_id):
            env = RoboticsDictToFlatObsWrapper(env)
        if add_pixel_observation_to_info:
            env = PixelObservationWrapper(env)
        if evaluate:
            env = RecordScores(env)
            # env = gym.wrappers.RecordEpisodeStatistics(env, 10)
        if is_atari:
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            if not evaluate:
                env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            if not evaluate:
                env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)

        if clip_actions:
            env = gym.wrappers.ClipAction(env)
        return env

    if parallel:
        envs = ParallelBatchedEnv([f for _ in range(num_envs)])
    else: 
        envs = SequencedBatchedEnv([f for _ in range(num_envs)])
    

    if norm_obs or norm_reward:
        envs = VecNormalize(
            venv=envs,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            clip_obs=clip_obs,
            clip_reward=clip_rewards,
            gamma=gamma,
            epsilon=epsilon,
            norm_obs_keys=norm_obs_keys,
        )

    if evaluate:
        envs = Evaluator(envs)

    return envs
        