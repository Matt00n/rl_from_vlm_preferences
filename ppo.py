from absl import logging, app
import base64
from functools import partial
from io import BytesIO
import os
import pickle
from PIL import Image
import random
import time
from typing import Any, Callable, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple, Union


# os.environ[
#     "XLA_PYTHON_CLIENT_MEM_FRACTION"
# ] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from openai import OpenAI, OpenAIError
import optax
from transformers import AutoProcessor, FlaxCLIPModel

from envs import make_env, Transition, has_discrete_action_space, is_atari_env
from networks.policy import Policy
from networks.networks import (FeedForwardNetwork, ActivationFn, make_policy_network, make_value_network, make_atari_feature_extractor,
                               make_reward_network)
from networks.distributions import NormalTanhDistribution, ParametricDistribution, PolicyNormalDistribution, DiscreteDistribution
import replay_buffers
import running_statistics



class Config:
    # experiment
    experiment_name = 'tests'
    api_key = os.environ['OPENAI_API_KEY']
    seed = 30
    platform = 'cpu' # CPU or GPU
    capture_video = True 
    write_logs_to_file = True
    save_model = False

    # environment
    env_id = 'AdroitHandDoor-v1' # 'FrankaKitchen-v1' 'Pendulum-v1' 'Humanoid-v4' 'HalfCheetah-v4' Hopper-v4 Ant-v4 'AdroitHandDoor-v1'
    episode_length = 1000 # TODO get this programmatically 
    env_kwargs =  {'terminate_when_unhealthy': False} # {'tasks_to_complete': ['microwave'], 'terminate_on_tasks_completed': False} {'terminate_when_unhealthy': False}
    task_description = ("Here are frames from two videos of an AI controlling a virtual robotic leg. " + 
                        "Look at the frames and select the video in which the following task is fulfilled " + 
                        "better or closer to being fulfilled. Task: The robot should move to the right " + 
                        "as fast as possible. Moving left is worse than not moving at all. " + 
                        "Falling over is wores then not falling over. " + 
                        "Answer only with 1 or 2."
                    )
    
    num_envs = 8
    parallel_envs = True 
    clip_actions = False
    normalize_observations = True 
    normalize_rewards = False
    clip_observations = 10.
    clip_rewards = 10.
    eval_env = True
    num_eval_episodes = 5
    eval_every = 1
    deterministic_eval = True

    # method / benchmark
    preference_source: str = 'vlm' # synthetic, human, vlm NOTE: synthetic preferences use oracle reward
    direct_vlm_score: bool = False
    reward_from_CLIP: bool = False

    # preference modelling
    query_schedule: Union[str, Callable[[float], float]] = "decay" # constant hyperbolic decay
    log_comparisons: bool = True
    total_comparisons: int = 750 #  5000
    initial_comparison_frac: float = 0.01 # 0.05
    rm_pretraining_epochs: int = 10
    num_sampled_trajectories = 16 #  32
    segment_step_size: int = 2 # spacing of timesteps in segment
    segment_length: int = 15 # number of timesteps per segment
    reward_hidden_layer_sizes = (256,) * 5 
    rm_return_threshold: float = 50.
    rm_return_discount: float = 1.
    preference_noise: float = 0. # 0.2
    rm_l2_coef: float = 1e-4
    normalize_reward_model: bool = True

    # preference dataset
    max_replay_size: int = 1000
    replay_buffer_batch_size: int = 256
    num_reward_minibatches: int = 4
    rm_update_epochs: int = 1

    # reward model optimizer
    rm_learning_rate = 3e-4 
    rm_anneal_lr = True
    rm_max_grad_norm = 0.5

    # algorithm hyperparameters
    total_timesteps = int(1e6) * 2 # * 8
    learning_rate = 3e-4 
    unroll_length = 2048 
    anneal_lr = True
    gamma = 0.99 
    gae_lambda = 0.95
    batch_size = 1 # number of unrolls per minibatch 
    num_minibatches = 8
    update_epochs = 10 
    normalize_advantages = True
    clip_eps = 0.2
    entropy_cost = 0.01 
    vf_cost = 0.5
    max_grad_norm = 0.5
    target_kl = None
    reward_scaling = 1. 
    
    # policy params
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4 
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5 
    activation: ActivationFn = nn.swish 
    squash_distribution: bool = True

    # atari params
    atari_dense_layer_sizes: Sequence[int] = (512,)


Metrics = Mapping[str, jnp.ndarray]

_PMAP_AXIS_NAME = 'i'

class PreferenceData(NamedTuple):
    """Container for a preferences."""
    segment_1: Transition
    segment_2: Transition
    preference: jnp.ndarray


def oric(x: jnp.ndarray) -> jnp.ndarray:
    """Optimal rounding under integer constraints.

    Given a vector of real numbers such that the sum is an integer, returns a vector
    of rounded integers that preserves the sum and which minimizes the Lp-norm of the
    difference between the rounded and original vectors for all p >= 1. Algorithm from
    https://arxiv.org/abs/1501.00014. Runs in O(n log n) time.

    Args:
        x: A 1D vector of real numbers that sum to an integer.

    Returns:
        A 1D vector of rounded integers, preserving the sum.
    """
    rounded = jnp.floor(x)
    shortfall = x - rounded

    # The total shortfall should be *exactly* an integer, but we
    # round to account for numerical error.
    total_shortfall = jnp.round(shortfall.sum()).astype(int)
    indices = jnp.argsort(-shortfall)

    # Apportion the total shortfall to the elements in order of
    # decreasing shortfall.
    rounded = rounded.at[indices[:total_shortfall]].add(1)
    return rounded.astype(int)

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # in order to avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


@flax.struct.dataclass
class PPONetworkParams:
    """Contains training state for the learner."""
    policy: Any
    value: Any


@flax.struct.dataclass
class PPONetworks:
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class AtariPPONetworkParams:
    """Contains training state for the learner."""
    feature_extractor: Any
    policy: Any
    value: Any


@flax.struct.dataclass
class AtariPPONetworks:
    feature_extractor: FeedForwardNetwork
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: Union[ParametricDistribution, DiscreteDistribution]


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Union[PPONetworkParams, AtariPPONetworkParams]
    env_steps: jnp.ndarray


@flax.struct.dataclass
class RewardNetworkParams:
    """Contains training state for the learner."""
    feature_extractor: Any
    reward_head: Any


@flax.struct.dataclass
class RewardModelTrainingState:
    """Contains training state for the reward model."""
    optimizer_state: optax.OptState
    params: RewardNetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    training_steps: jnp.ndarray


def make_reward_model(reward_network: FeedForwardNetwork):
    """Creates params and inference function for the reward model."""

    def make_reward_fn(params: Any) -> Policy:

        normalizer_params, reward_head_params = params
        
        @jax.jit
        def reward_fn(observations: jnp.ndarray,
                      actions: jnp.ndarray,
                      next_observations: jnp.ndarray
                      ) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
            concatenated_inputs = jnp.concatenate((observations, actions, next_observations), axis=-1)
            rewards = reward_network.apply(normalizer_params, reward_head_params, concatenated_inputs)
            return rewards

        return reward_fn

    return make_reward_fn


def make_inference_fn(ppo_networks: Union[PPONetworks, AtariPPONetworks]):
    """Creates params and inference function for the PPO agent."""

    def make_policy(params: Any,
                    deterministic: bool = False) -> Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        @jax.jit
        def policy(observations: jnp.ndarray,
                key_sample: jnp.ndarray) -> Tuple[jnp.ndarray, Mapping[str, Any]]:
            logits = policy_network.apply(params, observations)
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample)
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions)
            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions
            }

        return policy

    return make_policy


def make_feature_extraction_fn(ppo_networks: AtariPPONetworks):
    """Creates feature extractor for inference."""

    def make_feature_extractor(params: Any):
        shared_feature_extractor = ppo_networks.feature_extractor

        @jax.jit
        def feature_extractor(observations: jnp.ndarray) -> jnp.ndarray:
            return shared_feature_extractor.apply(params, observations)

        return feature_extractor

    return make_feature_extractor


def make_ppo_networks(
        observation_size: int,
        action_size: int,
        policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
        value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
        activation: ActivationFn = nn.swish,
        sqash_distribution: bool = True,
        discrete_policy: bool = False,
        shared_feature_extractor: bool = False,
        feature_extractor_dense_hidden_layer_sizes: Optional[Sequence[int]] = (512,),
    ) -> PPONetworks:
    """Make PPO networks with preprocessor."""
    if discrete_policy:
        parametric_action_distribution = DiscreteDistribution(
            param_size=action_size)
    elif sqash_distribution:
        parametric_action_distribution = NormalTanhDistribution(
            event_size=action_size)
    else:
        parametric_action_distribution = PolicyNormalDistribution(
            event_size=action_size)
    if shared_feature_extractor:
        feature_extractor = make_atari_feature_extractor(
            obs_size=observation_size,
            hidden_layer_sizes=feature_extractor_dense_hidden_layer_sizes,
            activation=nn.relu
        )
        policy_network = make_policy_network(
            parametric_action_distribution.param_size,
            feature_extractor_dense_hidden_layer_sizes[-1],
            hidden_layer_sizes=(),
            activation=activation)
        value_network = make_value_network(
            feature_extractor_dense_hidden_layer_sizes[-1],
            hidden_layer_sizes=(),
            activation=activation)
        return AtariPPONetworks(
            feature_extractor=feature_extractor,
            policy_network=policy_network,
            value_network=value_network,
            parametric_action_distribution=parametric_action_distribution)
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation)
    value_network = make_value_network(
        observation_size,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation)

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution)


vtake = jax.vmap(partial(jnp.take, axis=0)) 


def sample_fragment_pairs(
    trajectories: Transition,
    num_trajectories: int,
    num_sampled_pairs: int,
    segment_length: int,
    episode_length: int,
    key: jnp.ndarray,
    segment_step_size: int = 1,
) -> Transition:
    """Samples trajectory segment pairs from given trajectories.

    Args:
        trajectories: Transitions with leading dimension [B, T].
        num_trajectories: Number of given trajectories.
        num_sampled_pairs: Number of segment pairs to sample.
        segment_length: Number of transitions each segment consists of.
        episode_length: Fixed length of the episodes.
        key: Random key.
        segment_step_size: Number of steps between each transition. 

    Returns:
        Transitions with leading dimensions [S, 2, t], where S is the
        number of sampled segment pairs and t is the segment length.
    """
    trajectory_key, segment_key = jax.random.split(key)

    # sample trajectories with replacement
    idx = jax.random.randint(trajectory_key, (2*num_sampled_pairs, 1), 0, num_trajectories)
    # data = jax.tree_util.tree_map(lambda x: jnp.take(x, idx, axis=0), trajectories) # [S*2, T, ...]
    # chex.assert_shape(data.reward, (2*num_sampled_pairs, episode_length))

    # sample starting points within these trajectories
    traj_offset_end = episode_length - (segment_length - 1) * segment_step_size - 1
    traj_idx = jax.random.randint(segment_key, (2*num_sampled_pairs,), 0, traj_offset_end)

    # compute further points in those trajectories based on the step size and segment length
    end_idx = traj_idx + (segment_length - 1) * segment_step_size
    traj_idx = jnp.linspace(traj_idx, end_idx, segment_length, axis=-1).astype(int)

    chex.assert_shape(traj_idx, (2*num_sampled_pairs, segment_length))

    # retrieve segments and pair them
    # data = jax.tree_util.tree_map(lambda x: vtake(x, traj_idx), data)
    data = jax.tree_util.tree_map(lambda x: x[idx, traj_idx], trajectories)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1, 2,) + x.shape[1:]),
                                    data)
    chex.assert_shape(data.reward, (num_sampled_pairs, 2, segment_length))
    return data


def get_synthetic_preferences(segment_pairs: Transition) -> jnp.ndarray:
    """Preferences over batches of trajectory pairs as given by higher return.
    Preferences are indicated as the probability of the first segment being 
    preferred.

    Args:
        segment_pairs: Batch of trajectory pairs, shape [B, 2, T, ...].
    
    Returns:
        An array of preferences, shape [B]
    """   
    # oracle
    r_1 = jnp.sum(jnp.take(segment_pairs.reward, 0, axis=1), axis=-1)
    r_2 = jnp.sum(jnp.take(segment_pairs.reward, 1, axis=1), axis=-1)
    preferences = (r_1 > r_2).astype(float)
    return preferences


def img_to_bytes(img: Image.Image):
    """Converts a PIL Image object to a byte string."""
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img


def get_vlm_preferences(segment_pairs: Transition,
                        client: Any,
                        task_description: str,
                        use_soft_labels: bool = True,
                        log_path: str = None) -> jnp.ndarray:
    """Preferences over batches of trajectory pairs as given by a VLM.
    Pixels are contained in segment_pairs.extras['state_extras']['pixels'].
    Preferences are indicated as the probability of the first segment being 
    preferred.

    Args:
        segment_pairs: Batch of trajectory pairs, shape [B, 2, T, ...].
        use_soft_labels: whether to output probabilities or a binary preference.
    
    Returns:
        An array of preferences, shape [B]
    """   
    chex.assert_shape(segment_pairs.extras['state_extras']['pixels'], 
                      (None, 2, Config.segment_length, None, None, 3))

    preferences = []
    
    for pair in segment_pairs.extras['state_extras']['pixels']:
        chex.assert_shape(pair, (2, Config.segment_length, None, None, 3))
        video_a, video_b = np.asarray(pair)
        video_a = list(video_a)
        video_b = list(video_b)
        assert len(video_b) == Config.segment_length

        video_a = list(map(lambda x: Image.fromarray(x), video_a))
        video_b = list(map(lambda x: Image.fromarray(x), video_b))

        if log_path:
            comp_path = os.path.join(log_path, f'comparison_{int(time.time())}') 
            comp_path_a = os.path.join(comp_path, 'video_a') 
            comp_path_b = os.path.join(comp_path, 'video_b') 
            os.makedirs(comp_path_a)
            os.makedirs(comp_path_b)
            for i, img in enumerate(video_a):
                img.save(comp_path_a + f'/{i}.jpeg', format="JPEG")
            for i, img in enumerate(video_b):
                img.save(comp_path_b + f'/{i}.jpeg', format="JPEG")

        video_a = list(map(img_to_bytes, video_a))
        video_b = list(map(img_to_bytes, video_b))

        prompt_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task_description},
                    {"type": "text", "text": "Video 1:"},
                    *map(lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low"
                            },
                        }, video_a),
                    {"type": "text", "text": "Video 2:"},
                    *map(lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{x}",
                                "detail": "low"
                            },
                        }, video_b),
                ],
            },
        ]
        params = {
            "model": "gpt-4o",
            "logprobs": True,
            "top_logprobs": 2,
            "messages": prompt_messages,
            "max_tokens": 10,
        }

        api_error = False
        try:
            result = client.chat.completions.create(**params)
            result = result.choices[0]
            message = result.message.content
        except OpenAIError as e:
            print(e)
            message = 'API error: forced tie'
            api_error = True

        if log_path:
            with open(comp_path + '/vlm_preference.txt', 'w') as f:
                f.write(message)

        if api_error:
            result = 0.5
        elif use_soft_labels:
            result = result.logprobs.content[0].top_logprobs
            probs = {'1': -float('inf'), '2': -float('inf')}
            for token in result:
                if token.token in probs:
                    probs[token.token] = token.logprob
            probs = jnp.array([probs['1'], probs['2']])
            probs = jnp.exp(probs) / jnp.sum(jnp.exp(probs))
            result = probs[0]
        else:
            if result.startswith('1'):
                result = 1.
            elif result.startswith('2'):
                result = 0.
            elif result.startswith('0'):
                result = 0.5
            else: 
                print(result)
                result = 0.5

        preferences.append(result)

    preferences = jnp.array(preferences, dtype=float)

    # assert 1 == 2, 'unfinished and possibly dangerous'

    return preferences


def get_human_preferences(segment_pairs: Transition,
                        log_path: str = None) -> jnp.ndarray:
    """Query human for preferences over batches of trajectory pairs.
    Pixels are contained in segment_pairs.extras['state_extras']['pixels'].
    Preferences are indicated as the probability of the first segment being 
    preferred.

    Args:
        segment_pairs: Batch of trajectory pairs, shape [B, 2, T, ...].
    
    Returns:
        An array of preferences, shape [B]
    """   
    assert log_path is not None
    chex.assert_shape(segment_pairs.extras['state_extras']['pixels'], 
                      (None, 2, Config.segment_length, None, None, 3))

    preferences = []
    
    for pair in segment_pairs.extras['state_extras']['pixels']:
        chex.assert_shape(pair, (2, Config.segment_length, None, None, 3))
        video_a, video_b = np.asarray(pair)
        video_a = list(video_a)
        video_b = list(video_b)
        assert len(video_b) == Config.segment_length

        comp_path = os.path.join(log_path, f'comparison_{int(time.time())}') 
        os.makedirs(comp_path)

        clip = ImageSequenceClip(video_a, fps=30)
        path = os.path.join(comp_path, f"video_a.mp4")
        clip.write_videofile(path, logger=None)

        clip = ImageSequenceClip(video_b, fps=30)
        path = os.path.join(comp_path, f"video_b.mp4")
        clip.write_videofile(path, logger=None)

        result = input('Enter preference: ')

        with open(comp_path + '/human_preference.txt', 'w') as f:
            f.write(result)

        if result.upper().startswith('A'):
            result = 1.
        elif result.upper().startswith('B'):
            result = 0.
        elif result.startswith('tie'):
            result = 0.5
        else: 
            print(result)
            result = 0.5

        preferences.append(result)

    preferences = jnp.array(preferences, dtype=float)

    # assert 1 == 2, 'unfinished and possibly dangerous'

    return preferences


def compute_reward_model_loss(
    params: RewardNetworkParams,
    normalizer_params: running_statistics.RunningStatisticsState,
    data: PreferenceData,
    rng: jnp.ndarray,
    reward_network: FeedForwardNetwork,
    threshold: float = 50.,
    segment_length: int = 10,
    discount_factor: float = 1.,
    noise_prob: float = 0.2,
    l2_coef: float = 1e-4,
    shared_feature_extractor: bool = False,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Computes loss for the reward model following the Terry-Bradley model.

    Args:
        params: Network parameters.
        normalizer_params: Running statistics to normalize rewards.
        data: PreferenceData that holds to segments (Transition) with leading dimension [B, T]. 
        rng: Random key
        reward_network: Reward network.
        threshold: Threshold for return differences to prevent overflows.
        discount_factor: Optional discount factor when computing returns.
        noise_prob: Factor with which preference labels are smoothed to account for 
            noisy preferences.
        l2_coef: L2 penalty coefficient.
        shared_feature_extractor: Whether networks use a shared feature extractor.

    Returns:
        A tuple (loss, metrics)
    """    
    # data dim: [B, T, ...]

    reward_model_apply = reward_network.apply

    # concatenating inputs for reward model
    segment_1 = data.segment_1
    segment_2 = data.segment_2
    conc_inp_1 = jnp.concatenate((segment_1.observation, segment_1.action, segment_1.next_observation), axis=-1)
    conc_inp_2 = jnp.concatenate((segment_2.observation, segment_2.action, segment_2.next_observation), axis=-1)

    # applying reward model
    rewards_segment_1 = reward_model_apply(normalizer_params, params.reward_head, conc_inp_1)
    rewards_segment_2 = reward_model_apply(normalizer_params, params.reward_head, conc_inp_2)

    # optional discount factor
    discounts = discount_factor ** jnp.arange(segment_length)
    discounts = jnp.expand_dims(discounts, axis=0)
    
    # summing over rewards and computing preference
    returns_diff = jnp.sum(discounts * jnp.squeeze(rewards_segment_2 - rewards_segment_1), axis=-1)

    # Clip to avoid overflows (which in particular may occur
    # in the backwards pass even if they do not in the forward pass).
    returns_diff = -jnp.clip(returns_diff, -threshold, threshold)

    # We optionally smooth preferences due to noisy labels
    preferences = data.preference # shape: [B]
    smoothed_preference = jax.lax.stop_gradient(noise_prob * 0.5 + (1 - noise_prob) * preferences)

    smoothed_preference = smoothed_preference.astype(returns_diff.dtype)
    # probability that segment 1 is preferred.
    log_p = jax.nn.log_sigmoid(returns_diff)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter more numerically stable
    log_not_p = jax.nn.log_sigmoid(-returns_diff)
    rm_loss = jnp.mean(-smoothed_preference * log_p - (1. - smoothed_preference) * log_not_p)

    # l2 penalty
    l2_penalty = l2_coef * 0.5 * sum(
        jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(params))
    
    total_loss = rm_loss + l2_penalty



    # accuracy
    accuracy = jnp.mean(1 - jnp.logical_xor(jnp.exp(log_p) > 0.5, preferences > 0.5).astype(int))
    accuracy = jax.lax.stop_gradient(accuracy)

    metrics = {
        'total_rm_loss': total_loss,
        'reward_model_loss': rm_loss,
        'rm_l2_penalty': l2_penalty,
        'accuracy': accuracy,
    }

    return total_loss, metrics

def compute_reward_model_loss_from_score(
    params: RewardNetworkParams,
    normalizer_params: running_statistics.RunningStatisticsState,
    data: PreferenceData,
    rng: jnp.ndarray,
    reward_network: FeedForwardNetwork,
    threshold: float = 50.,
    segment_length: int = 10,
    discount_factor: float = 1.,
    l2_coef: float = 1e-4,
    shared_feature_extractor: bool = False,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Computes loss for the reward model by regressing on a given reward scores.

    Args:
        params: Network parameters.
        normalizer_params: Running statistics to normalize rewards.
        data: PreferenceData that holds to segments (Transition) with leading dimension [B, T]. 
        rng: Random key
        reward_network: Reward network.
        threshold: Threshold for return differences to prevent overflows.
        discount_factor: Optional discount factor when computing returns.
        l2_coef: L2 penalty coefficient.
        shared_feature_extractor: Whether networks use a shared feature extractor.

    Returns:
        A tuple (loss, metrics)
    """    
    # data dim: [B, T, ...]

    reward_model_apply = reward_network.apply

    # concatenating inputs and applying reward model
    segment_1 = data.segment_1
    conc_inp_1 = jnp.concatenate((segment_1.observation, segment_1.action, segment_1.next_observation), axis=-1)
    rewards_segment_1 = reward_model_apply(normalizer_params, params.reward_head, conc_inp_1)

    # optional discount factor and summing over rewards
    discounts = discount_factor ** jnp.arange(segment_length)
    discounts = jnp.expand_dims(discounts, axis=0)
    returns = jnp.sum(discounts * jnp.squeeze(rewards_segment_1), axis=-1)

    # Clip to avoid overflows (which in particular may occur
    # in the backwards pass even if they do not in the forward pass).
    returns = jnp.clip(returns, -threshold, threshold)

    # MSE loss
    v_error = jax.lax.stop_gradient(data.preference) - returns
    rm_loss = jnp.mean(jnp.squeeze(v_error * v_error * 0.5))

    # l2 penalty
    l2_penalty = l2_coef * 0.5 * sum(
        jnp.sum(jnp.square(w)) for w in jax.tree_util.tree_leaves(params))
    
    total_loss = rm_loss + l2_penalty

    metrics = {
        'total_rm_loss': total_loss,
        'reward_model_loss': rm_loss,
        'rm_l2_penalty': l2_penalty,
    }

    return total_loss, metrics



def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
    """Calculates the Generalized Advantage Estimation (GAE).

    Args:
        truncation: A float32 tensor of shape [T, B] with truncation signal.
        termination: A float32 tensor of shape [T, B] with termination signal.
        rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
        values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
        bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
        lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
        discount: TD discount.

    Returns:
        A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
        A float32 tensor of shape [T, B] of advantages.
    """

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, acc = carry
        truncation_mask, delta, termination = target_t
        acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs, (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True)
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount *
                    (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)



def compute_ppo_loss(
    params: Union[PPONetworkParams, AtariPPONetworkParams],
    data: Transition,
    rng: jnp.ndarray,
    ppo_network: Union[PPONetworks, AtariPPONetworks],
    vf_cost: float = 0.5,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    shared_feature_extractor: bool = False,
) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
    """Computes PPO loss including value loss and entropy bonus.

    Policy loss: $L_\pi = \frac{1}{\lvert \mathcal{D} \rvert} \sum_{\mathcal{D}} 
    \min \biggl( \frac{\pi_\theta (a \mid s)}{\pi_\text{old} 
    (a \mid s)} \hat{A}, \text{clip}\Bigl( \frac{\pi_\theta (a \mid s)}{\pi_\text{old} 
    (a \mid s)}, 1-\varepsilon, 1+\varepsilon \Bigr) \hat{A} \biggr)$

    Args:
        params: Network parameters,
        data: Transition that with leading dimension [B, T]. extra fields required
            are ['state_extras']['truncation'] ['policy_extras']['raw_action']
            ['policy_extras']['log_prob']
        rng: Random key
        ppo_network: PPO networks.
        entropy_cost: entropy cost.
        discounting: discounting,
        reward_scaling: reward multiplier.
        gae_lambda: General advantage estimation lambda.
        clipping_epsilon: Policy loss clipping epsilon
        normalize_advantage: whether to normalize advantage estimate
        shared_feature_extractor: Whether networks use a shared feature extractor.

    Returns:
        A tuple (loss, metrics)
    """
    parametric_action_distribution = ppo_network.parametric_action_distribution
    
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply

    # Put the time dimension first.
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    hidden = data.observation
    hidden_boot = data.next_observation[-1]
    if shared_feature_extractor:
        feature_extractor_apply = ppo_network.feature_extractor.apply
        hidden = feature_extractor_apply(params.feature_extractor, data.observation)
        hidden_boot = feature_extractor_apply(params.feature_extractor, 
                                          data.next_observation[-1])
    
    policy_logits = policy_apply(params.policy,
                                hidden)

    baseline = value_apply(params.value, hidden)

    
    bootstrap_value = value_apply(params.value,
                                    hidden_boot)

    rewards = data.reward * reward_scaling
    truncation = data.extras['state_extras']['truncation']
    termination = (1 - data.discount) * (1 - truncation) 

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.extras['policy_extras']['raw_action'])
    behaviour_action_log_probs = data.extras['policy_extras']['log_prob']

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting)
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_ratio = target_action_log_probs - behaviour_action_log_probs
    rho_s = jnp.exp(log_ratio)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon,
                                1 + clipping_epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
    approx_kl = ((rho_s - 1) - log_ratio).mean()

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_cost

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    total_loss = policy_loss + v_loss + entropy_loss

    metrics = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': v_loss,
        'entropy_loss': entropy_loss,
        'entropy': entropy, 
        'approx_kl': jax.lax.stop_gradient(approx_kl), 
    }

    return total_loss, metrics




def main(_):
    run_name = f"Exp_{Config.experiment_name}__{Config.env_id}__{Config.seed}__{int(time.time())}"

    if Config.capture_video or Config.write_logs_to_file:
        run_dir = os.path.join('experiments', run_name)
        video_dir = os.path.join(run_dir, 'videos')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

    if Config.write_logs_to_file:
        from absl import flags
        flags.FLAGS.alsologtostderr = True
        log_path = os.path.join(run_dir, 'training_logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logging.get_absl_handler().use_absl_log_file('logs', log_path)

    logging.get_absl_handler().setFormatter(None)

    # jax set up devices
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    device_count = local_devices_to_use * process_count
    assert Config.num_envs % device_count == 0


    assert Config.batch_size * Config.num_minibatches % Config.num_envs == 0
    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        Config.batch_size * Config.unroll_length * Config.num_minibatches)
    num_training_steps = np.ceil(Config.total_timesteps / env_step_per_training_step).astype(int)

    # log hyperparameters
    logging.info("|param: value|")
    for key, value in vars(Config).items():
        if not key.startswith('__'):
            logging.info(f"|{key}:  {value}|")

    random.seed(Config.seed)
    np.random.seed(Config.seed)
    # handle / split random keys
    key = jax.random.PRNGKey(Config.seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_envs, rb_key, key_pixel_env, eval_key = jax.random.split(local_key, 5)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    (key_policy, key_value, key_feature_extractor,
        key_rm_feature_extractor, key_reward_model) = jax.random.split(global_key, 5)
    del global_key
 
    is_atari = is_atari_env(Config.env_id)
    envs = make_env(
        env_id=Config.env_id,
        num_envs=Config.num_envs,
        parallel=Config.parallel_envs,
        clip_actions=Config.clip_actions,
        norm_obs=Config.normalize_observations,
        norm_reward=Config.normalize_rewards,
        clip_obs=Config.clip_observations,
        clip_rewards=Config.clip_rewards,
        is_atari=is_atari,
        env_kwargs=Config.env_kwargs,
    )

    discrete_action_space = has_discrete_action_space(envs)
    envs.seed(int(key_envs[0]))
    env_state = envs.reset() 


    if discrete_action_space:
        action_size = envs.action_space.n
        dummy_action = jnp.zeros(1)
    else:
        action_size = np.prod(envs.action_space.shape) # flatten action size for nested spaces
        dummy_action = jnp.zeros((action_size,))
    if is_atari:
        observation_shape = env_state.obs.shape[-3:]
    else:
        observation_shape = env_state.obs.shape[-1]

    ppo_network = make_ppo_networks(
        observation_size=observation_shape, # NOTE only works with flattened observation space
        action_size=action_size, # flatten action size for nested spaces
        policy_hidden_layer_sizes=Config.policy_hidden_layer_sizes, 
        value_hidden_layer_sizes=Config.value_hidden_layer_sizes,
        activation=Config.activation,
        sqash_distribution=Config.squash_distribution,
        discrete_policy=discrete_action_space,
        shared_feature_extractor=is_atari,
        feature_extractor_dense_hidden_layer_sizes=Config.atari_dense_layer_sizes,
    )
    make_policy = make_inference_fn(ppo_network)
    if is_atari:
        make_feature_extractor = make_feature_extraction_fn(ppo_network)

    if is_atari:
        flattened_obs_size = Config.atari_dense_layer_sizes[-1]
    else:
        flattened_obs_size = observation_shape

    # Normalization for reward model
    normalize = lambda x, y: x
    if Config.normalize_reward_model:
        normalize = running_statistics.normalize

    reward_network = make_reward_network(
        inp_size=dummy_action.shape[-1] + 2 * flattened_obs_size,
        normalization_fn=normalize,
        hidden_layer_sizes=Config.reward_hidden_layer_sizes,
    )
    make_reward_fn = make_reward_model(reward_network)

    QUERY_SCHEDULES: Dict[str, Callable[[float], float]] = {
        "constant": lambda t: 1.0,
        "hyperbolic": lambda t: 1.0 / (1.0 + t),
        "inverse_quadratic": lambda t: 1.0 / (1.0 + t**2),
        "decay": lambda t: num_training_steps / (t + num_training_steps),
    }

    # Compute the number of comparisons to request at each iteration in advance.
    query_schedule = Config.query_schedule
    if not callable(query_schedule):
        if query_schedule in QUERY_SCHEDULES:
            query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")
        
    initial_comparisons = int(Config.total_comparisons * Config.initial_comparison_frac)
    total_comparisons = Config.total_comparisons - initial_comparisons
    vec_schedule = jnp.vectorize(query_schedule)
    unnormalized_probs = vec_schedule(jnp.linspace(0, 1, num_training_steps - 1))
    probs = unnormalized_probs / jnp.sum(unnormalized_probs)
    shares = oric(probs * total_comparisons)
    schedule = [initial_comparisons] + shares.tolist()
    print(f"Query schedule: {schedule}")

    # NOTE: jitting currently causes error due to shape depending on num_sampled_pairs arg
    jitted_sample_fragment_pairs = partial(sample_fragment_pairs, num_trajectories=Config.num_sampled_trajectories,
                                    segment_length=Config.segment_length, episode_length=Config.episode_length,
                                    segment_step_size=Config.segment_step_size)
    
    if Config.preference_source == 'synthetic':
        get_preferences = get_synthetic_preferences
    elif Config.preference_source == 'vlm':
        client = OpenAI(api_key=Config.api_key)
        if Config.log_comparisons:
            comp_log_dir = os.path.join(run_dir, 'comparisons')
            os.makedirs(comp_log_dir)
        else:
            comp_log_dir = None
        get_preferences = partial(get_vlm_preferences, client=client, task_description=Config.task_description,
                                  log_path=comp_log_dir)
    elif Config.preference_source == 'human':
        if Config.log_comparisons:
            comp_log_dir = os.path.join(run_dir, 'comparisons')
            os.makedirs(comp_log_dir)
        else:
            comp_log_dir = None
        get_preferences = partial(get_human_preferences, log_path=comp_log_dir)
    else:
        raise NotImplementedError(f'Preference source must be one of "synthetic", "human" or "vlm", value: {Config.preference_source}')
    
    # intialize replay buffer
    dummy_obs = jnp.zeros(observation_shape,)
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.,
        discount=0.,
        next_observation=dummy_obs,
        extras=0.,
        # extras={
        #     'state_extras': {
        #         'truncation': 0.
        #     },
        # },
    )
    dummy_segment = jax.tree_util.tree_map(lambda *x: np.stack(x), *[dummy_transition for _ in range(Config.segment_length)])

    dummy_preference = PreferenceData(
        segment_1=dummy_segment,
        segment_2=dummy_segment,
        preference=0.,
    )
    
    replay_buffer = replay_buffers.UniformSamplingQueue( # UniformSamplingQueue Queue PrioritizedSamplingQueue
        max_replay_size=Config.max_replay_size // device_count,
        dummy_data_sample=dummy_preference,
        sample_batch_size=Config.replay_buffer_batch_size * Config.num_reward_minibatches // device_count)
    

    # create optimizer
    if Config.anneal_lr:    
        learning_rate = optax.linear_schedule(
            Config.learning_rate, 
            Config.learning_rate * 0.01, # 0
            transition_steps=Config.total_timesteps, 
        )
    else:
        learning_rate = Config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.max_grad_norm),
        optax.adam(learning_rate),
    )

    # create loss function via functools.partial
    loss_fn = partial(
        compute_ppo_loss,
        ppo_network=ppo_network,
        vf_cost=Config.vf_cost,
        entropy_cost=Config.entropy_cost,
        discounting=Config.gamma,
        reward_scaling=Config.reward_scaling,
        gae_lambda=Config.gae_lambda,
        clipping_epsilon=Config.clip_eps,
        normalize_advantage=Config.normalize_advantages,
        shared_feature_extractor=is_atari,
    )

    # create reward model optimizer
    if Config.rm_anneal_lr:    
        learning_rate = optax.linear_schedule(
            Config.rm_learning_rate, 
            Config.rm_learning_rate * 0.01, # 0
            transition_steps=num_training_steps, 
        )
    else:
        learning_rate = Config.rm_learning_rate
    rm_optimizer = optax.chain(
        optax.clip_by_global_norm(Config.rm_max_grad_norm),
        optax.adam(learning_rate),
    )

    # create reward loss function via functools.partial
    if not Config.direct_vlm_score:
        rm_loss_fn = partial(
            compute_reward_model_loss,
            reward_network=reward_network,
            threshold=Config.rm_return_threshold,
            segment_length=Config.segment_length,
            discount_factor=Config.rm_return_discount,
            noise_prob=Config.preference_noise,
            l2_coef=Config.rm_l2_coef,
            shared_feature_extractor=is_atari,
        )
    else:
        rm_loss_fn = partial(
            compute_reward_model_loss_from_score,
            reward_network=reward_network,
            threshold=Config.rm_return_threshold,
            segment_length=Config.segment_length,
            discount_factor=Config.rm_return_discount,
            l2_coef=Config.rm_l2_coef,
            shared_feature_extractor=is_atari,
        )

    if Config.reward_from_CLIP:
        reward_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Example usage
        # image = Image.open(requests.get(url, stream=True).raw)
        # inputs = processor(
        #     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True
        # )
        # outputs = model(**inputs)
        # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score, [num images, num texts]
        raise NotImplementedError
    


    def loss_and_pgrad(loss_fn: Callable[..., float],
                        pmap_axis_name: Optional[str],
                        has_aux: bool = False):
        g = jax.value_and_grad(loss_fn, has_aux=has_aux)

        def h(*args, **kwargs):
            value, grad = g(*args, **kwargs)
            return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

        return g if pmap_axis_name is None else h
    

    def gradient_update_fn(loss_fn: Callable[..., float],
                            optimizer: optax.GradientTransformation,
                            pmap_axis_name: Optional[str],
                            has_aux: bool = False):
        """Wrapper of the loss function that apply gradient updates.

        Args:
            loss_fn: The loss function.
            optimizer: The optimizer to apply gradients.
            pmap_axis_name: If relevant, the name of the pmap axis to synchronize
            gradients.
            has_aux: Whether the loss_fn has auxiliary data.

        Returns:
            A function that takes the same argument as the loss function plus the
            optimizer state. The output of this function is the loss, the new parameter,
            and the new optimizer state.
        """
        loss_and_pgrad_fn = loss_and_pgrad(
            loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

        def f(*args, optimizer_state):
            value, grads = loss_and_pgrad_fn(*args)
            params_update, optimizer_state = optimizer.update(grads, optimizer_state)
            params = optax.apply_updates(args[0], params_update)
            return value, params, optimizer_state

        return f
    
    ppo_gradient_update_fn = gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

    # minibatch training step
    def minibatch_step(carry, data: Transition,):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = ppo_gradient_update_fn(
            params,
            data,
            key_loss,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics


    # sgd step
    def sgd_step(carry, unused_t, data: Transition):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (Config.num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            minibatch_step, 
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=Config.num_minibatches)
        return (optimizer_state, params, key), metrics
    

    # learning 
    def learn(
        data: Transition,
        training_state: TrainingState,
        key_sgd: jnp.ndarray,
    ):
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(
                sgd_step, data=data),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=Config.update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            env_steps=training_state.env_steps + env_step_per_training_step)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return new_training_state, metrics
    
    learn = jax.pmap(learn, axis_name=_PMAP_AXIS_NAME)

    # train reward model 
    reward_model_gradient_update_fn = gradient_update_fn(
        rm_loss_fn, rm_optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
        )

    # minibatch training step
    def rm_minibatch_step(carry, data: Transition, normalizer_params: running_statistics.RunningStatisticsState):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = reward_model_gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics


    # sgd step
    def rm_sgd_step(carry, unused_t, data: Transition, normalizer_params: running_statistics.RunningStatisticsState):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (Config.num_reward_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(rm_minibatch_step, normalizer_params=normalizer_params), 
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=Config.num_reward_minibatches)
        return (optimizer_state, params, key), metrics
    

    def train_reward_model(
        data: PreferenceData,
        reward_model_training_state: RewardModelTrainingState,
        buffer_state: Any,
        key_sgd: jnp.ndarray,
    ):
        
        # insert data into replay buffer
        buffer_state = replay_buffer.insert(buffer_state, data)

        # sampling from replay buffer
        buffer_state, data = replay_buffer.sample(buffer_state)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(
                rm_sgd_step, data=data, normalizer_params=reward_model_training_state.normalizer_params),
            (reward_model_training_state.optimizer_state, reward_model_training_state.params, key_sgd), (),
            length=Config.rm_update_epochs)

        new_training_state = RewardModelTrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=reward_model_training_state.normalizer_params,
            training_steps=reward_model_training_state.training_steps + Config.rm_update_epochs * Config.num_reward_minibatches)
        
        metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return new_training_state, buffer_state, metrics
    
    train_reward_model = jax.pmap(train_reward_model, axis_name=_PMAP_AXIS_NAME)


    def update_reward_normalization(rm_training_state: RewardModelTrainingState,
                                    rewards: jnp.ndarray,) -> RewardModelTrainingState:
        normalizer_params = running_statistics.update(
            rm_training_state.normalizer_params,
            rewards,
            pmap_axis_name=_PMAP_AXIS_NAME)

        return rm_training_state.replace(
            normalizer_params=normalizer_params)
    
    update_reward_normalization = jax.pmap(update_reward_normalization, axis_name=_PMAP_AXIS_NAME)


    # initialize params & training state
    if is_atari:
        init_params = AtariPPONetworkParams(
            feature_extractor=ppo_network.feature_extractor.init(key_feature_extractor),
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value))
    else:
        init_params = PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value))
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        env_steps=0)
    training_state = jax.device_put_replicated(
        training_state,
        jax.local_devices()[:local_devices_to_use])
    
    # initialize reward model
    rm_from_pixels = False # TODO: enable option for reward model from pixels
    if rm_from_pixels:
        raise NotImplementedError
        rm_init_params = RewardNetworkParams(
            feature_extractor=ppo_network.feature_extractor.init(key_rm_feature_extractor),
            reward_head=reward_network.init(key_reward_model),)
    else:
        rm_init_params = RewardNetworkParams(
            feature_extractor=jnp.zeros(1),
            reward_head=reward_network.init(key_reward_model),)
    rm_training_state = RewardModelTrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=rm_optimizer.init(rm_init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=rm_init_params,
        normalizer_params=running_statistics.init_state(jnp.zeros_like(0.)),
        training_steps=0)
    rm_training_state = jax.device_put_replicated(
        rm_training_state,
        jax.local_devices()[:local_devices_to_use])
    
    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    # TODO: temporary
    # create additionaly env with added pixel observations
    pixel_envs = make_env(
        env_id=Config.env_id,
        num_envs=Config.num_envs,
        parallel=False,
        clip_actions=Config.clip_actions,
        norm_obs=False,
        norm_reward=False,
        clip_obs=Config.clip_observations,
        clip_rewards=Config.clip_rewards,
        is_atari=is_atari,
        add_pixel_observation_to_info=True,
        env_kwargs=Config.env_kwargs,
    )
    pixel_envs.seed(int(key_pixel_env[0]))
    pixel_env_state = pixel_envs.reset()

    # create eval env
    if Config.eval_env:
        eval_env = make_env(
            env_id=Config.env_id,
            num_envs=1, 
            parallel=False, 
            norm_obs=False,
            norm_reward=False,
            clip_actions=Config.clip_actions,
            clip_obs=Config.clip_observations,
            clip_rewards=Config.clip_rewards,
            evaluate=True,
            capture_video=Config.capture_video,
            video_path=video_dir,
            is_atari=is_atari,
            env_kwargs=Config.env_kwargs
        )
        eval_env.seed(int(eval_key[0]))
        eval_state = eval_env.reset()


    # initialize metrics
    global_step = 0
    start_time = time.time()
    training_walltime = 0
    scores = []

    # run initial eval
    if process_id == 0 and Config.eval_env:
        eval_start_time = time.time()
        eval_steps = 0
        if is_atari:
            feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
        policy_params = _unpmap(training_state.params.policy)
        policy = make_policy(policy_params, deterministic=Config.deterministic_eval)
        reward_params = _unpmap((rm_training_state.normalizer_params, rm_training_state.params.reward_head))
        reward_model = make_reward_fn(reward_params)

        predicted_returns_on_finished_traj = []
        curr_predicted_episode_returns = jnp.zeros_like(eval_state.reward)
        while True: 
            eval_steps += 1
            
            # run eval episode & record scores + lengths
            current_key, eval_key = jax.random.split(eval_key)
            obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
            if is_atari:
                obs = feature_extractor(obs)
            actions, policy_extras = policy(obs, current_key)
            actions = np.asarray(actions)
            eval_state = eval_env.step(actions) 

            # retrieve terminal observation for finished trajectories
            next_obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
            next_obs = jnp.array(next_obs)
            for env_idx, info in enumerate(eval_state.info):
                if "terminal_observation" in info:
                    next_obs = next_obs.at[env_idx].set(envs.normalize_obs(info["terminal_observation"]) 
                                                                 if Config.normalize_observations 
                                                                 else info["terminal_observation"])
            if discrete_action_space:
                actions = jnp.expand_dims(actions, axis=-1)
            predicted_reward = reward_model(observations=obs,
                                                actions=actions,
                                                next_observations=next_obs,)
            curr_predicted_episode_returns += predicted_reward
            for env_idx, info in enumerate(eval_state.info):
                if 'episode_return' in info:
                    predicted_returns_on_finished_traj.append(curr_predicted_episode_returns[env_idx])
                    curr_predicted_episode_returns.at[env_idx].set(0.)

            if len(eval_env.returns) >= Config.num_eval_episodes:
                eval_returns, eval_ep_lengths = eval_env.evaluate()
                break
        eval_state = eval_env.reset()
        eval_time = time.time() - eval_start_time

        # compute mean + std & record
        eval_metrics = {
            'eval/num_episodes': len(eval_returns),
            'eval/num_steps': eval_steps,
            'eval/mean_score': np.round(np.mean(eval_returns), 3),
            'eval/std_score': np.round(np.std(eval_returns), 3),
            'eval/mean_predicted_score': np.round(np.mean(predicted_returns_on_finished_traj), 3),
            'eval/std_predicted_score': np.round(np.std(predicted_returns_on_finished_traj), 3),
            'eval/mean_episode_length': np.mean(eval_ep_lengths),
            'eval/std_episode_length': np.round(np.std(eval_ep_lengths), 3),
            'eval/eval_time': eval_time,
        }
        logging.info(eval_metrics)
        scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), 0.))

    # training loop
    for training_step in range(1, num_training_steps + 1):
        update_time_start = time.time()

        new_key, local_key = jax.random.split(local_key)
        training_state, env_state = _strip_weak_type((training_state, env_state))
        (key_sgd, key_generate_unroll, key_collect_tractories, 
            key_segment_pairing, key_rm_sgd) = jax.random.split(new_key, 5)

        if is_atari:
            feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
        policy = make_policy(_unpmap(training_state.params.policy))

        # collect trajectories to compute preferences and train reward model
        pixel_env_state = pixel_envs.reset() 
        data = []
        for step in range(Config.num_sampled_trajectories // Config.num_envs):
            transitions = []
            for unroll_step in range(Config.episode_length):
                current_key, key_collect_tractories = jax.random.split(key_collect_tractories)  
                obs = pixel_env_state.obs
                obs = envs.normalize_obs(obs) if Config.normalize_observations else obs
                last_obs = jnp.copy(obs)
                if is_atari:
                    obs = feature_extractor(obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                nstate = pixel_envs.step(actions) 
                if discrete_action_space:
                    actions = jnp.expand_dims(actions, axis=-1)
                # NOTE: info transformed: Array[Dict] --> Dict[Array]
                state_extras = {'truncation': jnp.array([info['truncation'] for info in nstate.info]),
                                'pixels': jnp.array([info['pixels'] for info in nstate.info])} 
                
                # retrieve terminal observation for finished trajectories
                next_observations = jnp.array(nstate.obs)
                for env_idx, info in enumerate(nstate.info):
                    if "terminal_observation" in info:
                        next_observations = next_observations.at[env_idx].set(info["terminal_observation"])
                next_observations = envs.normalize_obs(next_observations) if Config.normalize_observations else next_observations

                transition = Transition(  
                    observation=last_obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    next_observation=next_observations,
                    extras={
                        'policy_extras': policy_extras,
                        'state_extras': state_extras
                })
                transitions.append(transition)
                pixel_env_state = nstate
            data.append(jax.tree_util.tree_map(lambda *x: np.stack(x), *transitions))
        data = jax.tree_util.tree_map(lambda *x: np.stack(x), *data)

        epoch_trajectory_collection_time = time.time() - update_time_start
        update_time_start = time.time()

        # Have leading dimensions (num_sampled_trajectories, episode_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
        assert data.discount.shape[1:] == (Config.episode_length,)
        
        # assert that each trajectory ends on a truncation (fixed episode length, no termination condition --> open-ended learning)
        assert jnp.all(jnp.take(data.extras['state_extras']['truncation'], indices=-1, axis=-1)), 'only fixed length episodes supported'
        
        
        # number of segment pairs on which preferences will be computed this iteration
        curr_num_preferences = schedule[training_step-1]
        # sample pairs of trajectory segments
        data = jitted_sample_fragment_pairs(trajectories=data,
                                                      num_sampled_pairs=curr_num_preferences,
                                                      key=key_segment_pairing)

        # compute preferences
        preferences = get_preferences(data)

        # discard image data before storing to save space (?)
        data = data._replace(extras=jnp.zeros_like(data.reward))

        # preference data
        preference_data = PreferenceData(segment_1=jax.tree_util.tree_map(lambda x: jnp.take(x, 0, axis=1), data),
                                         segment_2=jax.tree_util.tree_map(lambda x: jnp.take(x, 1, axis=1), data),
                                         preference=preferences)
        preference_data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (local_devices_to_use, -1,) + x.shape[1:]),
                                    preference_data)
        # pair dimension ([local_devices, B, 2, T, ...])
        chex.assert_shape(preference_data.preference, (local_devices_to_use, 
                            int(curr_num_preferences/local_devices_to_use))) 
        chex.assert_shape(preference_data.segment_1.discount, (local_devices_to_use, 
                            int(curr_num_preferences/local_devices_to_use), Config.segment_length)) 
        # buffer_state = _unpmap(buffer_state)
        # buffer_state = replay_buffer.insert(buffer_state, preference_data)
        # buffer_state = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (local_devices_to_use, -1,) + x.shape[1:]),
        #                             buffer_state)

        # train reward model
        keys_rm_sgd = jax.random.split(key_rm_sgd, local_devices_to_use)
        new_rm_training_state, buffer_state, metrics = train_reward_model(
            data=preference_data,
            reward_model_training_state=rm_training_state, 
            buffer_state=buffer_state, 
            key_sgd=keys_rm_sgd
        )
        rm_training_state, metrics = _strip_weak_type((new_rm_training_state, metrics))

        # logging     
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_update_time = time.time() - update_time_start
        training_walltime = time.time() - start_time # += epoch_update_time + epoch_rollout_time
        sps = Config.num_sampled_trajectories * Config.episode_length / (epoch_update_time + epoch_trajectory_collection_time)
                
        metrics = {
            'training/updates': training_step,
            'training/comparisons': curr_num_preferences,
            'training/sps': np.round(sps, 3),
            'training/walltime': np.round(training_walltime, 3), 
            'training/trajectory_collection_time': np.round(epoch_trajectory_collection_time, 3),
            'training/reward_training_time': np.round(epoch_update_time, 3),
            **{f'training/{name}': float(value) for name, value in metrics.items()}
        }

        logging.info(metrics)
        update_time_start = time.time()
        
        reward_params = _unpmap((rm_training_state.normalizer_params, rm_training_state.params.reward_head))
        reward_model = make_reward_fn(reward_params)

        # train policy
        data = []
        for step in range(Config.batch_size * Config.num_minibatches // Config.num_envs):
            transitions = []
            for unroll_step in range(Config.unroll_length):
                current_key, key_generate_unroll = jax.random.split(key_generate_unroll)  
                obs = env_state.obs
                if is_atari:
                    obs = feature_extractor(env_state.obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                nstate = envs.step(actions) 
                # NOTE: info transformed: Array[Dict] --> Dict[Array]
                state_extras = {'truncation': jnp.array([info['truncation'] for info in nstate.info])} 

                # retrieve terminal observation for finished trajectories
                next_observations = jnp.array(nstate.obs)
                for env_idx, info in enumerate(nstate.info):
                    if "terminal_observation" in info:
                        next_observations = next_observations.at[env_idx].set(info["terminal_observation"])

                if discrete_action_space:
                    actions = jnp.expand_dims(actions, axis=-1)
                # TODO: much more efficient to not compute reward on each step but on all transitions at once ?
                predicted_reward = reward_model(observations=env_state.obs,
                                                actions=actions,
                                                next_observations=next_observations,)

                transition = Transition(  
                    observation=env_state.obs,
                    action=actions,
                    reward=predicted_reward, # nstate.reward,
                    discount=1 - nstate.done,
                    next_observation=next_observations,
                    extras={
                        'policy_extras': policy_extras,
                        'state_extras': state_extras
                })
                transitions.append(transition)
                env_state = nstate
            data.append(jax.tree_util.tree_map(lambda *x: np.stack(x), *transitions))
        data = jax.tree_util.tree_map(lambda *x: np.stack(x), *data)

        

        epoch_rollout_time = time.time() - update_time_start
        update_time_start = time.time()

        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                    data)
        assert data.discount.shape[1:] == (Config.unroll_length,)

        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (local_devices_to_use, -1,) + x.shape[1:]),
                                    data)
        
        rm_training_state = update_reward_normalization(rm_training_state, data.reward)
        
        # as function 
        keys_sgd = jax.random.split(key_sgd, local_devices_to_use)
        new_training_state, metrics = learn(data=data, training_state=training_state, key_sgd=keys_sgd)
    
        # logging     
        training_state, env_state, metrics = _strip_weak_type((new_training_state, env_state, metrics))
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_update_time = time.time() - update_time_start
        training_walltime = time.time() - start_time # += epoch_update_time + epoch_rollout_time
        sps = env_step_per_training_step / (epoch_update_time + epoch_rollout_time)
        global_step += env_step_per_training_step
        
        current_step = int(_unpmap(training_state.env_steps))
        
        metrics = {
            'training/total_steps': current_step,
            'training/updates': training_step,
            'training/sps': np.round(sps, 3),
            'training/walltime': np.round(training_walltime, 3), 
            'training/trajectory_collection_time': np.round(epoch_trajectory_collection_time, 3),
            'training/rollout_time': np.round(epoch_rollout_time, 3),
            'training/update_time': np.round(epoch_update_time, 3),
            **{f'training/{name}': float(value) for name, value in metrics.items()}
        }

        logging.info(metrics)

        # run eval
        if process_id == 0 and Config.eval_env and training_step % Config.eval_every == 0:
            eval_start_time = time.time()
            eval_steps = 0
            if is_atari:
                feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
            policy_params = _unpmap(training_state.params.policy)
            policy = make_policy(policy_params, deterministic=Config.deterministic_eval)

            predicted_returns_on_finished_traj = []
            curr_predicted_episode_returns = jnp.zeros_like(eval_state.reward)
            while True: 
                eval_steps += 1
                
                # run eval episode & record scores + lengths
                current_key, eval_key = jax.random.split(eval_key)
                obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
                if is_atari:
                    obs = feature_extractor(obs)
                actions, policy_extras = policy(obs, current_key)
                actions = np.asarray(actions)
                eval_state = eval_env.step(actions) 

                # retrieve terminal observation for finished trajectories
                next_obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
                next_obs = jnp.array(next_obs)
                for env_idx, info in enumerate(eval_state.info):
                    if "terminal_observation" in info:
                        next_obs = next_obs.at[env_idx].set(envs.normalize_obs(info["terminal_observation"]) 
                                                                    if Config.normalize_observations 
                                                                    else info["terminal_observation"])
                if discrete_action_space:
                    actions = jnp.expand_dims(actions, axis=-1)
                predicted_reward = reward_model(observations=obs,
                                                    actions=actions,
                                                    next_observations=next_obs,)
                curr_predicted_episode_returns += predicted_reward
                for env_idx, info in enumerate(eval_state.info):
                    if 'episode_return' in info:
                        predicted_returns_on_finished_traj.append(curr_predicted_episode_returns[env_idx])
                        curr_predicted_episode_returns.at[env_idx].set(0.)

                if len(eval_env.returns) >= Config.num_eval_episodes:
                    eval_returns, eval_ep_lengths = eval_env.evaluate()
                    break
            eval_state = eval_env.reset()
            eval_time = time.time() - eval_start_time
            # compute mean + std & record
            eval_metrics = {
                'eval/num_episodes': len(eval_returns),
                'eval/num_steps': eval_steps,
                'eval/mean_score': np.round(np.mean(eval_returns), 3),
                'eval/std_score': np.round(np.std(eval_returns), 3),
                'eval/mean_predicted_score': np.round(np.mean(predicted_returns_on_finished_traj), 3),
                'eval/std_predicted_score': np.round(np.std(predicted_returns_on_finished_traj), 3),
                'eval/mean_episode_length': np.mean(eval_ep_lengths),
                'eval/std_episode_length': np.round(np.std(eval_ep_lengths), 3),
                'eval/eval_time': eval_time,
            }
            logging.info(eval_metrics)
            scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), metrics['training/approx_kl']))
        
    logging.info('TRAINING END: training duration: %s', time.time() - start_time)

    # final eval
    if process_id == 0 and Config.eval_env:
        eval_steps = 0
        if is_atari:
            feature_extractor = make_feature_extractor(_unpmap(training_state.params.feature_extractor))
        policy_params = _unpmap(training_state.params.policy)
        policy = make_policy(policy_params, deterministic=True)

        predicted_returns_on_finished_traj = []
        curr_predicted_episode_returns = jnp.zeros_like(eval_state.reward)
        while True: 
            eval_steps += 1
            
            # run eval episode & record scores + lengths
            current_key, eval_key = jax.random.split(eval_key)
            obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
            if is_atari:
                obs = feature_extractor(obs)
            actions, policy_extras = policy(obs, current_key)
            actions = np.asarray(actions)
            eval_state = eval_env.step(actions) 

            # retrieve terminal observation for finished trajectories
            next_obs = envs.normalize_obs(eval_state.obs) if Config.normalize_observations else eval_state.obs
            next_obs = jnp.array(next_obs)
            for env_idx, info in enumerate(eval_state.info):
                if "terminal_observation" in info:
                    next_obs = next_obs.at[env_idx].set(envs.normalize_obs(info["terminal_observation"]) 
                                                                if Config.normalize_observations 
                                                                else info["terminal_observation"])

            if discrete_action_space:
                actions = jnp.expand_dims(actions, axis=-1)
            predicted_reward = reward_model(observations=obs,
                                                actions=actions,
                                                next_observations=next_obs,)
            curr_predicted_episode_returns += predicted_reward
            for env_idx, info in enumerate(eval_state.info):
                if 'episode_return' in info:
                    predicted_returns_on_finished_traj.append(curr_predicted_episode_returns[env_idx])
                    curr_predicted_episode_returns.at[env_idx].set(0.)

            if len(eval_env.returns) >= Config.num_eval_episodes:
                eval_returns, eval_ep_lengths = eval_env.evaluate()
                break
        eval_state = eval_env.reset()
        # compute mean + std & record
        eval_metrics = {
            'final_eval/num_episodes': len(eval_returns),
            'final_eval/num_steps': eval_steps,
            'final_eval/mean_score': np.mean(eval_returns),
            'final_eval/std_score': np.std(eval_returns),
            'eval/mean_predicted_score': np.round(np.mean(predicted_returns_on_finished_traj), 3),
            'eval/std_predicted_score': np.round(np.std(predicted_returns_on_finished_traj), 3),
            'final_eval/mean_episode_length': np.mean(eval_ep_lengths),
            'final_eval/std_episode_length': np.std(eval_ep_lengths),
        }
        logging.info(eval_metrics)
        scores.append((global_step, np.mean(eval_returns), np.mean(eval_ep_lengths), None))

        # save scores 
        run_dir = os.path.join('experiments', run_name)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        with open(os.path.join(run_dir, "scores.pkl"), "wb") as f:
            pickle.dump(scores, f)

    if Config.save_model:
        model_path = f"weights/{run_name}.params"
        with open(model_path, "wb") as f:
            f.write(
                flax.serialization.to_bytes(
                    [
                        vars(Config),
                        [
                            training_state.params.policy,
                            training_state.params.value,
                            # agent_state.params.feature_extractor,
                        ],
                    ]
                )
            )
        print(f"model saved to {model_path}")

    envs.close()


if __name__ == "__main__":
    app.run(main)