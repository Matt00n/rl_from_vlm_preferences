# Multimodal Preference-based Reinforcement Learning with PPO

## Overview

This repository implements a Proximal Policy Optimization (PPO) agent that learns to perform tasks based on preference feedback from a multimodal foundation model. Inspired by [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) and [RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback](https://arxiv.org/pdf/2402.03681), the system learns solely from pairwise comparisons of agent-generated trajectories, using frame-based multimodal preferences instead of scalar rewards.

---

## Key features:

* PPO-based RL: Stable on-policy policy gradient updates via clipping and adaptive KL penalties.
* Multimodal preference model: Uses OpenAI’s multimodal foundation model to compare trajectory segments and infer reward signals.
* Flexible environment support: Works with any Gym-compatible environment producing visual observations (e.g., Atari, MuJoCo with rendered frames).

## Setup & Installation

Clone the repository:

```bash
git clone https://github.com/Matt00n/rl_from_vlm_preferences.git
cd rl_from_vlm_preferences
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Export your OpenAI API key (required for preference queries):

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## Configuration

All hyperparameters and settings are defined directly in ppo.py. You can edit values such as the environment name, PPO rollout length, clipping parameter, learning rate, and preference modelling parameters at the top of the script to suit your needs.

## Usage

Train the PPO agent with multimodal preference feedback by running:

python ppo.py

There are no additional command-line arguments; all settings must be adjusted within ppo.py.

## References

Christiano, P. et al. “Deep Reinforcement Learning from Human Preferences,” 2017. https://arxiv.org/abs/1706.03741

Shang, X. et al. “RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback,” 2024. https://arxiv.org/pdf/2402.03681

