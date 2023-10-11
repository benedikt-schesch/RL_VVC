

import envs.env
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from stable_baselines3 import SAC, DQN, PPO
from pathlib import Path
import os


def get_model(model_config: dict, env: gym.Env, logdir: Path):
    if model_config["model_name"] == "dqn":
        return DQN("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))
    elif model_config["model_name"] == "sac":
        return SAC("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))
    elif model_config["model_name"] == "ppo":
        return PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))
    elif model_config["model_name"] == "a2c":
        return A2C("MlpPolicy", env, verbose=1, tensorboard_log=str(logdir))
    else:
        raise ValueError("Invalid model name: ", model_config["model_name"])


def get_env(env_config: dict, eval: bool = False):
    env_config = {
        "env": env_config["env_name"],
        "state_option": env_config["state_option"],
        "reward_option": env_config["reward_option"],
        "discrete_action": env_config["discrete_action"],
        "path": os.getcwd(),
        "eval": eval,
    }
    env = gym.make("VCCEnv-0", options=env_config)
    check_env(env)
    return env