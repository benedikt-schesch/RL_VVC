from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from tqdm import tqdm
from utils import get_env, get_model


def eval(f_model,env,n_eval_episodes: int = 32):
    rewards = []
    rewards_baseline = []
    for j in tqdm(range(n_eval_episodes)):
        obs, info = env.reset(seed=j)
        rewards.append([])
        rewards_baseline.append([])
        done = False
        while not done:
            obs, reward, done, terminated, info = env.step(
                f_model(obs)
            )
            done = done or terminated
            rewards[j].append(reward)
            rewards_baseline[j].append(info["baseline_reward"])
    rewards = np.array(rewards)
    rewards_baseline = np.array(rewards_baseline)
    rewards = np.mean(rewards, axis=0)
    rewards_baseline = np.mean(rewards_baseline, axis=0)
    return rewards, rewards_baseline

def evaluate(
    model,
    env_config: dict,
    logdir: Path,
    n_eval_episodes: int = 32,
):
    env = get_env(env_config, eval=True)
    assert env.eval_mode # type: ignore
    rewards, _ = eval((lambda x: model.predict(x, deterministic=True)[0]),env,n_eval_episodes)
    plt.plot(range(len(rewards)), rewards, label="Model")
    rewards, _ = eval((lambda x: None),env, n_eval_episodes)
    plt.plot(range(len(rewards)), rewards, label="Baseline")
    rewards, _ = eval((lambda x: env.action_space.sample()),env, n_eval_episodes)
    plt.plot(range(len(rewards)), rewards, label="Random Agent", linestyle="--")
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.ylim(top=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(logdir / "rewards.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/13/sac/final_model.pt")
    parser.add_argument("--model_config", type=str, default="configs/models/sac.yaml")
    parser.add_argument("--env_config", type=str, default="configs/envs/13.yaml")
    parser.add_argument("--logdir", type=str, default="outputs/13/sac")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(args.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.env_config, "r") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)
    env = get_env(env_config, eval=True)
    model = get_model(model_config, env, logdir)
    model.load(args.checkpoint)

    evaluate(
        model=model,
        env_config=env_config,
        logdir=logdir,
    )
