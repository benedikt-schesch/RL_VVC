from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from utils import get_env, get_model
from pathlib import Path
import argparse
import yaml
from eval import evaluate

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record(
            "relative_reward", self.training_env.get_attr("relative_reward")
        )
        if self.num_timesteps % 2000 == 0:
            self.logger.dump(self.num_timesteps)
        return True


# envs = ['13', '123', '8500']
# algos = ['dqn', 'sac']
def train(
    env_config: dict,
    model_config: dict,
    logdir: Path,
):
    env = get_env(env_config, eval=False)

    # Create model
    model = get_model(model_config, env, logdir)

    # Instantiate the agent
    # Train the agent and display a progress bar
    save_dir = logdir / "checkpoints"
    callbacks = CallbackList(
        [
            TensorboardCallback(),
            CheckpointCallback(save_freq=10000, save_path=str(save_dir), verbose=1),
        ]
    )
    model.learn(total_timesteps=int(1), progress_bar=True, callback=callbacks)
    model.save(logdir / "final_model.pt")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="configs/models/ppo.yaml")
    parser.add_argument("--env_config", type=str, default="configs/envs/13_discrete.yaml")
    parser.add_argument("--logdir", type=str, default="outputs/13_discrete/ppo/")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    with open(args.model_config, "r") as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.env_config, "r") as f:
        env_config = yaml.load(f, Loader=yaml.FullLoader)

    model = train(
        env_config=env_config,
        model_config=model_config,
        logdir=logdir,
    )

    evaluate(
        model=model,
        env_config=env_config,
        logdir=logdir,
    )
