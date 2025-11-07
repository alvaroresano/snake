import os
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_env.snake_env.env import SnakeEnv


def make_env(env_kwargs: Dict[str, Any]):
    """Factory to create monitored environments for each vectorized worker."""

    def _init():
        env = SnakeEnv(**env_kwargs)
        return Monitor(env)

    return _init


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    env_kwargs = OmegaConf.to_container(cfg.env, resolve=True)
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    callbacks_cfg = OmegaConf.to_container(cfg.callbacks, resolve=True)

    os.makedirs(logging_cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(logging_cfg["tensorboard_dir"], exist_ok=True)
    os.makedirs(logging_cfg["eval_log_dir"], exist_ok=True)

    vec_envs = [
        make_env(env_kwargs) for _ in range(training_cfg["num_envs"])
    ]
    env = DummyVecEnv(vec_envs)
    eval_env = DummyVecEnv([make_env(env_kwargs)])

    model = DQN(
        training_cfg["policy"],
        env,
        learning_rate=training_cfg["learning_rate"],
        buffer_size=training_cfg["buffer_size"],
        learning_starts=training_cfg["learning_starts"],
        batch_size=training_cfg["batch_size"],
        tau=training_cfg["tau"],
        gamma=training_cfg["gamma"],
        train_freq=training_cfg["train_freq"],
        gradient_steps=training_cfg["gradient_steps"],
        target_update_interval=training_cfg["target_update_interval"],
        exploration_fraction=training_cfg["exploration_fraction"],
        exploration_final_eps=training_cfg["exploration_final_eps"],
        tensorboard_log=logging_cfg["tensorboard_dir"],
        verbose=training_cfg["verbose"],
        device=training_cfg["device"],
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=callbacks_cfg["checkpoint_freq"],
        save_path=logging_cfg["checkpoint_dir"],
        name_prefix=callbacks_cfg["checkpoint_prefix"],
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_freq = max(
        1, callbacks_cfg["eval_freq"] // max(1, training_cfg["num_envs"])
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=logging_cfg["eval_log_dir"],
        log_path=logging_cfg["eval_log_dir"],
        eval_freq=eval_freq,
        n_eval_episodes=callbacks_cfg["n_eval_episodes"],
        deterministic=callbacks_cfg["deterministic_eval"],
    )

    callback_list: List = [checkpoint_callback, eval_callback]

    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=CallbackList(callback_list),
    )

    model.save(logging_cfg["final_model_path"])
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
