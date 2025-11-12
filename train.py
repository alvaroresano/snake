from datetime import datetime
from pathlib import Path
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
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from snake_env.snake_env.env import SnakeEnv


def make_env(
    env_kwargs: Dict[str, Any],
    seed: int,
    rank: int,
    monitor_dir: Path,
):
    """Factory to create monitored environments for each vectorized worker."""

    monitor_dir = Path(monitor_dir)

    def _init():
        env = SnakeEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_file = monitor_dir / f"env_{rank}.monitor.csv"
        return Monitor(env, filename=str(monitor_file))

    return _init


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> float:
    seed = int(getattr(cfg, "seed", 0))
    set_random_seed(seed)

    env_kwargs = dict(OmegaConf.to_container(cfg.env, resolve=True))
    logging_cfg = dict(OmegaConf.to_container(cfg.logging, resolve=True))
    training_cfg = dict(OmegaConf.to_container(cfg.training, resolve=True))
    checkpoint_to_load = training_cfg.get("load_from_checkpoint", None)
    callbacks_cfg = dict(OmegaConf.to_container(cfg.callbacks, resolve=True))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(logging_cfg.get("run_root", "artifacts/runs")) / run_id
    checkpoint_dir = Path(logging_cfg["checkpoint_dir"]) / run_id
    tensorboard_dir = Path(logging_cfg["tensorboard_dir"]) / run_id
    eval_log_dir = Path(logging_cfg["eval_log_dir"]) / run_id
    monitor_dir = run_root / "monitor" / "train"
    eval_monitor_dir = run_root / "monitor" / "eval"

    for path in (
        run_root,
        checkpoint_dir,
        tensorboard_dir,
        eval_log_dir,
        monitor_dir,
        eval_monitor_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    config_dump_path = run_root / "config.yaml"
    config_dump_path.write_text(OmegaConf.to_yaml(cfg))

    vec_envs = [
        make_env(env_kwargs, seed, idx, monitor_dir)
        for idx in range(training_cfg["num_envs"])
    ]
    env = DummyVecEnv(vec_envs)
    eval_env = DummyVecEnv(
        [make_env(env_kwargs, seed + 10_000, 0, eval_monitor_dir)]
    )

    final_model_base = Path(logging_cfg["final_model_path"])
    final_model_path = final_model_base.with_name(
        f"{final_model_base.name}_{run_id}"
    )
    if final_model_path.suffix != ".zip":
        final_model_path = final_model_path.with_suffix(".zip")
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_to_load:
        print(f"Loading model from: {checkpoint_to_load}")
        model = DQN.load(checkpoint_to_load, env=env)
        model.learning_rate = training_cfg["learning_rate"]
    else:
        print("Creating new model")
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
            tensorboard_log=str(tensorboard_dir),
            verbose=0,
            device=training_cfg["device"],
            seed=seed,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=callbacks_cfg["checkpoint_freq"],
        save_path=str(checkpoint_dir),
        name_prefix=callbacks_cfg["checkpoint_prefix"],
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_freq = max(
        1, callbacks_cfg["eval_freq"] // max(1, training_cfg["num_envs"])
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_log_dir),
        log_path=str(eval_log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=callbacks_cfg["n_eval_episodes"],
        deterministic=callbacks_cfg["deterministic_eval"],
    )

    callback_list: List = [checkpoint_callback, eval_callback]

    model.learn(
        total_timesteps=training_cfg["total_timesteps"],
        callback=CallbackList(callback_list),
    )

    best_mean_reward = eval_callback.best_mean_reward

    model.save(str(final_model_path))
    env.close()
    eval_env.close()

    return best_mean_reward


if __name__ == "__main__":
    main()
