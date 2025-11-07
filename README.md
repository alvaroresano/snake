# snake
Reinforcement Learning Group Challenge – autonomous Snake agent.

## Project state

- **Custom Gymnasium env (`snake_env/snake_env/env.py`)**: exposes a configurable grid Snake world with OpenCV rendering, shaped rewards, max-step truncation, and a compact observation that now includes positional data, action history, directional sensors (apple/body visibility, wall distance in heading), and normalized distance cues. All observation features are normalized to `[0, 1]` or `[-1, 1]`, which stabilizes function approximation. We therefore *already* normalize observations; this avoids feature-scale imbalance and speeds up SB3 training.
- **Reward structure**: +10 for eating, −10 on death, ±1 proximity shaping, and −0.1 living cost—all adjustable via `reward_config`. Step/episode limits and auto-respawn prevent infinite loops.
- **Training stack**: `train.py` now uses Stable-Baselines3’s DQN driven by Hydra (`configs/train.yaml`). It instantiates monitored vector environments, writes TensorBoard logs, saves checkpoints/replay buffers, and evaluates policies mid-training.
- **Logging**: artifacts land under `artifacts/` by default—`tensorboard/`, `checkpoints/`, `eval/`, plus a final `snake_dqn_final.zip`. SB3’s TensorBoard summaries track episode reward/length, epsilon schedule, losses, etc.

## Why normalization matters

Snake coordinates range up to the grid size (default 200). Feeding raw pixels into an MLP forces it to learn vastly different scales for snake length vs. action IDs, which slows convergence or causes divergence when you tweak the grid. Normalizing (dividing by `grid_size` or `SNAKE_LEN_GOAL`) keeps gradients well behaved and allows configs to change without retraining feature scalers. Because we already return normalized floats, **no extra normalization wrapper is required** unless you introduce image observations later.

## Training pipeline (SB3 DQN + Hydra)

`train.py` is a Hydra entry point. Configuration lives in `configs/train.yaml`:

- `env`: parameters passed to `SnakeEnv` (grid size, max steps, reward weights).
- `training`: DQN hyperparameters (policy architecture, replay buffer, epsilon schedule, timesteps, number of parallel envs).
- `callbacks`: checkpoint cadence, filename prefix, evaluation frequency/episodes.
- `logging`: artifact directories (TensorBoard, checkpoints, eval stats, final model).
- `hydra`: pins Hydra’s run/output dirs so files stay in the project root.

### Running training

```bash
pip install -r requirements.txt
python3 train.py         # uses configs/train.yaml defaults
```

Override any field inline:

```bash
python3 train.py env.grid_size=300 env.max_steps=1500 \
  training.total_timesteps=2000000 \
  logging.checkpoint_dir=artifacts/big-grid
```

Hydra validates keys, so typos are caught early. Every run writes:

- `artifacts/tensorboard/<timestamp>` – SB3 summaries viewable via `tensorboard --logdir artifacts/tensorboard`.
- `artifacts/checkpoints/snake_dqn_<step>` – rolling checkpoints + replay buffers.
- `artifacts/eval/` – evaluation CSVs + best-model snapshots from `EvalCallback`.
- `artifacts/snake_dqn_final.zip` – final policy after `total_timesteps`.

### TensorBoard essentials

Launch:

```bash
tensorboard --logdir artifacts/tensorboard
```

Key signals:

- `rollout/ep_rew_mean`, `rollout/ep_len_mean`: did the agent learn longer survival?
- `train/loss`, `train/td_error`: DQN stability.
- `train/exploration_rate`: epsilon decay; adjust `exploration_fraction/final_eps` if exploration collapses too soon.

Use Hydra overrides to experiment systematically (e.g., `training.learning_rate=1e-4 callbacks.eval_freq=5000`).

### Evaluation + rendering

The training loop spawns a dedicated eval env that runs `callbacks.n_eval_episodes` episodes every `callbacks.eval_freq` steps and saves the best-performing checkpoint. To watch a trained agent:

```python
from stable_baselines3 import DQN
from snake_env.snake_env.env import SnakeEnv

model = DQN.load("artifacts/snake_dqn_final.zip")
env = SnakeEnv(render_mode="human", render_fps=30)
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()
env.close()
```

## Hydra crash course

- Config file lives at `configs/train.yaml`.
- Any nested key can be overridden from the CLI (`training.batch_size=128`).
- Lists/dicts can be replaced entirely by specifying the full path (e.g., `env.reward_config.step=-0.05`).
- You can create alternate configs (`configs/big_grid.yaml`) and select them with `python3 train.py --config-name=big_grid`.
- Current setup fixes `hydra.run.dir` to `.` so outputs stay in-place; remove that block if you prefer per-run folders.

## References

- https://www.researchgate.net/publication/374997396_Playing_the_Snake_Game_with_Reinforcement_Learning
