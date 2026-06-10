# 🐍 Snake — Reinforcement Learning with PPO

> A custom Gymnasium environment for the classic Snake game, trained with Proximal Policy Optimization (PPO) via Stable-Baselines3.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Design](#environment-design)
  - [State Space (Observation)](#state-space-observation)
  - [Action Space](#action-space)
  - [Reward Function](#reward-function)
  - [Episode Termination](#episode-termination)
  - [Registered Variants](#registered-variants)
- [Reward Wrapper: NoGrowthRewardWrapper](#reward-wrapper-nogrowthrewardwrapper)
- [Algorithm: PPO](#algorithm-ppo)
- [Training Results](#training-results)
- [Installation](#installation)
- [Usage](#usage)
  - [Validate the Environment](#validate-the-environment)
  - [Run a Random Agent](#run-a-random-agent)
  - [Evaluate the Trained Model](#evaluate-the-trained-model)
  - [Test the Wrapper](#test-the-wrapper)
- [Dependencies](#dependencies)

---

## Overview

This project was developed as a **Reinforcement Learning group challenge**. The goal is to train an autonomous agent to play the Snake game by learning from experience — with no hand-coded rules for movement decisions.

The environment is built from scratch using the [Gymnasium](https://gymnasium.farama.org/) API (the maintained successor of OpenAI Gym), and training is done using the **PPO** (Proximal Policy Optimization) algorithm from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/). The game is rendered in real-time using OpenCV.

---

## Project Structure

```
snake-main/
├── requirements.txt                     # Full pinned dependency list
└── snake_env/
    ├── requirements.txt                 # Minimal: stable-baselines3[extra]
    ├── evaluation.py                    # Load and visually run the trained PPO model
    ├── test_snippet.py                  # Run a random agent to validate the env
    ├── test_wrapper.py                  # Test NoGrowthRewardWrapper manually
    ├── models/
    │   └── PPO/
    │       ├── 100000.zip               # Checkpoint at 100,000 training steps
    │       └── best_model/
    │           └── best_model.zip       # Best model saved by EvalCallback
    ├── logs/
    │   ├── PPO_0/                       # TensorBoard logs for training run 0
    │   │   ├── events.out.tfevents.*    # Pre-training initialization log
    │   │   └── events.out.tfevents.*    # Main training log
    │   └── eval/
    │       └── evaluations.npz          # Evaluation snapshots (rewards + ep lengths)
    └── snake_env/
        ├── __init__.py                  # Gymnasium env registration
        ├── env.py                       # Core SnakeEnv class (gym.Env)
        ├── snake_wrapper.py             # NoGrowthRewardWrapper (gym.Wrapper)
        └── checkenv.py                  # Stable-Baselines3 env sanity check
```

---

## Environment Design

The Snake environment (`SnakeEnv`) is defined in `snake_env/env.py` and inherits from `gymnasium.Env`. The game runs on a **500×500 pixel grid** with cells of size 10×10 pixels, giving an effective grid of **50×50 cells**.

### State Space (Observation)

The observation is a **1D vector of shape `(10,)`** — specifically `(5 + STACKED_ACTIONS,)` where `STACKED_ACTIONS = 5`.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `head_x` | X coordinate of the snake's head (pixels) |
| 1 | `head_y` | Y coordinate of the snake's head (pixels) |
| 2 | `apple_delta_x` | `apple_x − head_x` (signed distance to apple, X axis) |
| 3 | `apple_delta_y` | `apple_y − head_y` (signed distance to apple, Y axis) |
| 4 | `snake_length` | Current number of body segments |
| 5–9 | `prev_actions[0..4]` | Last 5 actions taken (deque, initialized to −1) |

All values are `float64`. The observation space bounds are `[−500, 500]` for all dimensions.

The stacked action history gives the agent temporal context — it can infer its current direction of travel without explicit directional state, and detect oscillatory or circular movement patterns.

### Action Space

`Discrete(4)` — four possible movement directions:

| Action | Direction | Pixel delta |
|--------|-----------|-------------|
| `0` | Left | `head_x −= 10` |
| `1` | Right | `head_x += 10` |
| `2` | Down | `head_y += 10` |
| `3` | Up | `head_y −= 10` |

**Reverse-direction blocking:** The environment prevents the snake from reversing into itself. If the agent tries to move in the exact opposite of the current direction (e.g., left when currently moving right), the action is silently overridden to continue in the current direction. This matches real Snake game rules.

### Reward Function

The base `SnakeEnv` uses a **shaped reward** designed to guide the agent toward the apple while penalising death:

| Event | Reward |
|-------|--------|
| Eating an apple | **+10** |
| Death (boundary or self-collision) | **−10** |
| Moving closer to the apple | **+1** |
| Moving farther from the apple | **−1** |
| Each step (survival penalty) | **−0.1** |

Distance is computed as Euclidean distance (`np.linalg.norm`) between the snake head and the apple in pixel space.

The survival penalty (`−0.1` per step) discourages the agent from stalling or looping indefinitely.

### Episode Termination

An episode ends (`terminated = True`) when:
- The snake's head exits the grid boundaries (`x < 0`, `x ≥ 500`, `y < 0`, `y ≥ 500`).
- The snake's head overlaps any of its own body segments.

`truncated` is always `False` in the base environment — truncation is handled by Gymnasium's `TimeLimit` wrapper applied at registration time.

### Registered Variants

All variants are registered in `snake_env/__init__.py`:

| Environment ID | `max_episode_steps` | `action_size` | Notes |
|---|---|---|---|
| `Snake-v0` | 500 | 1 (10 px/step) | Default training env |
| `Snake-v0-step5` | 500 | 2 (20 px/step) | Faster snake movement |
| `Snake-v1` | 100 | 1 | Short episodes |
| `Snake-v2` | 500 | 1 | `reward_threshold=30` |
| `Snake-Base-For-Wrapping-v0` | 10,000 | 1 | Base for `NoGrowthRewardWrapper` |

---

## Reward Wrapper: NoGrowthRewardWrapper

Defined in `snake_env/snake_wrapper.py`, this `gym.Wrapper` completely overrides the base reward with a more elaborate scheme tailored for an agent that **does not grow** when eating an apple (the snake stays the same length throughout the episode).

### Reward Logic

| Event | Reward |
|-------|--------|
| Collision (death) | **−10,000** |
| Eating an apple | **+1,000** |
| Moving closer to apple | **+5** |
| Moving farther from apple | **−1** |
| Time penalty (per step without eating) | **−0.01 × steps_since_last_apple** |
| Distance penalty (per step) | **−current_distance / 100** |

The time penalty **accumulates** with each step since the last apple was eaten, applying increasing pressure on the agent to seek food efficiently. The distance penalty provides a continuous gradient toward the apple.

### No-Growth Mechanic

After the apple is eaten, the wrapper immediately calls `self.unwrapped.snake_position.pop()` to cancel the body growth that the base environment added. This simplifies the problem — the agent does not need to worry about its own body length increasing over time.

### Internal State

The wrapper tracks:
- `steps_since_last_apple` — reset to 0 on each apple collection.
- `previous_distance_to_apple` — updated every step; used for the proximity comparison.

---

## Algorithm: PPO

The agent is trained using **Proximal Policy Optimization (PPO)**, a robust on-policy policy gradient algorithm well-suited for continuous action feedback environments.

PPO optimizes a clipped surrogate objective to prevent destructively large policy updates:

```
L_CLIP(θ) = E[ min( r_t(θ) · A_t,  clip(r_t(θ), 1−ε, 1+ε) · A_t ) ]
```

Where `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` is the probability ratio and `A_t` is the advantage estimate.

### Why PPO for Snake?

- Snake has **discrete actions** and **dense, shaped rewards** — ideal for PPO.
- PPO handles **non-stationary environments** well (the apple repositions randomly on each collection).
- Stable-Baselines3's PPO implementation includes **Generalized Advantage Estimation (GAE)**, which reduces variance in the policy gradient.

### Saved Checkpoints

| File | Timestep | Description |
|------|----------|-------------|
| `models/PPO/100000.zip` | 100,000 | Periodic checkpoint |
| `models/PPO/best_model/best_model.zip` | Varies | Best mean reward during evaluation |

---

## Training Results

Evaluation was run every 10,000 timesteps using SB3's `EvalCallback` with **5 evaluation episodes** per checkpoint. Results are stored in `logs/eval/evaluations.npz`.

| Timestep | Mean Reward | Min | Max | Mean Episode Length |
|----------|-------------|-----|-----|---------------------|
| 10,000 | −18.9 | −36.5 | −5.1 | 33.4 |
| 20,000 | +31.7 | −28.8 | +96.4 | 83.0 |
| 30,000 | −3.0 | −36.5 | +23.9 | 45.8 |
| 40,000 | +9.3 | −36.5 | +105.2 | 55.2 |
| 50,000 | −16.9 | −37.6 | +32.8 | 45.4 |
| 60,000 | −22.8 | −39.1 | +3.3 | 39.6 |
| 70,000 | +27.8 | −36.5 | +86.9 | 58.2 |
| 80,000 | −1.2 | −36.5 | +23.6 | 35.8 |
| 90,000 | +4.5 | −36.5 | +58.5 | 60.0 |
| 100,000 | +5.6 | −36.5 | +53.9 | 49.6 |

The training curve shows high variance between evaluation episodes, which is expected for early-stage RL training on Snake — the episode outcome is very sensitive to the random initial apple position and the agent's stochastic policy. The best individual episode reward reached **+105.2** at the 40,000-step checkpoint.

To visualise training with TensorBoard:

```bash
tensorboard --logdir snake_env/logs/PPO_0
```

---

## Installation

**Requirements:** Python 3.12 (used during development, as indicated by `.cpython-312.pyc` cache files).

### 1. Clone the repository

```bash
git clone <repo-url>
cd snake-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

**Full pinned environment (recommended for exact reproducibility):**

```bash
pip install -r requirements.txt
```

**Minimal install (latest compatible versions):**

```bash
pip install -r snake_env/requirements.txt
```

The minimal install (`stable-baselines3[extra]`) pulls in Gymnasium, PyTorch, TensorBoard, OpenCV, and all other required packages automatically.

### 4. Install the custom environment as a package

```bash
cd snake_env
pip install -e .
```

> This step makes `import snake_env` work from any script and registers all `Snake-*` environment IDs with Gymnasium.

---

## Usage

All scripts below should be run from inside the `snake_env/` directory.

### Validate the Environment

Runs Stable-Baselines3's `check_env` to verify the environment conforms to the Gymnasium API (observation/action space shapes, dtypes, `reset`/`step` return signatures):

```bash
cd snake_env/snake_env
python checkenv.py
```

### Run a Random Agent

Runs a random policy (uniform action sampling) on `Snake-v0` with visual rendering. Useful for inspecting the environment's behaviour and rendering pipeline:

```bash
cd snake_env
python test_snippet.py
```

The snake moves randomly until it hits a wall or itself. The OpenCV window shows the snake (head in **cyan**, body in **green**) and the apple (in **red**).

### Evaluate the Trained Model

Loads the PPO checkpoint at 100,000 steps and runs 5 complete episodes with visual rendering:

```bash
cd snake_env
python evaluation.py
```

The script uses `deterministic=True` in `model.predict()` — the agent always picks the highest-probability action rather than sampling, which gives the cleanest evaluation of learned behaviour.

To use the best saved model instead, edit `evaluation.py`:

```python
# Change this line:
model_path = "models/PPO/100000"
# To:
model_path = "models/PPO/best_model/best_model"
```

### Test the Wrapper

Runs a random policy on `Snake-Base-For-Wrapping-v0` wrapped with `NoGrowthRewardWrapper`. Prints reward and total reward to the terminal whenever a non-zero reward is received:

```bash
cd snake_env
python test_wrapper.py
```

This is useful for verifying that the wrapper's reward logic behaves as expected before using it for training.

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `gymnasium` | 1.2.2 | RL environment API |
| `stable-baselines3` | 2.7.0 | PPO implementation |
| `torch` | 2.9.0 | Neural network backend |
| `numpy` | 2.3.4 | Numerical operations |
| `opencv-python` | 4.11.0.86 | Game rendering |
| `tensorboard` | 2.20.0 | Training visualisation |
| `matplotlib` | 3.10.7 | Plotting utilities |
| `pandas` | 2.3.3 | Data handling |

See `requirements.txt` for the full pinned dependency list.
