from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import snake_env  # Registers Snake-v0
from stable_baselines3 import DQN


def list_run_ids(runs_dir: Path) -> list[str]:
    if not runs_dir.exists():
        return []
    return sorted(d.name for d in runs_dir.iterdir() if d.is_dir())


def resolve_run_id(explicit: str | None, runs_dir: Path) -> str:
    if explicit:
        return explicit
    run_ids = list_run_ids(runs_dir)
    if not run_ids:
        raise FileNotFoundError(
            f"No runs found under {runs_dir}. Train an agent first."
        )
    return run_ids[-1]


def resolve_model_path(
    *,
    model_path: str | None,
    artifacts_dir: Path,
    eval_dir: Path,
    run_id: str,
    use_best: bool,
) -> Path:
    if model_path:
        path = Path(model_path)
    else:
        path = (
            eval_dir / run_id / "best_model.zip"
            if use_best
            else artifacts_dir / f"snake_dqn_final_{run_id}.zip"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find model file at {path}. "
            "Use --list-runs to inspect available run IDs."
        )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trained Snake agent using the latest (or specified) run."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Explicit path to a .zip policy. Overrides --run-id/--best.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Timestamped run ID (e.g., 20250109-165500)."
        " Defaults to the latest folder under artifacts/runs.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Load the EvalCallback best_model.zip instead of the final policy.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="How many episodes to render.",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="Render speed for the Gymnasium environment.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print discovered run IDs and exit.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Base artifacts directory (default: artifacts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    runs_dir = artifacts_dir / "runs"
    eval_dir = artifacts_dir / "eval"

    if args.list_runs:
        run_ids = list_run_ids(runs_dir)
        if run_ids:
            print("Available run IDs:")
            for rid in run_ids:
                print(f"  - {rid}")
        else:
            print(f"No runs found in {runs_dir}")
        return

    run_id = resolve_run_id(args.run_id, runs_dir)
    model_path = resolve_model_path(
        model_path=args.model_path,
        artifacts_dir=artifacts_dir,
        eval_dir=eval_dir,
        run_id=run_id,
        use_best=args.best,
    )

    print(f"Loading model from {model_path} (run-id={run_id})")
    model = DQN.load(model_path)
    env = gym.make("Snake-v0", render_mode="human", render_fps=args.render_fps)

    for episode in range(1, args.num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info: dict | None = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            last_info = info
            env.render()

        score = last_info.get("score") if last_info else None
        length = last_info.get("snake_length") if last_info else None
        print(
            f"Episode {episode:02d} | reward={total_reward:.2f} | "
            f"steps={steps} | score={score} | length={length}"
        )

    env.close()


if __name__ == "__main__":
    main()
