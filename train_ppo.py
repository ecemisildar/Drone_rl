"""Train PPO in the drone simulator and export a learned SCT YAML."""

from __future__ import annotations

import argparse
import sys
import types

matplotlib = types.ModuleType("matplotlib")
matplotlib_figure = types.ModuleType("matplotlib.figure")
matplotlib_figure.Figure = type("Figure", (), {})
matplotlib.figure = matplotlib_figure
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.figure", matplotlib_figure)

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from drone_env import DroneSCTEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--model-out", default="ppo_drone")
    parser.add_argument("--sct-out", default="learned_supervisor.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DroneSCTEnv(learned_sct_filename=args.sct_out, seed=args.seed)
    if args.check_env:
        check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_out)

    env.export_learned_sct(args.sct_out)
    env.close()
    print(f"saved model: {args.model_out}")
    print(f"saved learned SCT YAML: {args.sct_out}")


if __name__ == "__main__":
    main()
