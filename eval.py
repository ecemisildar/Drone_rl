"""Evaluate a trained PPO drone policy in the Gym simulator."""

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

from drone_env import DroneSCTEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_drone.zip")
    parser.add_argument("--sct", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DroneSCTEnv(sct_filename=args.sct, seed=args.seed) if args.sct else DroneSCTEnv(seed=args.seed)
    model = PPO.load(args.model, env=env)

    returns = []
    successes = 0

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if args.render:
                env.render()

        reached = info["distance"] <= env.target_radius
        successes += int(reached)
        returns.append(total_reward)
        print(
            f"episode={episode + 1} "
            f"return={total_reward:.2f} "
            f"steps={steps} "
            f"distance={info['distance']:.2f} "
            f"success={reached} "
            f"last_event={info['event']}"
        )

    avg_return = sum(returns) / max(len(returns), 1)
    print(f"avg_return={avg_return:.2f} success_rate={successes / args.episodes:.2%}")
    env.close()


if __name__ == "__main__":
    main()
