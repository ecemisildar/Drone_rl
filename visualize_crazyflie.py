"""Replay a trained PPO policy on a Crazyswarm2 Crazyflie simulation.

Start the Crazyflie sim first, then run this script from a sourced ROS 2 shell.
The PPO policy is evaluated in the Gym environment and each planned position is
sent to the simulated Crazyflie with high-level goTo commands.
"""

from __future__ import annotations

import argparse
import sys
import types

import numpy as np

matplotlib = types.ModuleType("matplotlib")
matplotlib_figure = types.ModuleType("matplotlib.figure")
matplotlib_figure.Figure = type("Figure", (), {})
matplotlib.figure = matplotlib_figure
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.figure", matplotlib_figure)

from crazyflie_py import Crazyswarm
from stable_baselines3 import PPO

from drone_env import DroneSCTEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_drone.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--altitude", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = DroneSCTEnv(seed=args.seed, max_steps=args.max_steps)
    model = PPO.load(args.model, env=env)

    swarm = Crazyswarm()
    time_helper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=args.altitude, duration=2.0)
    time_helper.sleep(2.5)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            done = False
            steps = 0

            while not done and steps < args.max_steps:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(int(action))

                sim_pos = np.asarray(info["position"], dtype=np.float32)
                target = np.array(
                    [args.scale * sim_pos[0], args.scale * sim_pos[1], args.altitude],
                    dtype=np.float32,
                )
                cf.goTo(target, yaw=0.0, duration=args.duration)
                time_helper.sleep(args.duration)

                done = terminated or truncated
                steps += 1

            print(
                f"episode={episode + 1} steps={steps} "
                f"distance={info['distance']:.2f} last_event={info['event']}"
            )
    finally:
        cf.land(targetHeight=0.04, duration=2.0)
        time_helper.sleep(2.5)
        env.close()


if __name__ == "__main__":
    main()
