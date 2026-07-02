"""Replay a trained PPO policy as Webots Crazyflie velocity commands."""

from __future__ import annotations

import argparse
import sys
import time
import types

matplotlib = types.ModuleType("matplotlib")
matplotlib_figure = types.ModuleType("matplotlib.figure")
matplotlib_figure.Figure = type("Figure", (), {})
matplotlib.figure = matplotlib_figure
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.figure", matplotlib_figure)

import rclpy
from geometry_msgs.msg import Twist
from stable_baselines3 import PPO

from drone_env import DroneSCTEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_drone.zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--speed", type=float, default=0.35)
    parser.add_argument("--z-speed", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def action_to_twist(event_name: str, speed: float, z_speed: float) -> Twist:
    msg = Twist()
    if event_name == "forward":
        msg.linear.x = speed
    elif event_name == "backward":
        msg.linear.x = -speed
    elif event_name == "left":
        msg.linear.y = speed
    elif event_name == "right":
        msg.linear.y = -speed
    elif event_name in {"up", "takeoff"}:
        msg.linear.z = z_speed
    elif event_name in {"down", "land"}:
        msg.linear.z = -z_speed
    return msg


def main() -> None:
    args = parse_args()

    rclpy.init()
    node = rclpy.create_node("ppo_webots_replay")
    pub = node.create_publisher(Twist, "cmd_vel", 10)

    env = DroneSCTEnv(seed=args.seed, max_steps=args.max_steps)
    model = PPO.load(args.model, env=env)
    period = 1.0 / args.rate

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            done = False
            steps = 0

            while rclpy.ok() and not done and steps < args.max_steps:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                event_name = env.event_names[int(action)]
                obs, reward, terminated, truncated, info = env.step(int(action))

                pub.publish(action_to_twist(event_name, args.speed, args.z_speed))
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(period)

                done = terminated or truncated
                steps += 1

            pub.publish(Twist())
            print(
                f"episode={episode + 1} steps={steps} "
                f"distance={info['distance']:.2f} last_event={info['event']}"
            )
    finally:
        pub.publish(Twist())
        env.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
