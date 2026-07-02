"""Launch Webots Crazyflie in a local empty world."""

from __future__ import annotations

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def generate_launch_description():
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    world = os.path.join(workspace_dir, "worlds", "crazyflie_empty.wbt")
    robot_description = os.path.join(
        get_package_share_directory("webots_ros2_crazyflie"),
        "resource",
        "crazyflie_webots.urdf",
    )

    webots = WebotsLauncher(world=world, ros2_supervisor=True)
    crazyflie_driver = WebotsController(
        robot_name="Crazyflie",
        parameters=[{"robot_description": robot_description}],
        respawn=True,
    )

    return LaunchDescription(
        [
            webots,
            webots._supervisor,
            crazyflie_driver,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessExit(
                    target_action=webots,
                    on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
                )
            ),
        ]
    )
