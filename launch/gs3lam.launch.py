from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = LaunchConfiguration("config_path")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value="configs/ros2.toml"),
            Node(
                package="anima_slam_gs3lam",
                executable="gs3lam_ros2",
                name="gs3lam",
                output="screen",
                parameters=[{"config_path": config_path}],
            ),
        ]
    )
