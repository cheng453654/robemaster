#!/usr/bin/env python3
"""启动 D435 摄像头 + YOLO 检测 + FoundationPose 三个节点。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("learn_camera_d435")
    default_params = os.path.join(pkg_share, "config", "params.yaml")

    params_file = LaunchConfiguration("params_file")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="三个节点的参数文件"),
        DeclareLaunchArgument(
            "use_viewer",
            default_value="true",
            description="是否打开 OpenCV 显示窗口（无显示器时设为 false）"),
        Node(
            package="learn_camera_d435",
            executable="d435_camera_node.py",
            name="d435_camera",
            parameters=[params_file],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="learn_camera_d435",
            executable="yolo_detect_node.py",
            name="yolo_detector",
            parameters=[params_file],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="learn_camera_d435",
            executable="foundationpose_node.py",
            name="foundationpose",
            parameters=[params_file],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="learn_camera_d435",
            executable="image_view_node.py",
            name="image_viewer",
            parameters=[params_file],
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_viewer")),
        ),
    ])
