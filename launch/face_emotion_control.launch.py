# SPDX-License-Identifier: Apache-2.0
"""Launch facial emotion camera recognition and bounded TurtleBot 4 controller."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robot_face_emotion_control')
    config_file = os.path.join(pkg_share, 'config', 'emotion_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_id',
            default_value='0',
            description='Webcam device ID (e.g. 0 for /dev/video0, 1 for external USB camera)',
        ),
        DeclareLaunchArgument(
            'show_window',
            default_value='true',
            description='Whether to display the real-time OpenCV HUD camera window',
        ),
        DeclareLaunchArgument(
            'cmd_vel_topic',
            default_value='/cmd_vel',
            description='Target velocity command topic for TurtleBot 4',
        ),
        Node(
            package='robot_face_emotion_control',
            executable='face_emotion_node',
            name='face_emotion_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'camera_id': LaunchConfiguration('camera_id'),
                    'show_window': LaunchConfiguration('show_window'),
                },
            ],
        ),
        Node(
            package='robot_face_emotion_control',
            executable='emotion_behavior_node',
            name='emotion_behavior_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'cmd_vel_topic': LaunchConfiguration('cmd_vel_topic'),
                },
            ],
        ),
    ])
