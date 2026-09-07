"""Start the repository-owned Gazebo Fortress camera simulation and perception node."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('ugv_navigation')
    world_path = os.path.join(package_share, 'worlds', 'camera_world.sdf')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    image_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/camera@sensor_msgs/msg/Image@gz.msgs.Image'],
        output='screen',
    )

    perception = Node(
        package='ugv_navigation',
        executable='perception_node',
        output='screen',
    )

    return LaunchDescription([gazebo, image_bridge, perception])
