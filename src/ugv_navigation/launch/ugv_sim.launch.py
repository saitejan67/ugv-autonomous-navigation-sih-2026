"""Launch the UGV platform in Gazebo Fortress simulation with robot_state_publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def launch_setup(context, *args, **kwargs):
    package_share = get_package_share_directory('ugv_navigation')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_path = os.path.join(package_share, 'worlds', 'ugv_world.sdf')
    xacro_path = os.path.join(package_share, 'urdf', 'ugv.urdf.xacro')

    headless = LaunchConfiguration('headless').perform(context).lower() in ['true', '1']
    gz_args = f'-s -r {world_path}' if headless else f'-r {world_path}'

    # Process Xacro to URDF string
    robot_description_raw = xacro.process_file(xacro_path).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': gz_args}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_raw,
            'use_sim_time': True,
        }],
    )

    # Spawn the UGV entity in Gazebo using the robot_description string
    spawn_ugv = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-world', 'ugv_world',
            '-string', robot_description_raw,
            '-name', 'ugv',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.15',
        ],
    )

    # Bridge simulation clock to ROS 2
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )

    return [gazebo, robot_state_publisher, spawn_ugv, clock_bridge]


def generate_launch_description():
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Whether to run Gazebo in headless mode (no GUI)',
    )

    return LaunchDescription([
        headless_arg,
        OpaqueFunction(function=launch_setup),
    ])
