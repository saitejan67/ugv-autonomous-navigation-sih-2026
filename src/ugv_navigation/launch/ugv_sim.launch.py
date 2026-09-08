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
    sensors_bridge_config = os.path.join(package_share, 'config', 'sensors_bridge.yaml')

    headless = LaunchConfiguration('headless').perform(context).lower() in ['true', '1']
    gz_args = f'-s -r {world_path}' if headless else f'-r {world_path}'

    # Process Xacro to URDF string (used for the TF tree only; the simulated
    # robot model and its sensors are declared directly in the world SDF).
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

    # Bridge simulation clock, motion commands, and wheel-based odometry between ROS 2 and Gazebo
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
        ],
        output='screen',
    )

    # Bridge camera, LiDAR and IMU sensor data from Gazebo to ROS 2
    sensors_bridge = Node(
        package='ros_gz_bridge',
        executable='bridge_node',
        parameters=[{'config_file': sensors_bridge_config}],
        output='screen',
    )

    # Camera-based obstacle perception (feeds the navigation controller bias)
    perception = Node(
        package='ugv_navigation',
        executable='perception_node',
        output='screen',
    )

    # Reactive navigation: odometry + LiDAR -> /cmd_vel to reach the goal
    nav_controller = Node(
        package='ugv_navigation',
        executable='nav_controller_node',
        output='screen',
        parameters=[{
            'goal_x': 8.0,
            'goal_y': 0.0,
        }],
    )

    return [gazebo, robot_state_publisher, bridge, sensors_bridge,
            perception, nav_controller]


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
