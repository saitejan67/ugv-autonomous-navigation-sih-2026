"""Automated tests for UGV Xacro/URDF model and Gazebo world definition."""

import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import pytest
import xacro

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
XACRO_PATH = os.path.join(PKG_DIR, 'urdf', 'ugv.urdf.xacro')
WORLD_PATH = os.path.join(PKG_DIR, 'worlds', 'ugv_world.sdf')


@pytest.fixture(scope='module')
def urdf_tree():
    """Load and parse the compiled URDF from Xacro."""
    urdf_xml = xacro.process_file(XACRO_PATH).toxml()
    return ET.fromstring(urdf_xml), urdf_xml


def test_required_links_exist(urdf_tree):
    """Confirm all essential links exist in the robot model."""
    root, _ = urdf_tree
    link_names = {link.get('name') for link in root.findall('link')}

    expected_links = {
        'base_footprint',
        'base_link',
        'front_left_wheel',
        'front_right_wheel',
        'rear_left_wheel',
        'rear_right_wheel',
        'camera_link',
        'camera_optical_frame',
        'lidar_link',
        'imu_link',
    }
    assert expected_links.issubset(link_names), f"Missing links: {expected_links - link_names}"


def test_required_wheel_joints_exist(urdf_tree):
    """Confirm all 4 wheel continuous joints are configured with pitch axis."""
    root, _ = urdf_tree
    wheel_joints = {
        'front_left_wheel_joint': 'front_left_wheel',
        'front_right_wheel_joint': 'front_right_wheel',
        'rear_left_wheel_joint': 'rear_left_wheel',
        'rear_right_wheel_joint': 'rear_right_wheel',
    }

    for joint_name, child_link in wheel_joints.items():
        joint = root.find(f"./joint[@name='{joint_name}']")
        assert joint is not None, f"Joint {joint_name} not found"
        assert joint.get('type') == 'continuous'
        assert joint.find('parent').get('link') == 'base_link'
        assert joint.find('child').get('link') == child_link
        axis = joint.find('axis')
        assert axis is not None
        assert axis.get('xyz') == '0 1 0'


def test_robot_mass_and_dimensions(urdf_tree):
    """Verify the mass and dimensions meet Phase 3A specifications."""
    root, _ = urdf_tree

    # Total mass calculation across links with inertial definitions
    total_mass = 0.0
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is not None:
            mass = inertial.find('mass')
            if mass is not None:
                total_mass += float(mass.get('value'))

    # Expect ~15 kg chassis + 4 * 1 kg wheels = ~19 kg
    assert 15.0 <= total_mass <= 20.0, f"Total mass {total_mass} kg not in 15-20 kg range"

    # Chassis box dimensions
    base_link = root.find("./link[@name='base_link']")
    assert base_link is not None
    box = base_link.find(".//visual/geometry/box")
    assert box is not None
    size = [float(val) for val in box.get('size').split()]
    assert size == [0.60, 0.40, 0.20]


def test_gazebo_sdf_validity(urdf_tree):
    """Confirm ign sdf validates the converted URDF model."""
    _, urdf_xml = urdf_tree
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=True) as tmp:
        tmp.write(urdf_xml)
        tmp.flush()
        result = subprocess.run(
            ['ign', 'sdf', '-k', tmp.name],
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0
    assert 'Valid.' in result.stdout


def test_ugv_world_sdf_validity():
    """Confirm ign sdf validates ugv_world.sdf."""
    result = subprocess.run(
        ['ign', 'sdf', '-k', WORLD_PATH],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'Valid.' in result.stdout


def test_diff_drive_plugin_configuration(urdf_tree):
    """Confirm the DiffDrive plugin is configured with all 4 wheels and /cmd_vel."""
    root, _ = urdf_tree
    plugin = root.find(".//plugin[@name='gz::sim::systems::DiffDrive']")
    assert plugin is not None, 'DiffDrive plugin not found in model'

    left_joints = [j.text for j in plugin.findall('left_joint')]
    right_joints = [j.text for j in plugin.findall('right_joint')]
    assert 'front_left_wheel_joint' in left_joints
    assert 'rear_left_wheel_joint' in left_joints
    assert 'front_right_wheel_joint' in right_joints
    assert 'rear_right_wheel_joint' in right_joints

    wheel_sep = plugin.find('wheel_separation')
    assert wheel_sep is not None
    assert float(wheel_sep.text) == 0.50

    wheel_rad = plugin.find('wheel_radius')
    assert wheel_rad is not None
    assert float(wheel_rad.text) == 0.10

    topic = plugin.find('topic')
    assert topic is not None
    assert topic.text == '/cmd_vel'


def test_diff_drive_odometry_configuration(urdf_tree):
    """Confirm the DiffDrive plugin exposes wheel-based odometry and dynamic TF."""
    root, _ = urdf_tree
    plugin = root.find(".//plugin[@name='gz::sim::systems::DiffDrive']")
    assert plugin is not None, 'DiffDrive plugin not found in model'

    odom_topic = plugin.find('odom_topic')
    assert odom_topic is not None
    assert odom_topic.text == '/odom'

    tf_topic = plugin.find('tf_topic')
    assert tf_topic is not None
    assert tf_topic.text == '/tf'

    frame_id = plugin.find('frame_id')
    assert frame_id is not None
    assert frame_id.text == 'odom'

    child_frame_id = plugin.find('child_frame_id')
    assert child_frame_id is not None
    assert child_frame_id.text == 'base_footprint'
