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


def _find_sensor(root, reference, sensor_name):
    gazebo = root.find(f"./gazebo[@reference='{reference}']")
    assert gazebo is not None, f"No <gazebo> block for link '{reference}'"
    sensor = gazebo.find(f"./sensor[@name='{sensor_name}']")
    assert sensor is not None, f"Sensor '{sensor_name}' not found on '{reference}'"
    return sensor


def test_camera_sensor_configuration(urdf_tree):
    """Confirm the camera sensor is configured on camera_link pointing forward."""
    root, _ = urdf_tree
    sensor = _find_sensor(root, 'camera_link', 'camera_sensor')

    assert sensor.get('type') == 'camera'
    topic = sensor.find('topic')
    assert topic is not None and topic.text == 'camera'

    camera = sensor.find('camera')
    assert camera is not None
    optical_frame = camera.find('optical_frame_id')
    assert optical_frame is not None
    assert optical_frame.text == 'camera_optical_frame'

    assert camera.find('image/width').text == '640'
    assert camera.find('image/height').text == '480'

    update_rate = sensor.find('update_rate')
    assert update_rate is not None
    assert float(update_rate.text) >= 10


def test_lidar_sensor_configuration(urdf_tree):
    """Confirm the GPU LiDAR sensor is configured on lidar_link."""
    root, _ = urdf_tree
    sensor = _find_sensor(root, 'lidar_link', 'gpu_lidar_sensor')

    assert sensor.get('type') == 'gpu_lidar'
    topic = sensor.find('topic')
    assert topic is not None and topic.text == 'scan'

    samples = sensor.find('lidar/scan/horizontal/samples')
    assert samples is not None
    assert int(samples.text) == 360

    max_range = sensor.find('lidar/range/max')
    assert max_range is not None
    assert float(max_range.text) >= 20.0

    frame_id = sensor.find('ignition_frame_id')
    assert frame_id is not None and frame_id.text == 'lidar_link'


def test_imu_sensor_configuration(urdf_tree):
    """Confirm the IMU sensor is configured on imu_link."""
    root, _ = urdf_tree
    sensor = _find_sensor(root, 'imu_link', 'imu_sensor')

    assert sensor.get('type') == 'imu'
    topic = sensor.find('topic')
    assert topic is not None and topic.text == 'imu'
    frame_id = sensor.find('ignition_frame_id')
    assert frame_id is not None and frame_id.text == 'imu_link'


def test_ugv_world_has_blocker_wall():
    """Confirm the demo world includes the blocker wall for Phase 3E."""
    tree = ET.parse(WORLD_PATH)
    world = tree.getroot().find('world')
    assert world is not None

    wall = world.find(".//model[@name='blocker_wall']")
    assert wall is not None, 'blocker_wall model not found in ugv_world.sdf'

    static = wall.find('static')
    assert static is not None and static.text == 'true'

    pose = wall.find('pose')
    assert pose is not None
    pose_values = [float(val) for val in pose.text.split()]
    assert pose_values[0] == 4.0, 'Blocker wall should sit at x=4 m'

    box = wall.find('.//collision/geometry/box')
    assert box is not None
    size = [float(val) for val in box.find('size').text.split()]
    assert size[1] >= 2.0, 'Blocker wall should span at least 2 m laterally'


def test_ugv_world_has_imu_system_plugin():
    """Confirm the world loads the gz::sim::systems::Imu plugin."""
    tree = ET.parse(WORLD_PATH)
    world = tree.getroot().find('world')
    assert world is not None

    plugin = world.find(".//plugin[@name='gz::sim::systems::Imu']")
    assert plugin is not None, 'gz::sim::systems::Imu plugin not found'
    assert plugin.get('filename') == 'ignition-gazebo-imu-system'


def test_ugv_world_sensors_use_ogre2():
    """
    Confirm the Sensors system uses the ogre2 render engine.

    gpu_lidar GPU rays are not supported by the legacy ogre engine and
    degenerate into range_min readings everywhere.
    """
    tree = ET.parse(WORLD_PATH)
    world = tree.getroot().find('world')
    assert world is not None

    plugin = world.find(".//plugin[@name='gz::sim::systems::Sensors']")
    assert plugin is not None
    engine = plugin.find('render_engine')
    assert engine is not None and engine.text == 'ogre2'


def test_ugv_world_has_robot_model_with_sensors():
    """
    Confirm the world declares the full UGV model with sensors.

    Sensors must be loaded with the world file so the gpu_lidar rendering
    rays initialise correctly (runtime-spawned sensors fail).
    """
    tree = ET.parse(WORLD_PATH)
    world = tree.getroot().find('world')
    assert world is not None

    ugv = world.find(".//model[@name='ugv']")
    assert ugv is not None, 'ugv robot model not found in ugv_world.sdf'

    for wheel in ['front_left_wheel', 'front_right_wheel',
                  'rear_left_wheel', 'rear_right_wheel']:
        assert ugv.find(f"link[@name='{wheel}']") is not None
    assert ugv.find("joint[@name='front_right_wheel_joint']") is not None

    lidar = ugv.find(".//sensor[@name='gpu_lidar_sensor']")
    assert lidar is not None
    assert lidar.find('ignition_frame_id').text == 'lidar_link'
    lidar_pose = [float(v) for v in lidar.find('pose').text.split()]
    assert lidar_pose[2] == 0.375, 'World lidar must clear the chassis top'

    camera = ugv.find(".//sensor[@name='camera_sensor']")
    assert camera is not None
    assert camera.find('camera/optical_frame_id').text == 'camera_optical_frame'

    assert ugv.find(".//sensor[@name='imu_sensor']") is not None
    assert ugv.find(".//plugin[@name='gz::sim::systems::DiffDrive']") is not None


def test_urdf_to_sdf_sensor_placement():
    """
    Confirm the URDF->SDF conversion keeps explicit sensor frames and poses.

    The URDF importer lumps fixed-joint links and would otherwise drop sensor
    poses and scope their frame ids, breaking /scan and /imu in simulation.
    """
    urdf_xml = xacro.process_file(XACRO_PATH).toxml()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=True) as tmp:
        tmp.write(urdf_xml)
        tmp.flush()
        result = subprocess.run(
            ['ign', 'sdf', '-p', tmp.name],
            capture_output=True, text=True,
        )
    assert result.returncode == 0, result.stderr
    model = ET.fromstring(result.stdout)
    sdf_root = model if model.tag == 'model' else model.find('.//model')
    assert sdf_root is not None

    lidar = sdf_root.find(".//sensor[@name='gpu_lidar_sensor']")
    assert lidar is not None
    assert lidar.find('ignition_frame_id').text == 'lidar_link'
    lidar_pose = [float(v) for v in lidar.find('pose').text.split()]
    assert lidar_pose[2] == 0.375, 'Lidar sensor must clear the chassis top'

    camera = sdf_root.find(".//sensor[@name='camera_sensor']")
    assert camera is not None
    assert camera.find('camera/optical_frame_id').text == 'camera_optical_frame'

    imu = sdf_root.find(".//sensor[@name='imu_sensor']")
    assert imu is not None
    assert imu.find('ignition_frame_id').text == 'imu_link'

    plugin = sdf_root.find(".//plugin[@name='gz::sim::systems::DiffDrive']")
    assert plugin is not None, 'DiffDrive plugin must survive the conversion'
