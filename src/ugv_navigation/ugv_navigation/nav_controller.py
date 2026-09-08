"""
ROS 2 node driving the UGV to a goal while avoiding obstacles.

Subscribes to /odom and /scan, and publishes twist commands on /cmd_vel.
Optionally uses /perception/obstacles as a secondary obstacle cue.
"""

import math
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from vision_msgs.msg import Detection2DArray
import rclpy
from rclpy.node import Node

from ugv_navigation.nav_logic import (
    NavController,
    quaternion_to_rpy,
    scan_angle_at,
    yaw_from_quaternion,
)


class NavControllerNode(Node):

    def __init__(self):
        super().__init__('nav_controller_node')

        self.declare_parameter('goal_x', 8.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.35)
        self.declare_parameter('max_speed', 0.35)
        self.declare_parameter('max_turn', 0.5)
        self.declare_parameter('detection_range', 1.0)
        self.declare_parameter('use_camera_bias', False)
        self.declare_parameter('startup_settle_seconds', 5.0)
        self.declare_parameter('topple_guard', True)

        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value

        self.controller = NavController(
            goal=(goal_x, goal_y),
            goal_tolerance=self.get_parameter('goal_tolerance').value,
            max_speed=self.get_parameter('max_speed').value,
            max_turn=self.get_parameter('max_turn').value,
            detection_range=self.get_parameter('detection_range').value,
        )

        self.position = None
        self.yaw = None
        self.ranges = None
        self.angles = None
        self.camera_blocked = False
        self.scan_errors = 0
        self.roll = 0.0
        self.pitch = 0.0
        self.started_at = time.time()
        self.halted = False
        self._topple_ticks = 0
        self._settle_logged = False

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        if self.get_parameter('use_camera_bias').value:
            self.create_subscription(
                Detection2DArray, '/perception/obstacles',
                self.perception_callback, 10)

        self.create_timer(1.0 / 20.0, self.control_loop)
        self.get_logger().info(
            f'UGV Navigation Controller started, '
            f'goal=({goal_x:.1f}, {goal_y:.1f})')

    def odom_callback(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def imu_callback(self, msg):
        q = msg.orientation
        roll, pitch, _ = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        self.roll = roll
        self.pitch = pitch

    def scan_callback(self, msg):
        expected = int(
            round((msg.angle_max - msg.angle_min) / msg.angle_increment)) + 1
        if len(msg.ranges) != expected:
            self.get_logger().warning(
                f'Dropping malformed scan ({len(msg.ranges)} != '
                f'{expected} ranges)')
            self.scan_errors += 1
            return
        self.angles = [
            scan_angle_at(msg.angle_min, msg.angle_increment, index)
            for index in range(len(msg.ranges))
        ]
        self.ranges = list(msg.ranges)

    def perception_callback(self, msg):
        self.camera_blocked = False
        for detection in msg.detections:
            center_x = detection.bbox.center.position.x
            size_x = detection.bbox.size_x
            if center_x > 200.0 and center_x < 440.0 and size_x > 120.0:
                self.camera_blocked = True
                break

    def _merge_scan_state(self):
        """Merge camera bias into scan state as a forward 'virtual' return."""
        if self.ranges is None or self.angles is None:
            return None
        if self.camera_blocked and self.yaw is not None:
            ranges = list(self.ranges)
            angles = list(self.angles)
            closest_index = None
            best_diff = float('inf')
            for index, angle in enumerate(angles):
                diff = abs(math.atan2(
                    math.sin(angle - self.yaw), math.cos(angle - self.yaw)))
                if diff < best_diff:
                    best_diff = diff
                    closest_index = index
            if closest_index is not None:
                ranges[closest_index] = min(ranges[closest_index], 0.6)
                return ranges, angles
        return self.ranges, self.angles

    def _publish_zero(self):
        cmd = Twist()
        self.cmd_publisher.publish(cmd)

    def _toppled(self):
        limit = math.radians(50.0)
        return abs(self.roll) > limit or abs(self.pitch) > limit

    def control_loop(self):
        settled = (time.time() - self.started_at
                   >= self.get_parameter('startup_settle_seconds').value)
        if not settled:
            if not self._settle_logged:
                self.get_logger().info(
                    'Waiting for the model to settle before driving')
                self._settle_logged = True
            self._publish_zero()
            return

        if self.get_parameter('topple_guard').value and self._toppled():
            self._topple_ticks += 1
            if self._topple_ticks > 20:
                if not self.halted:
                    self.get_logger().error(
                        f'UGV toppled (roll={self.roll:.2f}, '
                        f'pitch={self.pitch:.2f}); halting')
                    self.halted = True
                self._publish_zero()
                return
        else:
            self._topple_ticks = 0

        if self.halted:
            self._publish_zero()
            return

        if self.position is None or self.yaw is None:
            return
        merged = self._merge_scan_state()
        if merged is None:
            return
        ranges, angles = merged

        linear, angular, mode = self.controller.step(
            self.position, self.yaw, ranges, angles)

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_publisher.publish(cmd)

        if mode == NavController.DONE:
            self.get_logger().info(
                f'Goal reached at ({self.position[0]:.2f}, '
                f'{self.position[1]:.2f})')

    def reset_for_retry(self):
        self.controller.reset()


def main(args=None):
    rclpy.init(args=args)
    node = NavControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
