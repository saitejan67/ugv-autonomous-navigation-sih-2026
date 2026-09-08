"""Automated tests for the Phase 3E reactive navigation controller logic."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ugv_navigation.nav_logic import (  # noqa: E402
    NavController,
    clearest_heading,
    forward_blocked,
    heading_to_goal,
    normalize_angle,
    quaternion_to_rpy,
    yaw_from_quaternion,
)


def full_scan(ranges, angle_min=-math.pi):
    """Return parallel ranges/angles lists like a 360-sample 2D scan."""
    n = len(ranges)
    angles = [angle_min + i * (2.0 * math.pi / n) for i in range(n)]
    return ranges, angles


def test_normalize_angle_wraps():
    assert normalize_angle(3.5) == pytest.approx(3.5 - 2.0 * math.pi)
    assert normalize_angle(-3.5) == pytest.approx(-3.5 + 2.0 * math.pi)


def test_quaternion_to_rpy_upright_yaw_only():
    for yaw in (0.0, 1.2, -1.2):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        roll, pitch, got = quaternion_to_rpy(0.0, 0.0, qz, qw)
        assert abs(roll) < 1e-6
        assert abs(pitch) < 1e-6
        assert got == pytest.approx(yaw)


def test_quaternion_to_rpy_flags_toppled_roll():
    qx = 0.39
    qy = 0.92
    roll, pitch, _ = quaternion_to_rpy(qx, qy, 0.0, 0.0)
    assert abs(roll) > math.radians(50.0) or abs(pitch) > math.radians(50.0)
    assert normalize_angle(0.0) == 0.0


def test_yaw_from_quaternion_identity():
    yaw = yaw_from_quaternion(0.0, 0.0, 0.0, 1.0)
    assert yaw == pytest.approx(0.0)


def test_yaw_from_quaternion_right_turn():
    # 90 degrees around Z
    yaw = yaw_from_quaternion(0.0, 0.0, math.sin(math.pi / 4.0),
                              math.cos(math.pi / 4.0))
    assert yaw == pytest.approx(math.pi / 2.0, abs=1e-9)


def test_heading_to_goal():
    heading, distance = heading_to_goal((0.0, 0.0), (8.0, 0.0))
    assert distance == pytest.approx(8.0)
    assert heading == pytest.approx(0.0)


def test_forward_blocked_detects_close_obstacle():
    ranges, angles = full_scan([float('inf')] * 360)
    ranges[180] = 0.5  # straight ahead (bearing 0 at yaw=0)
    blocked, min_range, _ = forward_blocked(
        ranges, angles, 0.0, detection_range=1.0, cone_rad=math.radians(45))
    assert blocked
    assert min_range == pytest.approx(0.5)


def test_forward_blocked_ignores_sides():
    ranges, angles = full_scan([float('inf')] * 360)
    ranges[90] = 0.5  # left side only
    blocked, _, _ = forward_blocked(
        ranges, angles, 0.0, detection_range=1.0, cone_rad=math.radians(45))
    assert not blocked


def test_clearest_heading_clears_forward_wall():
    ranges, angles = full_scan([5.0] * 360)
    for i in range(150, 211):
        ranges[i] = 0.4  # wall straight ahead
    best, best_range = clearest_heading(
        ranges, angles, 0.0, math.radians(80.0), math.radians(12.0))
    assert best_range > 4.0
    assert abs(best) > math.radians(40.0)


def test_controller_seeks_goal_when_clear():
    ctrl = NavController(goal=(8.0, 0.0))
    ranges, angles = full_scan([float('inf')] * 360)
    linear, angular, mode = ctrl.step((0.0, 0.0), 0.0, ranges, angles)
    assert mode == NavController.SEEK
    assert linear > 0.0
    assert angular == pytest.approx(0.0, abs=1e-3)


def test_controller_enters_avoid_when_blocked():
    ctrl = NavController(goal=(8.0, 0.0))
    ranges, angles = full_scan([float('inf')] * 360)
    for i in range(150, 211):
        ranges[i] = 0.4
    linear, angular, mode = ctrl.step((0.0, 0.0), 0.0, ranges, angles)
    assert ctrl.mode == NavController.AVOID


def test_controller_done_at_goal():
    ctrl = NavController(goal=(8.0, 0.0), goal_tolerance=0.35)
    ranges, angles = full_scan([float('inf')] * 360)
    linear, angular, mode = ctrl.step((7.9, 0.0), 0.0, ranges, angles)
    assert mode == NavController.DONE
    assert linear == 0.0 and angular == 0.0


def test_controller_exits_avoid_once_clear():
    ctrl = NavController(goal=(8.0, 0.0))
    ranges, angles = full_scan([float('inf')] * 360)
    for i in range(150, 211):
        ranges[i] = 0.4
    ctrl.step((0.0, 0.0), 0.0, ranges, angles)
    assert ctrl.mode == NavController.AVOID

    for _ in range(30):  # simulate turning away and travelling clear
        ranges, angles = full_scan([float('inf')] * 360)
        ctrl.step((2.0, 0.0), 0.8, ranges, angles)
    assert ctrl.mode == NavController.SEEK


def test_controller_avoid_keeps_single_detour_direction():
    ctrl = NavController(goal=(8.0, 0.0))
    ranges, angles = full_scan([float('inf')] * 360)
    for i in range(150, 211):
        ranges[i] = 0.4
    ctrl.step((0.0, 0.0), 0.0, ranges, angles)
    assert ctrl.mode == NavController.AVOID
    assert ctrl.avoid_rel != 0.0

    side = ctrl.avoid_rel
    prev = ctrl.avoid_target_yaw
    yaw = 0.0
    for _ in range(200):  # closed loop: yaw follows the commanded angular
        _, angular, _ = ctrl.step((0.5, 0.0), yaw, ranges, angles)
        yaw = normalize_angle(yaw + angular * 0.05)
        # the committed target never moves against the chosen detour side,
        # i.e. no mid-manoeuvre re-picking back toward the other side
        assert (ctrl.avoid_target_yaw - prev) * side >= -1e-9
        prev = ctrl.avoid_target_yaw
    assert ctrl.mode == NavController.AVOID
    assert ctrl.avoid_target_yaw * side >= 0.0
