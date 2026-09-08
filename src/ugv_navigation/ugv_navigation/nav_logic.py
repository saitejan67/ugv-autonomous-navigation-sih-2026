"""
Pure logic for the reactive navigation controller.

The controller is a lightweight, deterministic waypoint follower that reacts
to 2D range data. It does not use a pre-built map or ground-truth pose.
"""

import math


def normalize_angle(angle):
    """Wrap an angle into the interval [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x, y, z, w):
    """Extract the yaw (rotation around Z) of a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_to_rpy(x, y, z, w):
    """Convert a quaternion to (roll, pitch, yaw) in radians."""
    roll = math.atan2(
        2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = yaw_from_quaternion(x, y, z, w)
    return roll, pitch, yaw


def scan_angle_at(angle_min, angle_increment, index):
    """Return the bearing (radians) of a given scan sample index."""
    return angle_min + index * angle_increment


def forward_blocked(ranges, angles, yaw, detection_range, cone_rad):
    """
    Check whether any return within the forward cone is too close.

    Returns (blocked, min_range, min_bearing) where min_bearing is the
    bearing (relative to the robot yaw) of the closest obstructing return.
    """
    min_range = float('inf')
    min_bearing = 0.0
    blocked = False
    for index, (rng, angle) in enumerate(zip(ranges, angles)):
        if not math.isfinite(rng):
            continue
        bearing = normalize_angle(angle - yaw)
        if abs(bearing) > cone_rad:
            continue
        if rng < min_range:
            min_range = rng
            min_bearing = bearing
        if rng < detection_range:
            blocked = True
    return blocked, min_range, min_bearing


def clearest_heading(ranges, angles, yaw, search_rad, window_rad,
                     step=math.radians(5.0)):
    """
    Return the heading (average of bearings) with the greatest minimum range.

    The heading is expressed relative to the *current* robot yaw and is a
    yaw-relative target, so the caller can track it against odometry.
    """
    best_target = None
    best_min = -1.0
    candidate = -search_rad
    while candidate <= search_rad:
        in_cone = []
        for rng, angle in zip(ranges, angles):
            if not math.isfinite(rng):
                continue
            bearing = normalize_angle(angle - yaw)
            if abs(bearing - candidate) <= window_rad:
                in_cone.append(rng)
        cone_min = min(in_cone) if in_cone else float('inf')
        if cone_min > best_min:
            best_min = cone_min
            best_target = candidate
        candidate += step
    return best_target if best_target is not None else 0.0, best_min


def choose_avoid_yaw(ranges, angles, yaw, search_rad, window_rad, desired_yaw):
    """Pick a detour heading, tying ties toward the direction of the goal."""
    plausible = {}
    candidate = -search_rad
    while candidate <= search_rad:
        in_cone = []
        for rng, angle in zip(ranges, angles):
            if not math.isfinite(rng):
                continue
            bearing = normalize_angle(angle - yaw)
            if abs(bearing - candidate) <= window_rad:
                in_cone.append(rng)
        plausible[candidate] = min(in_cone) if in_cone else float('inf')
        candidate += math.radians(5.0)

    best = max(plausible, key=plausible.get)
    tied = [h for h, r in plausible.items()
            if r == plausible[best] and math.isfinite(plausible[best])]
    if len(tied) > 1 and math.isfinite(plausible[best]):
        goal_bearing = normalize_angle(desired_yaw - yaw)
        return min(tied, key=lambda h: abs(h - goal_bearing))
    return best


def heading_to_goal(position, goal):
    """Desired world-frame yaw pointing from position to goal."""
    dx = goal[0] - position[0]
    dy = goal[1] - position[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        return None, 0.0
    return math.atan2(dy, dx), distance


class NavController:
    """State machine driving a two-wheeled robot toward a goal."""

    SEEK = 'seek'
    AVOID = 'avoid'
    DONE = 'done'

    def __init__(self, goal=(8.0, 0.0), goal_tolerance=0.35,
                 max_speed=0.35, max_turn=0.5,
                 detection_range=1.0, cone_rad=math.radians(45.0),
                 search_rad=math.radians(80.0), window_rad=math.radians(12.0),
                 avoid_exit_range=2.0, clear_ticks=10,
                 heading_gain=2.0, avoid_heading_gain=1.6,
                 straight_gate=math.radians(30.0)):
        self.goal = goal
        self.goal_tolerance = goal_tolerance
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.detection_range = detection_range
        self.cone_rad = cone_rad
        self.search_rad = search_rad
        self.window_rad = window_rad
        self.avoid_exit_range = avoid_exit_range
        self.clear_ticks = clear_ticks
        self.heading_gain = heading_gain
        self.avoid_heading_gain = avoid_heading_gain
        self.straight_gate = straight_gate

        self.mode = self.SEEK
        self.avoid_target_yaw = 0.0
        self.avoid_rel = 0.0
        self.clear_ticks_count = 0
        self.position = (0.0, 0.0)
        self.yaw = 0.0

    def reset(self):
        self.mode = self.SEEK
        self.clear_ticks_count = 0
        self.avoid_target_yaw = 0.0
        self.avoid_rel = 0.0

    def step(self, position, yaw, ranges, angles):
        """
        Compute a (linear, angular) velocity command for one control tick.

        `position` is (x, y), `yaw` the robot heading, and ranges/angles are
        the raw scan samples (any ordering).
        """
        self.position = tuple(position)
        self.yaw = float(yaw)

        linear = 0.0
        angular = 0.0

        if self.mode == self.DONE:
            return 0.0, 0.0, self.DONE

        desired_yaw, distance = heading_to_goal(self.position, self.goal)
        if distance <= self.goal_tolerance:
            self.mode = self.DONE
            return 0.0, 0.0, self.DONE

        blocked, min_range, min_bearing = forward_blocked(
            ranges, angles, yaw, self.detection_range, self.cone_rad)

        if self.mode == self.SEEK:
            if blocked:
                self.mode = self.AVOID
                self.clear_ticks_count = 0
                rel = choose_avoid_yaw(
                    ranges, angles, yaw, self.search_rad, self.window_rad,
                    desired_yaw)
                self.avoid_rel = rel
                self.avoid_target_yaw = yaw + rel
            else:
                steer_err = normalize_angle(desired_yaw - yaw)
                angular = max(-self.max_turn, min(self.max_turn,
                              self.heading_gain * steer_err))
                linear = self.max_speed * max(0.0, math.cos(steer_err))
                return linear, angular, self.mode

        if self.mode == self.AVOID:
            if min_range >= self.avoid_exit_range or not blocked:
                self.clear_ticks_count += 1
                if self.clear_ticks_count >= self.clear_ticks:
                    self.mode = self.SEEK
                    return linear, angular, self.mode
            else:
                self.clear_ticks_count = 0

            target_blocked, _, _ = forward_blocked(
                ranges, angles, self.avoid_target_yaw,
                self.detection_range, self.window_rad)
            if (target_blocked and abs(normalize_angle(
                    self.avoid_target_yaw - yaw))
                    <= math.radians(100.0)):
                # Commit to the chosen detour side: rotate only further in the
                # same direction instead of re-picking, which fishtailed at a
                # wide obstacle after each refresh re-chose the other side.
                self.avoid_target_yaw = normalize_angle(
                    self.avoid_target_yaw + math.copysign(
                        math.radians(10.0), self.avoid_rel))

            steer_err = normalize_angle(self.avoid_target_yaw - yaw)
            angular = max(-self.max_turn, min(self.max_turn,
                          self.avoid_heading_gain * steer_err))
            if abs(steer_err) < self.straight_gate:
                linear = min(self.max_speed, 0.75) * 0.9
            return linear, angular, self.mode

        return linear, angular, self.mode
