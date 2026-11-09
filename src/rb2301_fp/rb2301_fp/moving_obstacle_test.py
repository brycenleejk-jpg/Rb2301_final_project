"""Test ego-compensated object motion, stopping, zero-heading alignment and strafing.

Run with ROS sourced, for example:
python3 moving_obstacle_test.py --ros-args -p use_sim_time:=true
No forward crossing is performed. See ../MOVING_OBSTACLE_TEST.md.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


# ROS parameters: distances in metres, time in seconds, angles in degrees.
PARAMS = {
    "scan_topic": "scan",
    "pose_topic": "odom",
    "pose_type": "odom",                 # odom or pose_stamped (e.g. OptiTrack).
    "cmd_topic": "cmd_vel",
    "lidar_yaw_deg": 180.0,
    "lidar_x_m": 0.11423,
    "lidar_y_m": 0.0,
    "detect_near_m": 0.20,
    "detect_far_m": 1.50,
    "detect_half_width_m": 1.0,
    "reference_delay_s": 0.20,           # Compare scans this far apart.
    "reference_max_age_s": 0.35,
    "point_change_m": 0.04,              # Minimum distance from old surface hits.
    "free_space_margin_m": 0.04,         # New hit must be inside previously seen free space.
    "min_static_points": 12,             # Require supporting stationary background.
    "min_static_fraction": 0.50,
    "motion_cluster_radius_m": 0.10,      # Connect nearby changed points; no bar shape test.
    "min_motion_points": 3,
    "track_gap_s": 0.25,
    "association_slack_m": 0.06,
    "max_object_speed_mps": 1.5,
    "observation_window_s": 0.40,
    "min_observation_s": 0.30,
    "min_observations": 3,
    "motion_distance_m": 0.06,           # Fixed-frame displacement in any direction.
    "min_object_speed_mps": 0.15,
    "motion_fit_error_m": 0.025,
    "confirm_scans": 2,
    "approach_speed_mps": 0.10,
    "max_approach_m": 1.50,
    "heading_limit_deg": 10.0,
    "stop_ahead_m": 0.20,                 # Additional distance beyond inflated nose.
    "sensor_timeout_s": 0.50,
    "min_coverage_fraction": 0.80,         # Usable rays in the checked +/-25deg sector.
    "pose_sync_tolerance_s": 0.05,
    "stop_hold_s": 0.30,                 # Measured low speed must persist this long.
    "stopped_speed_mps": 0.02,
    "stopped_turn_rps": 0.05,
    "target_heading_deg": 0.0,           # Absolute heading in selected pose frame.
    "heading_tolerance_deg": 2.0,
    "heading_hold_s": 0.20,
    "heading_gain": 1.0,
    "side_speed_mps": 0.10,
    "wall_clearance_m": 0.25,
    "wall_tolerance_m": 0.02,
    "wall_angle_limit_deg": 25.0,        # Wall should roughly follow zero heading.
    "min_turn_rps": 0.15,
    "max_turn_rps": 0.30,
    # Bounds from the repository's collision meshes, relative to base_footprint.
    "body_front_m": 0.182,
    "body_rear_m": 0.034,
    "body_half_width_m": 0.096,
    "safety_margin_m": 0.04,
    "log_interval_s": 0.50,
}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def scan_points(scan, p):
    """Keep angular neighbours together, including NaN breaks for missing rays."""
    ranges = np.asarray(scan.ranges, dtype=float)
    angles = (scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
              + math.radians(p["lidar_yaw_deg"]))
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(angles)
    angles, ranges = angles[order], ranges[order]
    hit = (np.isfinite(ranges) & (ranges >= scan.range_min)
           & (ranges > 0) & (ranges < scan.range_max))
    distances = np.where(hit, ranges, np.nan)
    points = np.column_stack((distances * np.cos(angles) + p["lidar_x_m"],
                              distances * np.sin(angles) + p["lidar_y_m"]))
    # A max-range/+inf reading supplies coverage, but is not an object hit.
    covered = hit | np.isposinf(ranges) | (ranges == scan.range_max)
    return angles, points, covered


def in_start_frame(points, pose, origin):
    """Remove robot translation/rotation, then express points along initial heading."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    world = points @ np.array([[c, s], [-s, c]]) + pose[:2]
    c, s = math.cos(origin[2]), math.sin(origin[2])
    return (world - origin[:2]) @ np.array([[c, -s], [s, c]])


def nearest_distances(query, support):
    """Nearest surface distance, with bounded intermediate array size."""
    if not len(support):
        return np.full(len(query), np.inf)
    distances = []
    for start in range(0, len(query), 128):
        delta = query[start:start + 128, None, :] - support[None, :, :]
        distances.extend(np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1)))
    return np.asarray(distances)


def motion_clusters(points, radius, minimum):
    """Spatial connected components of changed points; no width or line assumption."""
    unused = set(range(len(points)))
    groups = []
    while unused:
        seed = unused.pop()
        group, pending = [seed], [seed]
        while pending:
            current = pending.pop()
            neighbours = [i for i in unused if np.linalg.norm(points[i] - points[current]) <= radius]
            unused.difference_update(neighbours)
            group.extend(neighbours)
            pending.extend(neighbours)
        if len(group) >= minimum:
            groups.append(points[group])
    return groups


@dataclass
class ScanFrame:
    stamp: float
    pose: np.ndarray
    angles: np.ndarray
    points: np.ndarray
    covered: np.ndarray
    range_max: float = math.inf

    def free_before(self, world_points, p):
        """Was each current hit visible free space in this previous scan?

        Behind an old foreground return is UNKNOWN, not free. Check neighbouring
        rays too so one edge ray cannot declare space behind an occluder free.
        """
        if len(self.angles) < 3:
            return np.zeros(len(world_points), dtype=bool)
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        robot = (world_points - self.pose[:2]) @ np.array([[c, -s], [s, c]])
        sensor = robot - [p["lidar_x_m"], p["lidar_y_m"]]
        directions = np.arctan2(sensor[:, 1], sensor[:, 0])
        distances = np.linalg.norm(sensor, axis=1)
        # The scanner may have a partial FOV. Never wrap across an absent sector.
        right = np.searchsorted(self.angles, directions) % len(self.angles)
        left = (right - 1) % len(self.angles)
        dr = np.abs((self.angles[right] - directions + np.pi) % (2*np.pi) - np.pi)
        dl = np.abs((self.angles[left] - directions + np.pi) % (2*np.pi) - np.pi)
        closest = np.where(dl < dr, left, right)
        indices = (closest[:, None] + np.array([-1, 0, 1])) % len(self.angles)
        differences = np.abs((self.angles[indices] - directions[:, None] + np.pi) % (2*np.pi) - np.pi)
        spacing = float(np.median(np.diff(self.angles)))
        visible = (differences <= 2.1 * spacing) & self.covered[indices]
        hit_ranges = np.linalg.norm(self.points - [p["lidar_x_m"], p["lidar_y_m"]], axis=1)
        hit_ranges = np.where(np.isfinite(hit_ranges), hit_ranges, self.range_max)
        return np.all(visible & (hit_ranges[indices] > distances[:, None] + p["free_space_margin_m"]), axis=1)


@dataclass
class Track:
    number: int
    samples: deque = field(default_factory=deque)
    confirmations: int = 0


class MotionDetector:
    """Find sustained motion supported by old free-space evidence, not bar shape."""
    def __init__(self, p):
        self.p = p
        self.tracks = []
        self.next_id = 1
        self.last_stamp = None
        self.frames = deque()

    def update(self, angles, points, covered, pose, stamp, range_max=math.inf):
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return None, "duplicate scan"  # Never count the same scan twice.
        self.last_stamp = stamp
        p = self.p
        self.frames.append(ScanFrame(stamp, pose.copy(), angles.copy(), points.copy(), covered.copy(), range_max))
        while self.frames and stamp - self.frames[0].stamp > p["reference_max_age_s"]:
            self.frames.popleft()
        references = [frame for frame in self.frames
                      if stamp - frame.stamp + 1e-9 >= p["reference_delay_s"]]
        if not references:
            self.tracks.clear()
            return None, "warming up scan reference"
        reference = references[-1]
        usable = np.isfinite(points).all(axis=1)
        current_world = in_start_frame(points[usable], pose, np.zeros(3))
        old_world = in_start_frame(reference.points[np.isfinite(reference.points).all(axis=1)], reference.pose, np.zeros(3))
        residuals = nearest_distances(current_world, old_world)
        stationary = residuals <= p["point_change_m"]
        robot_points = points[usable]
        roi = ((robot_points[:, 0] > p["detect_near_m"])
               & (robot_points[:, 0] < p["detect_far_m"])
               & (np.abs(robot_points[:, 1]) < p["detect_half_width_m"]))
        # Overlapping parts of a moving face can appear unchanged. They must not
        # be used to validate robot pose: require background OUTSIDE the ROI.
        background = ~roi
        if (np.count_nonzero(stationary & background) < p["min_static_points"]
                or np.mean(stationary[background]) < p["min_static_fraction"]):
            self.tracks.clear()
            return None, "motion uncertain: insufficient stationary background / pose mismatch"
        changed = roi & ~stationary
        witnessed_free = reference.free_before(current_world[changed], p)
        evidence = current_world[changed][witnessed_free]
        groups = motion_clusters(evidence, p["motion_cluster_radius_m"], p["min_motion_points"])
        fixed = np.array([group.mean(axis=0) for group in groups]).reshape(-1, 2)
        status = (f"background={np.count_nonzero(stationary & background)}/{np.count_nonzero(background)} "
                  f"changed={np.count_nonzero(changed)} free_before={len(evidence)} groups={len(groups)}")
        detected, messages = self.update_tracks(fixed, stamp)
        return detected, status + (" | " + "; ".join(messages) if messages else "")

    def update_tracks(self, fixed, stamp):
        """Require sustained, coherent fixed-frame displacement of a changed region."""
        self.tracks = [t for t in self.tracks
                       if stamp - t.samples[-1][0] <= self.p["track_gap_s"]]
        pairs = []
        for i, track in enumerate(self.tracks):
            dt = stamp - track.samples[-1][0]
            previous = track.samples[-1][1]
            for j, point in enumerate(fixed):
                delta = point - previous
                if np.linalg.norm(delta) <= self.p["association_slack_m"] + self.p["max_object_speed_mps"] * dt:
                    pairs.append((np.linalg.norm(delta), i, j))
        used_tracks, used_points, observed = set(), set(), []
        # One-to-one matching: a candidate cannot update two tracks.
        for _, i, j in sorted(pairs):
            if i not in used_tracks and j not in used_points:
                observed.append((self.tracks[i], j))
                used_tracks.add(i)
                used_points.add(j)
        for i, track in enumerate(self.tracks):
            if i not in used_tracks:
                track.confirmations = 0
        for j in range(len(fixed)):
            if j not in used_points:
                track = Track(self.next_id)
                self.next_id += 1
                self.tracks.append(track)
                observed.append((track, j))
        messages = []
        for track, j in observed:
            track.samples.append((stamp, fixed[j]))
            while stamp - track.samples[0][0] > self.p["observation_window_s"] + 1e-9:
                track.samples.popleft()
            duration = stamp - track.samples[0][0]
            if len(track.samples) < self.p["min_observations"] or duration + 1e-9 < self.p["min_observation_s"]:
                continue
            positions = np.array([s[1] for s in track.samples])
            times = np.array([s[0] - stamp for s in track.samples])
            velocity, intercept = np.linalg.lstsq(
                np.column_stack((times, np.ones(len(times)))), positions, rcond=None)[0]
            fit_error = float(np.sqrt(np.mean(np.sum((positions - (times[:, None] * velocity + intercept))**2, axis=1))))
            dx, dy = positions[-1] - positions[0]
            speed = np.linalg.norm(velocity)
            moving = (np.hypot(dx, dy) > self.p["motion_distance_m"]
                      and self.p["min_object_speed_mps"] < speed <= self.p["max_object_speed_mps"]
                      and fit_error <= self.p["motion_fit_error_m"])
            track.confirmations = track.confirmations + 1 if moving else 0
            report = (f"track={track.number} "
                      f"dt={duration:.2f}s dx={dx:+.3f}m dy={dy:+.3f}m "
                      f"speed={speed:.2f}m/s fit={fit_error:.3f}m "
                      f"confirm={track.confirmations}/{self.p['confirm_scans']}")
            messages.append(report)
            if track.confirmations >= self.p["confirm_scans"]:
                return report, messages
        return None, messages


class SidePositioner:
    """Retain nearer-wall choice and clearance control; heading is handled by ALIGN."""
    def __init__(self, p):
        self.p, self.side, self.wall_normal = p, 0, 0.0
        self.done = False

    def command(self, angles, points, clear):
        valid = np.isfinite(points).all(axis=1)
        if not self.side:
            distances = []
            for direction in (math.pi / 2, -math.pi / 2):
                mask = valid & (np.abs(angles - direction) < math.radians(5))
                if not np.any(mask):
                    return (0., 0., 0.), "SIDE: side readings unavailable"
                distances.append(np.min(np.linalg.norm(
                    points[mask] - [self.p["lidar_x_m"], self.p["lidar_y_m"]], axis=1)))
            self.side = 1 if distances[0] <= distances[1] else -1
            self.wall_normal = self.side * math.pi / 2
        samples = []
        for offset in (-10, 10):
            direction = self.wall_normal + math.radians(offset)
            difference = (angles - direction + np.pi) % (2 * np.pi) - np.pi
            mask = valid & (np.abs(difference) < math.radians(2))
            if not np.any(mask):
                return (0., 0., 0.), "SIDE: wall samples unavailable"
            samples.append(points[mask].mean(axis=0))
        tangent = samples[1] - samples[0]
        length = np.linalg.norm(tangent)
        if length < 0.03:
            return (0., 0., 0.), "SIDE: wall estimate too short"
        tangent /= length
        if tangent[0] < 0:
            tangent = -tangent
        normal = self.side * np.array([-tangent[1], tangent[0]])
        distance = np.dot((samples[0] + samples[1]) / 2, normal)
        direction = math.atan2(normal[1], normal[0])
        if distance <= 0 or abs(wrap(direction - self.wall_normal)) > math.radians(20):
            return (0., 0., 0.), "SIDE: wall estimate rejected"
        self.wall_normal = direction
        heading = math.atan2(tangent[1], tangent[0])
        if abs(heading) > math.radians(self.p["wall_angle_limit_deg"]):
            return (0., 0., 0.), "SIDE: wall direction inconsistent with target heading"
        error = distance - self.p["wall_clearance_m"]
        if abs(error) < self.p["wall_tolerance_m"]:
            self.done = True
            return (0., 0., 0.), f"POSITIONED: wall distance={distance:.2f}m; test complete"
        sideways = self.side * np.sign(error) * self.p["side_speed_mps"]
        if not clear(np.sign(sideways) * math.pi / 2, 0.05):
            return (0., 0., 0.), "SIDE: sideways path blocked"
        return (0., float(sideways), 0.), f"SIDE: {'left' if self.side == 1 else 'right'} wall; distance={distance:.2f}m"


def pose_at(samples, stamp, tolerance):
    """Interpolate poses to the scan time, allowing only a small endpoint skew."""
    if not samples:
        return None
    for (ta, a), (tb, b) in zip(samples, list(samples)[1:]):
        if ta <= stamp <= tb:
            if abs(stamp-ta) <= 1e-9:
                return a.copy()
            if abs(stamp-tb) <= 1e-9:
                return b.copy()
            if tb - ta > 2 * tolerance + 1e-9:
                return None  # Do not interpolate across a long localization dropout.
            ratio = (stamp - ta) / (tb - ta) if tb > ta else 0.
            result = a + ratio * (b - a)
            result[2] = a[2] + ratio * wrap(b[2] - a[2])
            return result
    nearest_time, nearest = min(samples, key=lambda sample: abs(sample[0] - stamp))
    return nearest.copy() if abs(nearest_time - stamp) <= tolerance else None


def measured_motion(samples, stamp, tolerance):
    """Estimate actual translation/turn rates over 0.2 s of measured poses."""
    current = pose_at(samples, stamp, tolerance)
    previous = pose_at(samples, stamp - 0.20, tolerance)
    if current is None or previous is None:
        return None
    return (np.linalg.norm(current[:2] - previous[:2]) / 0.20,
            abs(wrap(current[2] - previous[2])) / 0.20)


class MovingObstacleTest(Node):
    def __init__(self):
        super().__init__("moving_obstacle_test")
        self.p = {key: self.declare_parameter(key, value).value for key, value in PARAMS.items()}
        p = self.p
        positive = (
            "detect_half_width_m", "track_gap_s", "association_slack_m", "max_object_speed_mps",
            "motion_distance_m", "motion_fit_error_m", "max_approach_m", "heading_limit_deg",
            "stop_ahead_m", "sensor_timeout_s", "pose_sync_tolerance_s", "stop_hold_s",
            "stopped_speed_mps", "stopped_turn_rps", "heading_tolerance_deg", "heading_hold_s",
            "heading_gain", "wall_tolerance_m", "wall_angle_limit_deg", "min_turn_rps", "max_turn_rps",
            "reference_delay_s", "reference_max_age_s", "point_change_m", "free_space_margin_m",
            "motion_cluster_radius_m", "min_object_speed_mps",
            "body_front_m", "body_rear_m", "body_half_width_m", "safety_margin_m", "log_interval_s")
        if (not 0 < p["detect_near_m"] < p["detect_far_m"]
                or not 0 < p["min_observation_s"] <= p["observation_window_s"]
                or not 0 < p["approach_speed_mps"] <= 0.3
                or not 0 < p["side_speed_mps"] <= 0.3
                or any(p[key] <= 0 for key in positive)
                or not 0 < p["min_coverage_fraction"] <= 1
                or not 0 < p["min_static_fraction"] <= 1
                or p["min_motion_points"] < 3 or p["min_static_points"] < 3
                or p["confirm_scans"] < 1 or p["min_observations"] < 3
                or p["reference_delay_s"] >= p["min_observation_s"]
                or p["reference_max_age_s"] < p["reference_delay_s"]
                or p["min_object_speed_mps"] >= p["max_object_speed_mps"]
                or p["min_turn_rps"] > p["max_turn_rps"]
                or p["wall_clearance_m"] <= p["body_half_width_m"] + p["safety_margin_m"] + 0.05):
            raise ValueError("Invalid detection, speed, or wall-clearance parameters")
        self.scan = None
        self.scan_received = 0.
        self.poses = deque(maxlen=300)
        self.origin = None
        self.processed_stamp = None
        self.state = "APPROACH"
        self.stopped_since = self.aligned_since = None
        self.last_log = -math.inf
        self.last_forward_command = 0.
        self.motion_ready = True
        self.detector = MotionDetector(p)
        self.side = SidePositioner(p)
        self.pub = self.create_publisher(Twist, p["cmd_topic"], 10)
        self.create_subscription(LaserScan, p["scan_topic"], self.scan_callback, qos_profile_sensor_data)
        if p["pose_type"] not in ("odom", "pose_stamped"):
            raise ValueError("pose_type must be odom or pose_stamped")
        pose_class = Odometry if p["pose_type"] == "odom" else PoseStamped
        self.create_subscription(pose_class, p["pose_topic"], self.pose_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info(
            f"APPROACH -> STOP -> ALIGN ({p['target_heading_deg']:g}deg in pose frame) -> SIDE -> HOLD")

    @staticmethod
    def stamp(message):
        return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9

    def scan_callback(self, msg):
        if self.scan is not None:
            if self.stamp(msg) == self.stamp(self.scan):
                return  # Repeated header timestamps cannot keep old data fresh.
            if self.stamp(msg) < self.stamp(self.scan):
                self.state = "HOLD"
                self.publish()
                self.get_logger().error("Scan time moved backwards; restart test")
        self.scan, self.scan_received = msg, time.monotonic()

    def pose_callback(self, msg):
        pose = msg.pose.pose if self.p["pose_type"] == "odom" else msg.pose
        q = pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        value = np.array([pose.position.x, pose.position.y, yaw])
        if not np.isfinite(value).all():
            self.state = "HOLD"
            self.publish()
            self.get_logger().error("Invalid pose; restart test after checking localization")
            return
        stamp = self.stamp(msg)
        if self.poses and stamp < self.poses[-1][0]:
            self.poses.clear()
            self.state = "HOLD"
            self.publish()
            self.get_logger().error("Pose time moved backwards; restart test after simulator reset")
        if self.poses and stamp == self.poses[-1][0]:
            return
        self.poses.append((stamp, value))

    def publish(self, x=0., y=0., turn=0.):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(x), float(y), float(turn)
        self.pub.publish(msg)
        self.last_forward_command = float(x)

    def log(self, message):
        now = time.monotonic()
        if now - self.last_log >= self.p["log_interval_s"]:
            self.get_logger().info(message)
            self.last_log = now

    def clear(self, angles, points, covered, heading, distance, rotating=False):
        """Check an inflated swept rectangle; alignment uses a covering circle."""
        p = self.p
        diff = (angles - heading + np.pi) % (2*np.pi) - np.pi
        sector = np.ones(len(angles), dtype=bool) if rotating else np.abs(diff) < math.radians(25)
        if np.count_nonzero(sector) < 3 or np.mean(covered[sector]) < p["min_coverage_fraction"]:
            return False
        points = points[np.isfinite(points).all(axis=1)]
        margin = p["safety_margin_m"]
        if rotating:
            radius = math.hypot(max(p["body_front_m"], p["body_rear_m"]), p["body_half_width_m"]) + margin
            return bool(np.all(np.linalg.norm(points, axis=1) > radius))
        endpoint = distance * np.array([math.cos(heading), math.sin(heading)])
        lower = np.array([-p["body_rear_m"], -p["body_half_width_m"]]) - margin + np.minimum(endpoint, 0.)
        upper = np.array([p["body_front_m"], p["body_half_width_m"]]) + margin + np.maximum(endpoint, 0.)
        return not np.any(np.all((points >= lower) & (points <= upper), axis=1))

    def tick(self):
        if self.state == "HOLD":
            self.publish()
            return
        if self.scan is None or time.monotonic() - self.scan_received > self.p["sensor_timeout_s"]:
            self.publish()
            self.stopped_since = self.aligned_since = None
            self.log("STOP: scan unavailable/stale")
            return
        angles, points, covered = scan_points(self.scan, self.p)
        clear = lambda heading, distance, rotating=False: self.clear(
            angles, points, covered, heading, distance, rotating)
        stamp = self.stamp(self.scan)
        pose = pose_at(self.poses, stamp, self.p["pose_sync_tolerance_s"])
        if pose is None:
            self.publish()
            self.stopped_since = self.aligned_since = None
            self.log("STOP: waiting for pose matched to scan time (check pose_topic and clocks)")
            return
        if self.origin is None:
            self.origin = pose.copy()
        motion = measured_motion(self.poses, stamp, self.p["pose_sync_tolerance_s"])
        stationary = (motion is not None and motion[0] <= self.p["stopped_speed_mps"]
                      and motion[1] <= self.p["stopped_turn_rps"])
        heading_error = wrap(math.radians(self.p["target_heading_deg"]) - pose[2])
        aligned = abs(heading_error) <= math.radians(self.p["heading_tolerance_deg"])
        if self.state == "STOP":
            self.publish()
            if not stationary:
                self.stopped_since = None
            elif self.stopped_since is None:
                self.stopped_since = stamp
            elif stamp - self.stopped_since >= self.p["stop_hold_s"]:
                self.state = "ALIGN"
                self.get_logger().info("STOP confirmed from measured pose; aligning heading")
            self.log(f"STOP: measured speed/turn={motion}; holding before alignment")
            return
        if self.state == "ALIGN":
            if aligned:
                self.publish()
                if not stationary:
                    self.aligned_since = None
                elif self.aligned_since is None:
                    self.aligned_since = stamp
                elif stamp - self.aligned_since >= self.p["heading_hold_s"]:
                    self.state = "SIDE"
                    self.get_logger().info("ALIGN confirmed; beginning side positioning")
            else:
                self.aligned_since = None
                # Never translate during alignment; check room for the body to rotate.
                if clear(0., 0., rotating=True):
                    turn = np.sign(heading_error) * np.clip(
                        abs(heading_error) * self.p["heading_gain"], self.p["min_turn_rps"], self.p["max_turn_rps"])
                    self.publish(turn=turn)
                else:
                    self.publish()
            self.log(f"ALIGN: heading error={math.degrees(heading_error):+.1f}deg")
            return
        if self.state == "SIDE":
            if not aligned:
                self.publish()
                self.state = "ALIGN"
                self.aligned_since = None
                self.get_logger().info("SIDE paused: restoring target heading")
                return
            command, status = self.side.command(angles, points, clear)
            self.publish(*command)
            self.log(status)
            if self.side.done:
                self.state = "HOLD"
                self.get_logger().info(status)
            return
        if stamp != self.processed_stamp:
            if self.processed_stamp is not None and stamp < self.processed_stamp:
                self.state = "HOLD"
                self.publish()
                self.get_logger().error("Scan time moved backwards; restart test")
                return
            self.processed_stamp = stamp
            detected, status = self.detector.update(angles, points, covered, pose, stamp, self.scan.range_max)
            self.log("APPROACH | " + status)
            self.motion_ready = not status.startswith("motion uncertain")
            if not self.motion_ready:
                self.publish()
                return
            if detected:
                was_forward = self.last_forward_command > 0.
                self.publish()
                self.get_logger().info(
                    f"MOVING OBJECT DETECTED | forward_command_active={was_forward} "
                    f"measured_robot_motion={motion} | " + detected)
                self.state = "STOP"
                self.stopped_since = self.aligned_since = None
                return
        travel = np.linalg.norm(pose[:2] - self.origin[:2])
        if travel >= self.p["max_approach_m"] or abs(wrap(pose[2] - self.origin[2])) > math.radians(self.p["heading_limit_deg"]):
            self.publish()
            self.state = "HOLD"
            self.get_logger().warning("HOLD: approach travel/heading limit reached without detection")
        elif not self.motion_ready:
            self.publish()
        elif not clear(0., self.p["stop_ahead_m"]):
            self.publish()
            self.log("STOP: approach path blocked/unknown; detection continues")
        else:
            self.publish(x=self.p["approach_speed_mps"])


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MovingObstacleTest()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.timer.cancel()
            if rclpy.ok():
                node.publish()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
