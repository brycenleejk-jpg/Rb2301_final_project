"""Goal/depth LiDAR navigation with checked curved motion below 50 degrees.

G is relative to the COURSE origin, not the robot's initial position.
Keep optitrack_variables.config beside this file. Define course +x as right
and +y as up the course, then enable course_frame_ready.
"""
import argparse
import configparser
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# ========================= TUNE THESE SETTINGS =========================
PARAMS = {
    "goal_x_m": 1.421,                 # Right from course bottom-left.
    "goal_y_m": 0.831,                 # Up the course from bottom-left.
    "course_frame_ready": True,      # True AFTER confirming origin and axes.
    "goal_tolerance_m": 0.13,         # Stop translating within this distance.
    "goal_heading_deg": -90,         # None = last intended approach heading.
                                     # Or set course heading: 0=+x, 90=+y.
    "check_distance_m": 0.30,         # Length of each straight clearance check.
    "candidate_step_deg": 3.0,        # Smaller = more directions/computation.
    "view_half_angle_deg": 90.0,      # Candidates: right 90 to left 90.
    "robot_radius_m": 0.110,           # TOTAL clearance radius, including margin.
    "depth_weight": 1.5,             # 0 disables depth preference.
    "depth_cap_m": 3.00,              # Cap the depth reward at this distance.
    "lidar_yaw_deg": 180.0,           # From friend's code: VERIFY mounting.
    "lidar_x_m": 0.11423,             # From friend's code: VERIFY offset.
    "lidar_y_m": 0.0,                 # Sensor's left offset from robot centre.
    "forward_speed_mps": 0.30,        # Real translation remains capped at 0.3.
    "goal_speed_gain": 0.6,           # Speed <= gain * distance to G.
    "turn_gain": 1.0,                 # Turn rad/s per radian of error.
    "min_turn_rps": 0.20,             # About 11.5 degrees/second
    "max_turn_rps": 0.50,             # Turn speed limit; real robot remains capped at 0.5rad/s = 28.6deg/s.x   
    "moving_turn_limit_deg": 50.0,    # Below this: allow checked forward + turn.
    "final_heading_tolerance_deg": 3.0,
    "sensor_timeout_s": 0.50,         # Stop if scan OR pose stops updating.
}
# =====================================================================

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


def wrap_deg(angle):
    """Keep a degree angle in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def optitrack_to_maze(x, y, heading_deg, frame):
    """Express the OptiTrack pose in the calibrated CA2 maze frame."""
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))


def maze_to_course(x, y, heading_deg, course):
    """Express a maze pose relative to the course origin and axes."""
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - course["rotation_deg"]))


def load_frame(robot_number):
    """Retain template calibration and per-robot heading corrections."""
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    keys = ("origin_x", "origin_y", "rotation_deg")
    frame = {k: parser.getfloat("frame", k) for k in keys}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    frame["heading_offset_deg"] = (wrap_deg(-(frame["rotation_deg"] + 90.0))
                                   if raw == "auto" else float(raw))
    if parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in keys}
    return frame, course


class LidarNode(Node):
    """Retain the template's simulation/real scan-length detection."""
    def __init__(self):
        super().__init__("LidarNode")
        self.is_simulation = None
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.callback, 2)

    def callback(self, msg):
        self.is_simulation = len(msg.ranges) <= 360


class ObstacleCourseNode(Node):
    def __init__(self, is_simulation=True, calibrate=False):
        super().__init__("obstaclecourse")
        self.p = PARAMS.copy()
        self.is_simulation = is_simulation
        self.calibrate = calibrate
        self.max_translate_velocity = 1.4 if is_simulation else 0.3
        # Measured circle about the robot reference centre; includes margin.
        self.robot_radius = self.p["robot_radius_m"]
        self.pose = self.raw_pose = self.scan_msg = None
        self.frame = self.course = None
        self.last_scan_time = self.last_pose_time = 0.0
        self.finishing = self.done = False
        self.approach_heading_deg = self.finish_heading_deg = None

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2)
        if is_simulation:
            self.subscription = self.create_subscription(Odometry, "odom", self.odometer_callback, 2)
        else:
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(number)
            topic = f"/vrpn_mocap/bingda_{number:03d}/pose"
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.map_sub = self.create_subscription(PoseStamped, topic, self.optitrack_callback, qos)
            self.get_logger().info(f"Reading pose from {topic}")
        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def sub_scan_callback(self, msg):
        self.scan_msg = msg  # Keep full scan and angle metadata.
        self.last_scan_time = time.monotonic()

    @staticmethod
    def yaw_from_quaternion(q):
        return math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y * q.y + q.z * q.z))

    def optitrack_callback(self, msg):
        heading = math.degrees(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array([msg.pose.position.x, msg.pose.position.y, heading])
        self.pose = np.array(optitrack_to_maze(*self.raw_pose, self.frame))
        self.last_pose_time = time.monotonic()

    def odometer_callback(self, msg):
        pose = msg.pose.pose
        self.pose = np.array([pose.position.x, pose.position.y,
                              math.degrees(self.yaw_from_quaternion(pose.orientation))])
        self.last_pose_time = time.monotonic()

    def to_course(self):
        # Simulation uses its odometry frame as course coordinates. Real
        # OptiTrack uses the [course] transformation in the existing config.
        return (np.array(maze_to_course(*self.pose, self.course))
                if self.course is not None else self.pose.copy())

    def move_2D(self, x=0.0, y=0.0, turn=0.0):
        msg = Twist()
        cap = self.max_translate_velocity
        msg.linear.x = float(np.clip(x, -cap, cap))
        msg.linear.y = float(np.clip(y, -cap, cap))
        msg.angular.z = float(np.clip(turn, -2 * cap, 2 * cap))
        self.publisher_.publish(msg)

    def scan_geometry(self):
        """Return finite obstacle points and their original ray angles."""
        scan = self.scan_msg
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = (scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
                  + math.radians(self.p["lidar_yaw_deg"]))
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        # Ignore +/-inf and NaN completely: neither obstacle points nor targets.
        # Do NOT replace inf with range_max.
        valid = (np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0)
                 & (ranges >= scan.range_min) & (ranges <= scan.range_max))
        ranges, angles = ranges[valid], angles[valid]
        points = np.column_stack((ranges * np.cos(angles) + self.p["lidar_x_m"],
                                  ranges * np.sin(angles) + self.p["lidar_y_m"]))
        return points, angles

    def path_is_clear(self, points, heading, distance):
        """Check ALL finite points against a body-sized straight corridor."""
        if len(points) == 0:
            return False
        c, s = math.cos(heading), math.sin(heading)
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        closest = np.clip(along, 0.0, distance)
        return bool(np.all(np.hypot(along - closest, side) > self.robot_radius))

    def path_depth(self, points, heading):
        """Estimated free travel before the body circle touches a finite hit."""
        c, s = math.cos(heading), math.sin(heading)
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        relevant = (along > 0.0) & (np.abs(side) <= self.robot_radius)
        if not np.any(relevant):
            return 0.0  # Missing supporting returns do not earn maximum depth.
        contact = along[relevant] - np.sqrt(np.maximum(
            0.0, self.robot_radius**2 - side[relevant]**2))
        return float(np.clip(np.min(contact), 0.0, self.p["depth_cap_m"]))

    def motion_is_clear(self, points, speed, turn, distance):
        """Check the actual constant-speed curved command over the lookahead.

        Sample centres at <=2 cm arc spacing. Add half that spacing to the
        radius so obstacles between sample centres cannot slip through.
        This is a short collision check, not a second path planner.
        """
        if speed <= 0.0:
            return self.path_is_clear(points, 0.0, 0.0)
        if abs(turn) < 1e-6:
            return self.path_is_clear(points, 0.0, distance)
        if len(points) == 0:
            return False
        count = max(1, int(math.ceil(distance / 0.02)))
        travel = np.linspace(0.0, distance, count + 1)
        phi = turn * travel / speed
        centres = np.column_stack((speed / turn * np.sin(phi),
                                   speed / turn * (1.0 - np.cos(phi))))
        separation = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        padding = distance / (2 * count)
        return bool(np.all(separation > self.robot_radius + padding))

    def select_direction(self, points, angles, goal_bearing, distance):
        """Choose by goal alignment plus a small capped depth reward."""
        view = np.sort(angles[np.abs(angles) <= math.radians(self.p["view_half_angle_deg"])])
        self._candidate_count = 0
        if len(view) == 0:
            self.plot_paths(points, [], [], distance)
            return None
        # Space candidates by angle. Also include measured directions nearest
        # G and straight ahead. Each candidate is tested without region grouping.
        step = math.radians(self.p["candidate_step_deg"])
        candidates = []
        for angle in view:
            if not candidates or angle - candidates[-1] >= step:
                candidates.append(float(angle))
        error = (view - goal_bearing + np.pi) % (2 * np.pi) - np.pi
        candidates.extend([float(view[np.argmin(np.abs(error))]),
                           float(view[np.argmin(np.abs(view))])])
        candidates = list(dict.fromkeys(candidates))
        self._candidate_count = len(candidates)
        clear, scores = [], []
        best_direction, best_key = None, (-math.inf, -math.inf)
        for angle in candidates:
            # A high reward NEVER overrides a failed 0.5 m clearance check.
            ok = self.path_is_clear(points, angle, distance)
            clear.append(ok)
            scores.append(None)  # Rejected directions do not have a score.
            if not ok:
                continue
            alignment = math.cos(angle - goal_bearing)
            depth_score = self.path_depth(points, angle) / self.p["depth_cap_m"]
            score = alignment + self.p["depth_weight"] * depth_score
            scores[-1] = score
            key = (score, -abs(angle))  # Equal scores: prefer less turning.
            if key > best_key:
                best_key, best_direction = key, angle
        self.plot_paths(points, candidates, clear, distance, scores)
        return best_direction

    def plot_goal(self):
        """Send course goal expressed relative to the current robot pose."""
        now = time.monotonic()
        valid = (self.pose is not None and np.all(np.isfinite(self.pose))
                 and now - self.last_pose_time <= self.p["sensor_timeout_s"])
        if (valid == getattr(self, "_goal_plot_valid", None)
                and now - getattr(self, "_goal_plot_time", 0.0) < 0.5):
            return
        self._goal_plot_valid, self._goal_plot_time = valid, now
        if not valid:
            self.publish_plot("Goal: none")
            return
        x, y, heading = self.to_course()
        dx, dy = self.p["goal_x_m"] - x, self.p["goal_y_m"] - y
        yaw = math.radians(heading)
        gx = dx * math.cos(yaw) + dy * math.sin(yaw)
        gy = -dx * math.sin(yaw) + dy * math.cos(yaw)
        self.publish_plot(
            f"Goal: x={gx:+.3f} m, y={gy:+.3f} m, radius={self.robot_radius:.3f} m")

    def blocked_reason(self, points, angles, distance):
        """Two short reasons: overlap at the start, or no tested path fits."""
        nearest = float(np.min(np.linalg.norm(points, axis=1)))
        if nearest <= self.robot_radius:
            # This circle includes the safety margin and all body corners.
            # A point inside it blocks every path, even if behind the robot.
            return (f"STOP: footprint blocked | nearest={nearest:.3f}m "
                    f"| radius={self.robot_radius:.3f}m")
        return f"STOP: no clear path | tested={getattr(self, '_candidate_count', 0)}"

    def plot_paths(self, points=None, candidates=(), clear=(), distance=0.5, scores=()):
        """Send checked paths at 2 Hz; clear immediately when data is invalid.

        Viewer data uses a separate topic to avoid long terminal messages.
        Red means rejected, pale green means clear.
        Infinite rays never become candidates or plotted obstacle points.
        """
        now = time.monotonic()
        active = points is not None
        if (active == getattr(self, "_paths_active", None)
                and now - getattr(self, "_paths_time", 0.0) < 0.5):
            return
        self._paths_active, self._paths_time = active, now
        message = "Paths: none"
        if active:
            inside = points[np.linalg.norm(points, axis=1) <= self.robot_radius]
            payload = {"angles": [round(math.degrees(a), 3) for a in candidates],
                       "clear": clear, "scores": scores, "distance": float(distance),
                       "radius": self.robot_radius, "inside": inside.tolist()}
            message = "Paths: " + json.dumps(payload, separators=(",", ":"), allow_nan=False)
        if not hasattr(self, "_paths_publisher"):
            self._paths_publisher = self.create_publisher(String, "navigation/paths", 2)
        self._paths_publisher.publish(String(data=message))

    def plot_direction(self, heading=None, distance=0.5):
        """Viewer-only endpoint in ROBOT coordinates; never affects motion.

        Send plotting data through a topic, without flooding the terminal.
        Send target/no-target transitions immediately; otherwise update at 2 Hz.
        """
        now = time.monotonic()
        active = heading is not None
        if (active == getattr(self, "_plot_active", None)
                and now - getattr(self, "_plot_time", 0.0) < 0.5):
            return
        self._plot_active, self._plot_time = active, now
        message = "Carrot: none"
        if active:
            x, y = distance * math.cos(heading), distance * math.sin(heading)
            message = f"Carrot: x={x:+.3f} m, y={y:+.3f} m"
        self.publish_plot(message)

    def publish_plot(self, message):
        """Carrot/goal telemetry only; has no effect on navigation."""
        if not hasattr(self, "_plot_publisher"):
            self._plot_publisher = self.create_publisher(String, "navigation/plot", 10)
        self._plot_publisher.publish(String(data=message))

    def report_status(self, message):
        """Log once per second, or immediately when the reason changes."""
        now = time.monotonic()
        # Ignore changing numbers so DRIVE/angle updates do not flood logs.
        reason = message.split("|")[0].split("=")[0].strip()
        if (reason != getattr(self, "_status_reason", None)
                or now - getattr(self, "_status_time", 0.0) >= 1.0):
            self._status_reason, self._status_time = reason, now
            self.get_logger().info(message)

    def stop(self, reason, keep_paths=False):
        self.move_2D()
        self.plot_direction()  # Clear the selected-direction display immediately.
        if not keep_paths:
            self.plot_paths()
        self.report_status(reason)

    def timer_callback(self):
        self.plot_goal()  # Visible even while navigation is blocked or finished.
        if self.done:
            self.report_status("DONE: goal reached and heading aligned")
            self.plot_paths()
            self.plot_direction()
            self.move_2D()
            return
        now = time.monotonic()
        if (self.pose is None or not np.all(np.isfinite(self.pose))
                or now - self.last_pose_time > self.p["sensor_timeout_s"]):
            self.stop("STOP: pose missing, invalid or stale")
            return
        if self.calibrate:
            self.plot_paths()
            self.plot_direction()
            self.move_2D()
            self.get_logger().info(
                f"Raw={self.raw_pose}; maze={self.pose}; course={self.to_course()}",
                throttle_duration_sec=0.5)
            return
        if not self.p["course_frame_ready"]:
            self.stop("STOP: course_frame_ready=False")
            return
        if (self.scan_msg is None
                or now - self.last_scan_time > self.p["sensor_timeout_s"]):
            self.stop("STOP: scan missing or stale")
            return
        points, angles = self.scan_geometry()
        if len(points) == 0:
            self.stop("STOP: no clear path | tested=0 (no finite returns)")
            return

        x, y, heading = self.to_course()
        dx, dy = self.p["goal_x_m"] - x, self.p["goal_y_m"] - y
        goal_distance = math.hypot(dx, dy)
        if goal_distance <= self.p["goal_tolerance_m"] and not self.finishing:
            # Freeze arrival heading once; don't chase a noisy near-zero goal bearing.
            self.finishing = True
            configured = self.p["goal_heading_deg"]
            self.finish_heading_deg = (configured if configured is not None
                                       else self.approach_heading_deg)
            if self.finish_heading_deg is None:
                self.finish_heading_deg = heading

        if self.finishing:
            self.plot_paths()
            self.plot_direction()  # Final rotation is not a selected travel path.
            error_deg = wrap_deg(self.finish_heading_deg - heading)
            if abs(error_deg) <= self.p["final_heading_tolerance_deg"]:
                self.done = True
                self.stop("DONE: goal reached and heading aligned")
            elif self.path_is_clear(points, 0.0, 0.0):
                # A zero-length corridor checks the body circle for rotation.
                turn = np.clip(self.p["turn_gain"] * math.radians(error_deg),
                               -self.p["max_turn_rps"], self.p["max_turn_rps"])
                self.move_2D(turn=float(turn))
                self.report_status("ALIGN: final goal heading")
            else:
                self.stop("STOP: footprint blocked during final alignment")
            return

        goal_bearing = math.radians(wrap_deg(math.degrees(math.atan2(dy, dx)) - heading))
        distance = min(self.p["check_distance_m"], goal_distance)
        selected = self.select_direction(points, angles, goal_bearing, distance)
        if selected is None:
            self.stop(self.blocked_reason(points, angles, distance), keep_paths=True)
            return
        self.approach_heading_deg = wrap_deg(heading + math.degrees(selected))
        error_deg = abs(math.degrees(selected))
        turn = float(np.clip(self.p["turn_gain"] * selected,
                             -min(self.p["max_turn_rps"], 2 * self.max_translate_velocity),
                             min(self.p["max_turn_rps"], 2 * self.max_translate_velocity)))
        # Slow down as turning becomes sharper, and as G approaches.
        speed = min(self.p["forward_speed_mps"], self.max_translate_velocity,
                    self.p["goal_speed_gain"] * goal_distance)
        speed *= max(0.0, math.cos(selected))

        if error_deg >= self.p["moving_turn_limit_deg"]:
            reason = f"ALIGN: angle={error_deg:.1f}deg >= {self.p['moving_turn_limit_deg']:g}deg"
        #elif True:
        elif self.motion_is_clear(points, speed, turn, distance):
            self.move_2D(x=speed, turn=turn)
            self.plot_direction(selected, distance)
            self.report_status(
                f"DRIVE | speed={speed:.2f}m/s | goal={goal_distance:.2f}m")
            return
        elif self.path_is_clear(points, 0.0, distance):
            # Curve blocked, but straight ahead is clear.
            self.move_2D(x=speed)
            self.approach_heading_deg = heading
            self.plot_direction(0.0, distance)
            self.report_status(
                f"DRIVE: straight path clear; curve blocked | speed={speed:.2f}m/s")
            return
        else:
            reason = "ALIGN: curved forward path blocked"

        # Large angle or blocked curve: stop translating, rotate if there is room.
        if not self.path_is_clear(points, 0.0, 0.0):
            self.stop(self.blocked_reason(points, angles, distance), keep_paths=True)
            return
        turn = float(np.sign(turn) * np.clip(
            abs(turn),
            self.p["min_turn_rps"],
            self.p["max_turn_rps"],
        ))
        self.move_2D(turn=turn)
        self.plot_direction(selected, distance)
        self.report_status(reason)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Print poses without driving")
    cli, ros_args = parser.parse_known_args(args=args)
    rclpy.init(args=ros_args)
    lidar = LidarNode()
    obstacle = None
    try:
        while rclpy.ok() and lidar.is_simulation is None:
            rclpy.spin_once(lidar, timeout_sec=0.2)
        if rclpy.ok():
            obstacle = ObstacleCourseNode(lidar.is_simulation, cli.calibrate)
            rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if obstacle is not None:
            obstacle.timer.cancel()
            if rclpy.ok():
                obstacle.move_2D()
            obstacle.destroy_node()
        lidar.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

