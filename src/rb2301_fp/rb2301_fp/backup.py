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


# Navigation settings (distances in metres, angles in degrees).
PARAMS = {
    "goal_x_m": 5.5,                   # Goal x from course origin; +x points right.
    "goal_y_m": -3.0,                  # Goal y from course origin; +y points up.
    "course_frame_ready": True,        # Enable after checking course origin and axes.
    "goal_tolerance_m": 0.13,           # Stop moving forward this close to the goal.
    "goal_heading_deg": -90,           # Final angle: 0=+x, 90=+y; None=approach angle.
    "check_distance_m": 0.30,          # How far ahead to check for obstacles.
    "candidate_step_deg": 3.0,         # Smaller steps test more directions.
    "view_half_angle_deg": 90.0,       # Search up to 90 degrees to either side.
    "robot_radius_m": 0.110,           # Minimum radius used for collision checks.
    "preferred_clearance_m": 0.130,    # Try this larger radius first for extra space.
    "reduced_clearance_speed_mps": 0.10, # Speed cap when extra space is unavailable.
    "depth_weight": 1.5,               # Preference for longer open paths; 0 disables it.
    "depth_cap_m": 3.00,               # Paths beyond this earn no extra depth score.
    "similar_score_tolerance": 0.03,   # Keep a clear choice this close to the best score.
    "lidar_yaw_deg": 180.0,            # Correct for the backward-facing LiDAR.
    "lidar_x_m": 0.11423,              # LiDAR position ahead of robot centre.
    "lidar_y_m": 0.0,                  # LiDAR position left of robot centre.
    "forward_speed_mps": 0.30,         # Maximum requested forward speed (m/s).
    "goal_speed_gain": 0.6,            # Speed cap = gain * distance to goal.
    "turn_gain": 1.0,                  # Turn rate = gain * angle error in radians.
    "min_turn_rps": 0.20,              # Minimum turn rate when turning in place (rad/s).
    "max_turn_rps": 0.50,              # Maximum turn rate (rad/s).
    "moving_turn_limit_deg": 50.0,     # Turn in place at or above this angle error.
    "final_heading_tolerance_deg": 3.0, # Allowed final angle error.
    "sensor_timeout_s": 0.50,          # Stop if scan or pose is older than this.
}

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


def wrap_deg(angle):
    """Keep a degree angle in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def optitrack_to_maze(x, y, heading_deg, frame):
    """Convert OptiTrack position and heading to maze coordinates."""
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))


def maze_to_course(x, y, heading_deg, course):
    """Convert maze position and heading to course coordinates."""
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - course["rotation_deg"]))


def load_frame(robot_number):
    """Load course calibration and this robot's heading correction."""
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
    """Identify simulation or real hardware from the number of scan readings."""
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
        self.robot_radius = self.p["robot_radius_m"]  # Circle used to represent the robot.
        self.pose = self.raw_pose = self.scan_msg = None
        self.frame = self.course = None
        self.last_scan_time = self.last_pose_time = 0.0
        self.finishing = self.done = False
        self.approach_heading_deg = self.finish_heading_deg = None
        self.selected_course_heading_deg = None  # Remember the chosen direction as the robot turns.

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
        self.scan_msg = msg  # Save distances and scan angles together.
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
        # Real hardware needs the course transform; simulation already uses course coordinates.
        return (np.array(maze_to_course(*self.pose, self.course))
                if self.course is not None else self.pose.copy())

    def move_2D(self, x=0.0, y=0.0, turn=0.0):
        msg = Twist()
        cap = self.max_translate_velocity
        msg.linear.x = float(np.clip(x, -cap, cap))
        msg.linear.y = float(np.clip(y, -cap, cap))
        msg.angular.z = float(np.clip(turn, -2 * cap, 2 * cap))
        self.publisher_.publish(msg)
        self._last_command = (msg.linear.x, msg.angular.z)  # Sent speeds for logging.

    def scan_geometry(self):
        """Convert valid LiDAR readings to robot-relative obstacle points and ray angles."""
        scan = self.scan_msg
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = (scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
                  + math.radians(self.p["lidar_yaw_deg"]))
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        # Discard invalid readings; missing readings do not count as open space.
        valid = (np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0)
                 & (ranges >= scan.range_min) & (ranges <= scan.range_max))
        ranges, angles = ranges[valid], angles[valid]
        # Account for the LiDAR's position relative to the robot centre.
        points = np.column_stack((ranges * np.cos(angles) + self.p["lidar_x_m"],
                                  ranges * np.sin(angles) + self.p["lidar_y_m"]))
        return points, angles

    def path_is_clear(self, points, heading, distance, radius=None):
        """Check whether the robot's circle can move along a straight path."""
        if len(points) == 0:
            return False
        # Default to the robot radius; any supplied radius must be at least this large.
        radius = self.robot_radius if radius is None else max(self.robot_radius, radius)
        c, s = math.cos(heading), math.sin(heading)
        # Measure each obstacle along and across the proposed path.
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        closest = np.clip(along, 0.0, distance)  # Nearest position on the path to each obstacle.
        return bool(np.all(np.hypot(along - closest, side) > radius))

    def path_depth(self, points, heading):
        """Estimate how far the robot can travel before touching an obstacle."""
        c, s = math.cos(heading), math.sin(heading)
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        relevant = (along > 0.0) & (np.abs(side) <= self.robot_radius)
        if not np.any(relevant):
            return 0.0  # No supporting readings means no depth reward.
        # Allow for the robot's curved front edge. Used for scoring, not collision checks.
        contact = along[relevant] - np.sqrt(np.maximum(
            0.0, self.robot_radius**2 - side[relevant]**2)) #calc how far robot centre can move forward before its circular body touches obstacle
        return float(np.clip(np.min(contact), 0.0, self.p["depth_cap_m"]))

    def motion_is_clear(self, points, speed, turn, distance, radius=None):
        """Check the curved path produced by the requested forward and turn speeds."""
        if speed <= 0.0:
            return self.path_is_clear(points, 0.0, 0.0)
        radius = self.robot_radius if radius is None else max(self.robot_radius, radius)
        if abs(turn) < 1e-6:
            return self.path_is_clear(points, 0.0, distance, radius=radius)
        if len(points) == 0:
            return False
        count = max(1, int(math.ceil(distance / 0.02)))  # Check at most 2 cm apart.
        travel = np.linspace(0.0, distance, count + 1)
        # Predict robot-centre positions along the curve.
        phi = turn * travel / speed
        centres = np.column_stack((speed / turn * np.sin(phi),
                                   speed / turn * (1.0 - np.cos(phi))))
        separation = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        padding = distance / (2 * count)  # Extra radius covers gaps between checked positions.
        return bool(np.all(separation > radius + padding))

    def select_direction(self, points, angles, goal_bearing, distance, radius=None,
                         previous_direction=None):
        """Choose a clear direction using goal alignment and open-path distance."""
        preferred = max(self.robot_radius, self.p["preferred_clearance_m"]) # preferred is a soft limit (larger than robot radius)
        # Try extra space first; a retry may use the robot radius, but never less.
        radius = preferred if radius is None else max(self.robot_radius, radius)
        self._selection_radius = radius  # Use the same radius for the motion check.
        # Keep forward scan angles in radians, ordered right to left.
        view = np.sort(angles[np.abs(angles) <= math.radians(self.p["view_half_angle_deg"])])
        self._candidate_count = 0  # Reset the number of tested directions.
        self._selection_summary = None
        self._selection_held = False
        if len(view) == 0:
            # No valid forward readings means no direction can be selected.
            self.plot_paths(points, [], [], distance)
            return None
        # Space out scan directions; also include those closest to the goal and straight ahead.
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
        clear, scores, depths = [], [], []
        best_direction, best_key = None, (-math.inf, -math.inf)
        runner_direction, runner_key = None, (-math.inf, -math.inf)  # Second-best choice for logs.
        for angle in candidates:
            # Reject blocked straight paths before scoring; check the actual curve later.
            ok = self.path_is_clear(points, angle, distance, radius=radius)
            clear.append(ok)
            scores.append(None)  # Blocked directions stay unscored.
            depths.append(None)
            if not ok:
                continue
            alignment = math.cos(angle - goal_bearing)
            depths[-1] = self.path_depth(points, angle)  # Estimated free travel in metres.
            depth_score = depths[-1] / self.p["depth_cap_m"]
            score = alignment + self.p["depth_weight"] * depth_score  # Reward goal alignment and space.
            scores[-1] = score
            key = (score, -abs(angle))  # Equal scores: prefer less turning.
            if key > best_key:
                runner_key, runner_direction = best_key, best_direction  # Old best becomes second best.
                best_key, best_direction = key, angle
            elif key > runner_key:
                runner_key, runner_direction = key, angle
        # If no path fits with extra space, retry once using the robot radius.
        if best_direction is None and radius > self.robot_radius:
            self.get_logger().info(
                f"SOFT LIMIT: no path at {radius:.2f}m; trying {self.robot_radius:.2f}m",
                throttle_duration_sec=1.0)
            return self.select_direction(points, angles, goal_bearing, distance,
                                         radius=self.robot_radius,
                                         previous_direction=previous_direction)
        self.plot_paths(points, candidates, clear, distance, scores)  # Show the final search result.
        if best_direction is not None:
            # Only compare clear paths close to the highest combined score.
            tolerance = max(0.0, self.p["similar_score_tolerance"])  # Dimensionless score band.
            near_best = [i for i, ok in enumerate(clear)
                         if ok and best_key[0] - scores[i] <= tolerance + 1e-12]
            # New choice: prefer more depth, then score and less turning for ties.
            chosen = max(near_best, key=lambda i: (depths[i], scores[i], -abs(candidates[i])))
            if previous_direction is not None:
                previous = min(range(len(candidates)),
                               key=lambda i: abs(candidates[i] - previous_direction))
                # Match the remembered heading to a nearby tested ray. It must
                # still be clear and competitive; never lock onto a blocked path.
                if (abs(candidates[previous] - previous_direction) <= step + 1e-9
                        and previous in near_best):
                    chosen = previous
                    self._selection_held = True
            selected = candidates[chosen]
            # A retained/deeper choice can score slightly below the raw winner.
            # Show the strongest OTHER option; gap remains the raw top-two gap.
            alternative = runner_direction if selected == best_direction else best_direction
            alternative_key = runner_key if selected == best_direction else best_key
            choice = "kept" if self._selection_held else "depth"
            if len(near_best) > 1:  # Only announce a tie-break when multiple scores are close.
                self.get_logger().info(
                    f"TIE BREAK: {'keeping direction' if self._selection_held else 'choosing greater depth'} "
                    f"({math.degrees(selected):+.1f}deg)", throttle_duration_sec=1.0)
            self._selection_summary = (
                f"selected={math.degrees(selected):+.1f}deg "
                f"score={scores[chosen]:.3f} depth={depths[chosen]:.2f}m | choice={choice}")
            if alternative is not None:
                self._selection_summary += (
                    f" | alternative={math.degrees(alternative):+.1f}deg "
                    f"score={alternative_key[0]:.3f} | gap={best_key[0] - runner_key[0]:.3f}")
            mode = "reduced" if radius < preferred else "preferred"
            self._selection_summary += f" | clearance={mode} ({radius:.3f}m)"
            return selected
        return best_direction

    # FOR DEBUGGING (LIVE MAP), NOT PART OF LOGIC OF CODE!
    def plot_goal(self):
        """Send the goal's position relative to the robot to the viewer."""
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

    def blocked_reason(self, points):
        """Explain why the robot stopped: missing data, nearby obstacle, or blocked paths."""
        if len(points) == 0:
            return "STOP: no forward candidates | finite_returns=0"
        nearest = float(np.min(np.linalg.norm(points, axis=1)))
        if nearest <= self.robot_radius:
            # An obstacle inside the robot's circle blocks every direction.
            return (f"STOP: footprint blocked | nearest={nearest:.4f}m "
                    f"| radius={self.robot_radius:.4f}m")
        tested = getattr(self, "_candidate_count", 0)
        if tested == 0:
            return (f"STOP: no forward candidates | finite_returns={len(points)} "
                    "| forward_finite_returns=0")
        # All tested directions were blocked.
        message = (f"STOP: no clear forward path | tested={tested} "
                   f"| nearest={nearest:.4f}m")
        message += f" | required>{self.robot_radius:.4f}m"  # Minimum required clearance.
        return message

    def plot_paths(self, points=None, candidates=(), clear=(), distance=0.5, scores=()):
        """Send paths to the viewer at 2 Hz; send active/cleared changes immediately."""
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
        """Show the chosen target relative to the robot; update at 2 Hz or on active/cleared changes."""
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
        """Publish target and goal positions for display only."""
        if not hasattr(self, "_plot_publisher"):
            self._plot_publisher = self.create_publisher(String, "navigation/plot", 10)
        self._plot_publisher.publish(String(data=message))

    def report_status(self, message):
        """Log once per second, or immediately when the reason changes."""
        now = time.monotonic()
        # Group logs by action so changing numbers do not trigger extra messages.
        reason = message.split("|")[0].split("=")[0].strip()
        if (reason != getattr(self, "_status_reason", None)
                or now - getattr(self, "_status_time", 0.0) >= 1.0):
            self._status_reason, self._status_time = reason, now
            speed, turn = getattr(self, "_last_command", (0.0, 0.0))
            message += f" | v={speed:+.2f}m/s w={turn:+.2f}rad/s"
            # Include position and goal distance only when pose data is recent and valid.
            if (self.pose is not None and np.all(np.isfinite(self.pose))
                    and now - self.last_pose_time <= self.p["sensor_timeout_s"]):
                x, y, heading = self.to_course()
                goal_distance = math.hypot(self.p["goal_x_m"] - x,
                                           self.p["goal_y_m"] - y)
                message += (f" | goal={goal_distance:.2f}m "
                            f"| course=({x:+.3f},{y:+.3f})m heading={heading:+.1f}deg")
            selection = getattr(self, "_selection_summary", None)
            if message.startswith(("DRIVE", "ALIGN:")) and selection:
                message += " | " + selection
            self.get_logger().info(message)

    def stop(self, reason, keep_paths=False):
        self.selected_course_heading_deg = None  # Reconsider direction after a full stop.
        self.move_2D()
        self.plot_direction()  # Clear the displayed target.
        if not keep_paths:
            self.plot_paths()
        self.report_status(reason)

    def timer_callback(self):
        self._selection_summary = None  # Clear the previous update's selection log.
        self.plot_goal()  # Keep the goal visible while stopped or finished.
        if self.done:
            self.plot_paths()
            self.plot_direction()
            self.move_2D()
            self.report_status("DONE: goal reached and heading aligned")
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
            self.stop(self.blocked_reason(points))
            return

        x, y, heading = self.to_course()
        dx, dy = self.p["goal_x_m"] - x, self.p["goal_y_m"] - y
        goal_distance = math.hypot(dx, dy)
        if goal_distance <= self.p["goal_tolerance_m"] and not self.finishing:
            self.finishing = True  # Set the final heading once to avoid jitter near the goal.
            configured = self.p["goal_heading_deg"]
            self.finish_heading_deg = (configured if configured is not None
                                       else self.approach_heading_deg)
            if self.finish_heading_deg is None:
                self.finish_heading_deg = heading

        if self.finishing:
            self.plot_paths()
            self.plot_direction()  # Clear the travel target during the final turn.
            error_deg = wrap_deg(self.finish_heading_deg - heading)
            if abs(error_deg) <= self.p["final_heading_tolerance_deg"]:
                self.done = True
                self.stop("DONE: goal reached and heading aligned")
            elif self.path_is_clear(points, 0.0, 0.0):
                # Zero travel distance checks whether there is room to turn in place.
                turn = np.clip(self.p["turn_gain"] * math.radians(error_deg),
                               -self.p["max_turn_rps"], self.p["max_turn_rps"])
                self.move_2D(turn=float(turn))
                self.report_status(f"ALIGN: final goal heading | error={error_deg:+.1f}deg")
            else:
                self.stop(self.blocked_reason(points) + " | stage=final alignment")
            return

        goal_bearing = math.radians(wrap_deg(math.degrees(math.atan2(dy, dx)) - heading))
        distance = min(self.p["check_distance_m"], goal_distance)
        # Convert the remembered course heading to the robot's current frame.
        previous_direction = (None if self.selected_course_heading_deg is None else
                              math.radians(wrap_deg(self.selected_course_heading_deg - heading)))
        selected = self.select_direction(points, angles, goal_bearing, distance,
                                         previous_direction=previous_direction)
        if selected is None:
            # No clear direction: stop instead of turning to search.
            self.stop(self.blocked_reason(points), keep_paths=True)
            return
        clearance = self._selection_radius
        if not self._selection_held:
            self.selected_course_heading_deg = wrap_deg(heading + math.degrees(selected))
        # Keep the original heading when holding; ray rounding must not slowly shift it.
        self.approach_heading_deg = wrap_deg(heading + math.degrees(selected))
        error_deg = abs(math.degrees(selected))
        turn = float(np.clip(self.p["turn_gain"] * selected,
                             -min(self.p["max_turn_rps"], 2 * self.max_translate_velocity),
                             min(self.p["max_turn_rps"], 2 * self.max_translate_velocity)))
        # Slow down near the goal and for sharper turns.
        speed = min(self.p["forward_speed_mps"], self.max_translate_velocity,
                    self.p["goal_speed_gain"] * goal_distance)
        speed *= max(0.0, math.cos(selected))
        if clearance < max(self.robot_radius, self.p["preferred_clearance_m"]):
            # Use a lower speed when there is no room for the extra clearance.
            speed = min(speed, self.p["reduced_clearance_speed_mps"])

        # Turn first for large angle errors; otherwise try a checked curve, then straight ahead.
        if error_deg >= self.p["moving_turn_limit_deg"]:
            reason = ("ALIGN: turn toward selected direction "
                      f"| angle={error_deg:.1f}deg >= {self.p['moving_turn_limit_deg']:g}deg")
        elif self.motion_is_clear(points, speed, turn, distance, radius=clearance):
            self.move_2D(x=speed, turn=turn)
            self.plot_direction(selected, distance)
            self.report_status("DRIVE: curve clear")
            return
        elif self.path_is_clear(points, 0.0, distance, radius=clearance):
            self.move_2D(x=speed)  # Drive straight when the curve is blocked.
            self.approach_heading_deg = heading
            self.plot_direction(0.0, distance)
            self.report_status("DRIVE: straight fallback; curve blocked")
            return
        else:
            # Both driving options are blocked; face the chosen opening first.
            reason = "ALIGN: curve blocked; turn toward selected direction"

        # Turning in place uses the same robot circle; check it before turning.
        if not self.path_is_clear(points, 0.0, 0.0):
            self.stop(self.blocked_reason(points), keep_paths=True)
            return
        # Keep the turn direction and limit its speed to the configured range.
        turn = float(np.sign(turn) * np.clip(
            abs(turn),
            self.p["min_turn_rps"],
            self.p["max_turn_rps"],
        ))
        self.move_2D(turn=turn)  # Turn in place; check forward movement again next update.
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

