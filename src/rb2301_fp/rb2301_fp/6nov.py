"""LiDAR depth navigation with a virtual direction and measured turn tracking.

Both goal directions are fixed relative to the original starting heading:
45 degrees right in segment 1, then original forward (0 degrees) in segment 2. Switch after
the configured net measured right turn, compensating both bearings for rotation.
Use OptiTrack orientation on hardware and odometry in simulation; positions are ignored.
There is no automatic arrival test: stop externally at the course finish.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# Navigation settings (distances in metres, angles in degrees).
PARAMS = {
    "virtual_goal_initial_deg": -45.0, # From starting heading: negative right, positive left.
    "virtual_goal_after_turn_deg": 0.0, # Original forward, fixed relative to starting heading.
    "segment_right_turn_deg": 90.0,    # Net clockwise rotation since startup.
    "segment_confirm_samples": 3,     # Require consecutive fresh orientation samples.
    "virtual_goal_weight": 1.5,        # Weight of cos(candidate - virtual bearing).
    "alignment_release_deg": 8.0,     # Hold a clear selected turn until this close.
    "max_measured_yaw_rate_deg_s": 120.0, # Reject implausible sensor jumps.
    "yaw_jump_allowance_deg": 3.0,     # Allow small orientation measurement noise.
    "check_distance_m": 0.50,          # How far ahead to check for obstacles.
    "candidate_step_deg": 3.0,         # Smaller steps test more directions.
    "view_half_angle_deg": 135.0,      # Search +/-135 degrees: 270 total, excluding rear 90.
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
    "turn_gain": 1.0,                  # Turn rate = gain * angle error in radians.
    "min_turn_rps": 0.20,              # Minimum turn rate when turning in place (rad/s).
    "max_turn_rps": 0.50,              # Maximum turn rate (rad/s).
    "moving_turn_limit_deg": 50.0,     # Turn in place at or above this angle error.
    "sensor_timeout_s": 0.50,          # Stop if scan or orientation is older than this.
}

def wrap_deg(angle):
    """Keep a degree angle in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


class LidarNode(Node):
    """Identify simulation or real hardware from the number of scan readings."""
    def __init__(self):
        super().__init__("LidarNode")
        self.is_simulation = None
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.callback, qos_profile_sensor_data)

    def callback(self, msg):
        self.is_simulation = len(msg.ranges) <= 360


class ObstacleCourseNode(Node):
    def __init__(self, is_simulation=True, calibrate=False, heading_source=None):
        super().__init__("obstaclecourse")
        self.p = PARAMS.copy()
        self.is_simulation = is_simulation
        self.calibrate = calibrate
        self.max_translate_velocity = 1.4 if is_simulation else 0.3
        self.robot_radius = self.p["robot_radius_m"]  # Circle used to represent the robot.
        self.scan_msg = None
        self.last_scan_time = 0.0
        self.heading_deg = None  # Unwrapped yaw relative to the first sample, never a world heading.
        self._last_sensor_yaw_deg = None
        self.last_heading_time = 0.0
        self._last_heading_stamp_ns = 0
        self._heading_fault = None
        self.net_right_turn_deg = 0.0
        self.segment = 1
        self._switch_confirmations = 0
        self.selected_heading_deg = None
        self._alignment_active = False

        self.sub_scan = self.create_subscription(
            LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data)
        self.heading_source = heading_source or ("odom" if is_simulation else "optitrack")
        if self.heading_source == "optitrack":
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            topic = f"/vrpn_mocap/bingda_{number:03d}/pose"
            self.subscription = self.create_subscription(
                PoseStamped, topic, self.optitrack_callback, qos_profile_sensor_data)
        elif self.heading_source == "odom":
            topic = "odom"
            self.subscription = self.create_subscription(
                Odometry, topic, self.odometer_callback, qos_profile_sensor_data)
        else:
            raise ValueError("heading_source must be optitrack or odom")
        self.get_logger().info(
            f"Virtual goal: reading {self.heading_source} orientation only from {topic}; starting 45deg right")
        self.publisher_ = self.create_publisher(Twist, "cmd_vel", 10)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def sub_scan_callback(self, msg):
        self.scan_msg = msg  # Save distances and scan angles together.
        self.last_scan_time = time.monotonic()

    def odometer_callback(self, msg):
        # Deliberately never read msg.pose.pose.position or any transform.
        self.update_orientation(msg.pose.pose.orientation, msg.header.stamp)

    def optitrack_callback(self, msg):
        # Only orientation differences are used: no position or course calibration.
        self.update_orientation(msg.pose.orientation, msg.header.stamp)

    def heading_fault(self, reason):
        """Latch a loss of turn-tracking confidence; restarting establishes a new run."""
        self._heading_fault = reason
        self.selected_heading_deg = None
        self._alignment_active = False

    def update_orientation(self, q, stamp=None):
        """Validate fresh measured yaw and accumulate signed rotation, not commands."""
        if self._heading_fault:
            return
        now = time.monotonic()
        stamp_ns = 0 if stamp is None else stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp is not None and self._last_heading_stamp_ns > 0:
            if stamp_ns == self._last_heading_stamp_ns:
                return  # A duplicate cannot refresh the sensor or confirm the switch.
            if stamp_ns < self._last_heading_stamp_ns:
                self.heading_fault("orientation timestamp moved backwards")
                return
        components = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = float(np.linalg.norm(components))
        if not np.all(np.isfinite(components)) or not math.isfinite(norm) or norm < 1e-9:
            self.heading_fault("invalid orientation quaternion")
            return
        x, y, z, w = components / norm
        yaw = math.degrees(math.atan2(2 * (w * z + x * y),
                                      1 - 2 * (y * y + z * z)))
        if self._last_sensor_yaw_deg is None:
            self.heading_deg = 0.0  # Initial absolute yaw is irrelevant.
        else:
            elapsed = now - self.last_heading_time
            source_elapsed = ((stamp_ns - self._last_heading_stamp_ns) / 1e9
                              if stamp_ns > 0 and self._last_heading_stamp_ns > 0 else elapsed)
            if max(elapsed, source_elapsed) > self.p["sensor_timeout_s"]:
                self.heading_fault("orientation gap; accumulated turn is uncertain")
                return
            delta = wrap_deg(yaw - self._last_sensor_yaw_deg)
            limit = (self.p["max_measured_yaw_rate_deg_s"] * max(0.0, source_elapsed)
                     + self.p["yaw_jump_allowance_deg"])
            if abs(delta) > limit:
                self.heading_fault("implausible orientation jump")
                return
            self.heading_deg += delta
        self._last_sensor_yaw_deg = yaw
        self.last_heading_time = now
        self._last_heading_stamp_ns = stamp_ns
        self.net_right_turn_deg = -self.heading_deg
        if self.segment == 1:
            if self.net_right_turn_deg >= self.p["segment_right_turn_deg"] - 1e-9:
                self._switch_confirmations += 1
            else:
                self._switch_confirmations = 0
            if self._switch_confirmations >= max(1, self.p["segment_confirm_samples"]):
                self.segment = 2
                self.selected_heading_deg = None
                self._alignment_active = False
                self.get_logger().info(
                    f"SEGMENT: goal={self.p['virtual_goal_after_turn_deg']:+.1f}deg from start "
                    f"| net_right={self.net_right_turn_deg:.1f}deg")

    def virtual_bearing_deg(self):
        """Convert a goal fixed in the starting frame into a robot-relative bearing."""
        target = (self.p["virtual_goal_initial_deg"] if self.segment == 1
                  else self.p["virtual_goal_after_turn_deg"])
        return wrap_deg(target - self.heading_deg)

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
        if speed <= 0.0: # No forward movement, simply check path_is_clear
            return self.path_is_clear(points, 0.0, 0.0)
        radius = self.robot_radius if radius is None else max(self.robot_radius, radius)
        if abs(turn) < 1e-6: # No turn movement, simply check path_is_clear
            return self.path_is_clear(points, 0.0, distance, radius=radius)
        if len(points) == 0: # Reject scan with no usuable obstacle points
            return False
        count = max(1, int(math.ceil(distance / 0.02)))  # Count = Number of intervals along path to checl, Check at most 2cm intervals along curved path
        travel = np.linspace(0.0, distance, count + 1) # Generate distance along curved path
        # Predict robot-centre positions along the curve.
        phi = turn * travel / speed # time = travel / speed, rotation = turn * time
        centres = np.column_stack((speed / turn * np.sin(phi),
                                   speed / turn * (1.0 - np.cos(phi)))) # Calcuate centre of arc
        separation = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        padding = distance / (2 * count)  # Padding = half the interval length.
        return bool(np.all(separation > radius + padding)) # Every obstacle must be farther than radius + padding from every predicted centre.

    def select_direction(self, points, angles, goal_bearing, distance, radius=None,
                         previous_direction=None, hold_previous=False):
        """Score clear directions by depth and alignment with a virtual bearing."""
        preferred = max(self.robot_radius, self.p["preferred_clearance_m"]) # preferred is a soft limit (larger than robot radius)
        # Try extra space first; a retry may use the robot radius, but never less.
        radius = preferred if radius is None else max(self.robot_radius, radius)
        self._selection_radius = radius  # Use the same radius for the motion check.
        # Honour the configured sector, including directions behind either side.
        half_angle = math.radians(float(np.clip(self.p["view_half_angle_deg"], 0.0, 180.0)))
        view = np.sort(angles[np.abs(angles) <= half_angle + 1e-9])
        self._candidate_count = 0  # Reset the number of tested directions.
        self._selection_summary = None
        self._selection_held = False
        if len(view) == 0:
            # No valid readings in the search sector means no direction can be selected.
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
        # Include the remembered direction itself when nearby measurements support it.
        # Check its actual path; a nearby clear ray alone must not retain a blocked heading.
        if (previous_direction is not None
                and abs(previous_direction) <= half_angle + 1e-9
                and np.min(np.abs(view - previous_direction)) <= step + 1e-9):
            candidates.append(float(previous_direction))
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
            score = (self.p["depth_weight"] * depth_score
                     + self.p["virtual_goal_weight"] * alignment)
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
                                         previous_direction=previous_direction,
                                         hold_previous=hold_previous)
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
                # Keep the previous clear direction only while its score
                # remains within the tolerance of the best, even during alignment.
                if (abs(candidates[previous] - previous_direction) <= step + 1e-9
                        and clear[previous]
                        and previous in near_best):
                    chosen = previous
                    self._selection_held = True
            selected = candidates[chosen]
            # A retained/deeper choice can score slightly below the raw winner.
            # Show the strongest OTHER option; gap remains the raw top-two gap.
            alternative = runner_direction if selected == best_direction else best_direction
            alternative_key = runner_key if selected == best_direction else best_key
            choice = ("alignment" if self._selection_held and hold_previous
                      else "kept" if self._selection_held else "depth")
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
        """Display only an angular virtual goal, never an invented goal position."""
        now = time.monotonic()
        state = (self.segment, self.heading_deg is not None and not self._heading_fault
                 and now - self.last_heading_time <= self.p["sensor_timeout_s"])
        if (state == getattr(self, "_goal_plot_state", None)
                and now - getattr(self, "_goal_plot_time", 0.0) < 0.5):
            return
        self._goal_plot_state, self._goal_plot_time = state, now
        if not state[1]:
            self.publish_plot("Goal: none")
            return
        self.publish_plot(
            f"Goal: bearing={self.virtual_bearing_deg():+.1f} deg, segment={self.segment}, "
            f"net_right={self.net_right_turn_deg:+.1f} deg, radius={self.robot_radius:.3f} m")

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
                       "radius": self.robot_radius, "inside": inside.tolist(),
                       "segment": self.segment, "virtual_bearing_deg": self.virtual_bearing_deg()}
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
            bearing = self.virtual_bearing_deg() if self.heading_deg is not None else self.p["virtual_goal_initial_deg"]
            message += (f" | segment={self.segment} virtual={bearing:+.1f}deg "
                        f"net_right={self.net_right_turn_deg:+.1f}deg")
            selection = getattr(self, "_selection_summary", None)
            if message.startswith(("DRIVE", "ALIGN:")) and selection:
                message += " | " + selection
            self.get_logger().info(message)

    def stop(self, reason, keep_paths=False):
        self.selected_heading_deg = None  # Reconsider direction after a full stop.
        self._alignment_active = False
        self.move_2D()
        self.plot_direction()  # Clear the displayed target.
        if not keep_paths:
            self.plot_paths()
        self.report_status(reason)

    def timer_callback(self):
        self._selection_summary = None  # Clear the previous update's selection log.
        self.plot_goal()
        now = time.monotonic()
        if self._heading_fault:
            self.stop(f"STOP: {self._heading_fault}; restart controller after checking source")
            return
        if (self.heading_deg is None or not math.isfinite(self.heading_deg)
                or now - self.last_heading_time > self.p["sensor_timeout_s"]):
            self.stop("STOP: orientation missing, invalid or stale")
            return
        if self.calibrate:
            self.stop("MONITOR: virtual goal and signed turn; no driving")
            return
        if (self.scan_msg is None
                or now - self.last_scan_time > self.p["sensor_timeout_s"]):
            self.stop("STOP: scan missing or stale")
            return
        points, angles = self.scan_geometry()
        if len(points) == 0:
            self.stop(self.blocked_reason(points))
            return

        heading = self.heading_deg
        goal_bearing = math.radians(self.virtual_bearing_deg())
        distance = self.p["check_distance_m"]
        previous_direction = (None if self.selected_heading_deg is None else
                              math.radians(wrap_deg(self.selected_heading_deg - heading)))
        if (previous_direction is None or abs(math.degrees(previous_direction))
                <= self.p["alignment_release_deg"]):
            self._alignment_active = False
        selected = self.select_direction(points, angles, goal_bearing, distance,
                                         previous_direction=previous_direction,
                                         hold_previous=self._alignment_active)
        if selected is None:
            # No clear direction: stop instead of turning to search.
            self.stop(self.blocked_reason(points), keep_paths=True)
            return
        clearance = self._selection_radius
        if not self._selection_held:
            self.selected_heading_deg = wrap_deg(heading + math.degrees(selected))
            self._alignment_active = False
        # Keep the original heading when holding; ray rounding must not slowly shift it.
        error_deg = abs(math.degrees(selected))
        turn = float(np.clip(self.p["turn_gain"] * selected,
                             -min(self.p["max_turn_rps"], 2 * self.max_translate_velocity),
                             min(self.p["max_turn_rps"], 2 * self.max_translate_velocity)))
        # Slow down for sharper turns; no goal distance is known.
        speed = min(self.p["forward_speed_mps"], self.max_translate_velocity)
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
        self._alignment_active = True
        self.move_2D(turn=turn)  # Turn in place; check forward movement again next update.
        self.plot_direction(selected, distance)
        self.report_status(reason)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true", help="Monitor virtual bearing and signed rotation without driving")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulation", action="store_true", help="Use simulation velocity limits")
    mode.add_argument("--hardware", action="store_true", help="Use hardware velocity limits")
    parser.add_argument("--heading-source", choices=("optitrack", "odom"),
                        help="Orientation only; default: optitrack on hardware, odom in simulation")
    cli, ros_args = parser.parse_known_args(args=args)
    rclpy.init(args=ros_args)
    lidar = None
    obstacle = None
    try:
        simulation = True if cli.simulation else False if cli.hardware else None
        if simulation is None:
            lidar = LidarNode()
            while rclpy.ok() and lidar.is_simulation is None:
                rclpy.spin_once(lidar, timeout_sec=0.2)
            simulation = lidar.is_simulation
        if rclpy.ok():
            obstacle = ObstacleCourseNode(simulation, cli.calibrate, cli.heading_source)
            rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if obstacle is not None:
            obstacle.timer.cancel()
            if rclpy.ok():
                obstacle.move_2D()
            obstacle.destroy_node()
        if lidar is not None:
            lidar.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

