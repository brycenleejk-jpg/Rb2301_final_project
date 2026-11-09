import argparse
import configparser
import math
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

# Navigation settings: start with the first section; the remaining values are setup/defaults.
PARAMS = {
    # TUNE THESE FIRST: goals, path preference, speed and switching margin.
    "first_segment_virtual_goal": -45.0, # Segment A direction from the start, degrees; negative = right.
    "second_segment_virtual_goal": 0.0, # Segment B direction from the start, degrees.
    "depth_weight": 1.5, # Increase to favour paths with more space ahead.
    "virtual_goal_weight": 1.5, # Increase to favour the current virtual-goal direction.
    "preferred_clearance_m": 0.17, # Preferred radius around the robot; larger = more clearance.
    "forward_speed_mps": 0.3, # Requested forward speed, m/s.
    "turn_gain": 1.0, # How strongly heading error produces a turn command.
    "switch_tolerance_m": 0.03, # Extra margin for leaving/re-entering the starting corridor.
    # USUALLY LEAVE AS IS: robot and LiDAR setup.
    "robot_radius_m": 0.11, # Measured collision radius; change only when correcting robot geometry.
    "lidar_yaw_deg": 180.0, # LiDAR mounting rotation; 180 means backward-facing.
    "lidar_x_m": 0.11423, # LiDAR forward offset from the robot centre, metres.
    "lidar_y_m": 0.0, # LiDAR left offset from the robot centre, metres.
    # USUALLY LEAVE AS IS: controller defaults; adjust for a specific issue.
    "check_distance_m": 0.3, # Lookahead distance for collision checks, metres.
    "depth_cap_m": 3.0, # Maximum depth used in scoring, metres.
    "view_half_angle_deg": 135.0, # Search this far left/right; 135 includes rear-side openings.
    "candidate_step_deg": 3.0, # Minimum angle between directions tested, degrees.
    "starting_side_sector_deg": 5.0, # Width-measurement rays within this angle of each side.
    "similar_score_tolerance": 0.03, # Prefer greater depth among scores within this margin.
    "moving_turn_limit_deg": 50.0, # At or above this error, turn in place instead of driving.
    "straight_drive_tolerance_deg": 5.0, # If nearly facing the chosen path, allow checked straight motion.
    "min_turn_rps": 0.2, # Minimum turn-in-place speed, rad/s.
    "max_turn_rps": 0.5, # Maximum requested turning speed, rad/s.
    "reduced_clearance_speed_mps": 0.1, # Forward-speed cap when only minimum clearance fits.
    "sensor_timeout_s": 0.5, # Stop after this long without a received pose or scan.
}

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

def wrap_deg(angle:float) -> float:
    '''Wrap an angle in degrees into [-180, 180).'''
    return (angle + 180.0) % 360.0 - 180.0

def optitrack_to_maze(x:float, y:float, heading_deg:float, frame:dict) -> tuple:
    '''Optitrack-frame pose (x, y, heading in degrees) -> CA2 maze-frame pose (see optitrack_variables.config).'''
    dx, dy = x - frame["origin_x"], y - frame["origin_y"]
    r = math.radians(frame["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy,
            wrap_deg(heading_deg - frame["rotation_deg"] - frame["heading_offset_deg"]))

def maze_to_course(x:float, y:float, heading_deg:float, course:dict) -> tuple:
    '''CA2 maze-frame pose -> final-project course frame (origin/rotation of the course in the maze frame, [course] in the config).'''
    dx, dy = x - course["origin_x"], y - course["origin_y"]
    r = math.radians(course["rotation_deg"])
    c, s = math.cos(r), math.sin(r)
    return (c * dx + s * dy, -s * dx + c * dy, wrap_deg(heading_deg - course["rotation_deg"]))

def load_frame(robot_number:int) -> tuple:
    '''Read optitrack_variables.config: returns (frame, course) for this robot.'''
    parser = configparser.ConfigParser()
    path = os.path.join(_PACKAGE_DIR, "optitrack_variables.config")
    if not parser.read(path):
        raise FileNotFoundError(f"Could not read {path}")
    frame = {k: parser.getfloat("frame", k) for k in ("origin_x", "origin_y", "rotation_deg")}
    raw = parser.get("frame", "heading_offset_deg", fallback="auto").strip().lower()
    # Rigid bodies are created facing +y of the maze frame, so Motive's zero heading is maze +y (90 deg)
    frame["heading_offset_deg"] = wrap_deg(-(frame["rotation_deg"] + 90.0)) if raw == "auto" else float(raw)
    if parser.has_section("heading_offsets") and parser.has_option("heading_offsets", str(robot_number)):
        frame["heading_offset_deg"] = parser.getfloat("heading_offsets", str(robot_number))
    course = {k: parser.getfloat("course", k, fallback=0.0) for k in ("origin_x", "origin_y", "rotation_deg")}
    return frame, course

class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True, calibrate:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")
        self.is_simulation = is_simulation # Operating mode selected at startup.
        ################ Our own attributes ################
        # SETTINGS: tune PARAMS above; each node receives its own copy.
        self.p = PARAMS.copy()
        # SETUP: derived from geometry/configuration; no routine tuning here.
        self.robot_radius = self.p["robot_radius_m"] # Minimum circular collision footprint.
        self.calibrate = calibrate # Command-line mode: print poses without driving.
        self.frame = self.course = None # Frame transforms loaded for the hardware pose source.
        # INTERNAL STATE: leave these initial values alone; callbacks update them automatically.
        # Sensor readings and calibration display.
        self.pose = None # Latest [x, y, heading_deg].
        self.last_scan = None # Full LaserScan: distances and angle metadata.
        self.last_scan_time = self.last_pose_time = -math.inf # Wait for both sensors before driving.
        self.raw_pose = None # Optitrack pose before frame conversion.
        self._ticks = 0 # Calibration display counter.
        # Starting corridor: measured once, then used for segment switching.
        self.segment = "A" # Current segment; updated automatically.
        self.width_checked = False # True after both side boundaries are measured.
        self.start_pose = None # Saved pose defining the fixed starting frame.
        self.starting_width = None # Initial left distance + right distance, metres.
        self.start_left_m = self.start_right_m = None # Measured side distances from robot centre.
        self.switch_right_m = None # Calculated threshold for switching to segment B.
        self.x_position = 0.0 # Displacement along the original forward axis, metres.
        self.y_position = 0.0 # Displacement along the original left axis; rightward is negative.
        # Collision clearance chosen from the current scan.
        self._selection_radius = self.robot_radius # Clearance used by selection and motion checks.
        # Diagnostic state only; no tuning needed, and no effect on movement decisions.
        self._log = {} # Created on first use by log_data().
        ################ ROS setup ################
        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data)
        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            # On the Bingda robots ROS_DOMAIN_ID is the robot's number, so the Optitrack topic is picked automatically
            domain = os.environ.get("ROS_DOMAIN_ID", "")
            self.robot_number = int(domain) if domain.isdigit() and 1 <= int(domain) <= 99 else 3
            self.frame, self.course = load_frame(self.robot_number)
            self.get_logger().info(f"Reading pose from /vrpn_mocap/bingda_{self.robot_number:03d}/pose")
            self.map_sub = self.create_subscription(
                PoseStamped, f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose', self.optitrack_callback, qos_profile)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.
        self.get_logger().info("Status: direction is relative to the robot (+left, -right); v=m/s, w=rad/s")

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.last_scan = msg # Keep distances and angle metadata together.
        self.last_scan_time = time.monotonic()

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        return self.store_pose(x, y, heading)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        return self.store_pose(latest_pose_msg.position.x, latest_pose_msg.position.y, heading)

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)
        # Observe the actual command, including reversals hidden between status samples.
        log = self.log_data()
        log["command"] = (twist_msg.linear.x, twist_msg.angular.z)
        sign = int(np.sign(twist_msg.angular.z))
        if sign and log["turn_sign"] and sign != log["turn_sign"]:
            log["reversals"] += 1
            direction = "right -> left" if sign > 0 else "left -> right"
            self.report_event("TURN CHANGE", f"{direction} | {log['reason']}")
        if sign:
            log["turn_sign"] = sign

    ################ Our own functions ################
    def log_data(self):
        """Diagnostics only: these counters and labels never control navigation."""
        if not getattr(self, "_log", None):
            self._log = {"status_time": -math.inf, "events": {}, "state": None,
                         "command": (0.0, 0.0), "turn_sign": 0, "reversals": 0,
                         "direction": None, "reason": "startup"}
        return self._log

    def report_event(self, kind, message):
        """Report decisions immediately; limit each event type to two lines per second."""
        now = time.monotonic()
        events = self.log_data()["events"]
        event = events.setdefault(kind, {"time": -math.inf, "pending": 0, "message": ""})
        event["message"] = message
        if now - event["time"] < 0.5:
            event["pending"] += 1 # Keep the latest reason; status output flushes it later.
            return
        extra = f" | {event['pending']} events suppressed" if event["pending"] else ""
        self.get_logger().info(f"{kind} | {message}{extra}")
        event.update(time=now, pending=0)

    def report_status(self, action, reason):
        """One compact line every 0.5 s; count fast reversals between printed samples."""
        log = self.log_data()
        now = time.monotonic()
        state = (action, reason.split(":")[0]) # Ignore changing numeric ages in stop reasons.
        changed = state != log["state"]
        log["state"], log["reason"] = state, reason
        if now - log["status_time"] < 0.5:
            if changed:
                self.report_event("STATE", f"{action} | {reason}")
            return
        # Flush a rate-limited event even if it stopped recurring before this status tick.
        for kind, event in list(log["events"].items()):
            if event["pending"] and now - event["time"] >= 0.5:
                self.report_event(kind, event["message"])
        log["status_time"] = now
        v, w = log["command"] # Speeds actually published, after clipping.
        if action == "STOP":
            message = f"STOP {self.segment} | {reason} | v={v:.2f} w={w:+.2f}"
        else:
            direction = log["direction"]
            aim = "none" if direction is None else f"{direction:+.1f}deg"
            message = (f"{action} {self.segment} | direction={aim} | v={v:.2f} w={w:+.2f}"
                       f" | {reason} | clearance={self._selection_radius:.2f}m")
        if log["reversals"]:
            message += f" | turn-direction-changes={log['reversals']}"
        self.get_logger().info(message)
        log["reversals"] = 0

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        values = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = np.linalg.norm(values)
        if not np.all(np.isfinite(values)) or not np.isfinite(norm) or norm < 1e-9:
            return math.nan
        x, y, z, w = values / norm
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def store_pose(self, x, y, heading):
        """Store a finite pose and record its reception time."""
        pose = np.array((x, y, heading), dtype=float)
        self.pose = pose if np.all(np.isfinite(pose)) else None
        if self.pose is not None:
            self.last_pose_time = time.monotonic()
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def virtual_bearing_deg(self):
        """Express the selected goal, fixed relative to the start, as a local bearing."""
        target = (self.p["first_segment_virtual_goal"] if self.segment == "A"
            else self.p["second_segment_virtual_goal"])
        heading_from_start = wrap_deg(self.pose[2] - self.start_pose[2])
        return wrap_deg(target - heading_from_start)

    def measurement_starting_width(self, points, angles):
        """Measure the initial side boundaries once, while the robot is stopped."""
        if self.width_checked:
            return True
        sector = math.radians(self.p["starting_side_sector_deg"])
        left = np.abs(angles - np.pi / 2) <= sector
        right = np.abs(angles + np.pi / 2) <= sector
        if not np.any(left) or not np.any(right):
            return False # Wait until both sides have valid readings.
        # Project onto the sideways axis, using the robot-centred obstacle points.
        left_distance = float(np.median(points[left, 1]))
        right_distance = float(np.median(-points[right, 1]))
        if not (math.isfinite(left_distance) and math.isfinite(right_distance)
                and left_distance > 0 and right_distance > 0):
            return False
        self.start_left_m, self.start_right_m = left_distance, right_distance
        self.starting_width = left_distance + right_distance
        self.switch_right_m = (right_distance + self.robot_radius
                               + self.p["switch_tolerance_m"])
        self.start_pose = self.pose.copy()
        self.width_checked = True
        self.get_logger().info(
            f"Starting width: {self.starting_width:.3f} m; "
            f"switch after {self.switch_right_m:.3f} m rightward")
        return True

    def rightward_displacement(self):
        """Update displacement in the starting frame; return positive rightward travel."""
        dx, dy = self.pose[:2] - self.start_pose[:2]
        heading = math.radians(self.start_pose[2])
        self.x_position = math.cos(heading) * dx + math.sin(heading) * dy
        self.y_position = -math.sin(heading) * dx + math.cos(heading) * dy
        return -self.y_position

    def update_segment(self):
        """Switch goals on leaving/re-entering the starting corridor, with a boundary buffer."""
        rightward = self.rightward_displacement() # Updates coordinates in the fixed starting frame.
        margin = self.robot_radius + self.p["switch_tolerance_m"]
        inside = -self.start_right_m + margin <= self.y_position <= self.start_left_m - margin
        old_segment = self.segment
        if self.segment == "A" and rightward >= self.switch_right_m:
            self.segment = "B"
        elif self.segment == "B" and inside:
            self.segment = "A"
        if self.segment != old_segment:
            reentry = self.start_right_m - margin
            self.report_event("SEGMENT", f"{old_segment} -> {self.segment} | right={rightward:.3f}m"
                              f" exit={self.switch_right_m:.3f}m reenter={reentry:.3f}m")

    def scan_geometry(self):
        '''Use LIDAR readings to calculate relative position of obstacle points'''
        scan = self.last_scan
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = (scan.angle_min + np.arange(len(ranges)) * scan.angle_increment # arange generates position number
                    + math.radians(self.p["lidar_yaw_deg"])) # Correct for the LiDAR mounting direction.
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        # Discard invalid readings; missing readings do not count as open space.
        valid = (np.isfinite(ranges) & np.isfinite(angles) & (ranges > 0) # Check range and angle is not infinite
                    & (ranges >= scan.range_min) & (ranges <= scan.range_max)) # range within max and min scan range
        ranges, angles = ranges[valid], angles[valid]
        points = np.column_stack((ranges * np.cos(angles) + self.p["lidar_x_m"], # Account for the LiDAR's position relative to the robot centre.
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
        """Check the robot's circular footprint along the proposed curved motion."""
        radius = self.robot_radius if radius is None else max(self.robot_radius, radius)
        if speed <= 0.0:
            return self.path_is_clear(points, 0.0, 0.0, radius=radius)
        if abs(turn) < 1e-6:
            return self.path_is_clear(points, 0.0, distance, radius=radius)
        if len(points) == 0:
            return False
        # Sample travel along the curve at most 2 cm apart.
        count = max(1, int(math.ceil(distance / 0.02)))
        travel = np.linspace(0.0, distance, count + 1)
        phi = turn * travel / speed
        centres = np.column_stack((speed / turn * np.sin(phi),
                                   speed / turn * (1.0 - np.cos(phi))))
        separation = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        padding = distance / (2 * count) # Cover gaps between sampled positions.
        return bool(np.all(separation > radius + padding))

    def select_direction(self, points, angles, goal_bearing, distance):
        """Choose a clear direction using only the current scan and virtual goal."""
        log = self.log_data()
        log["direction"] = None # Display only; never used to choose a direction.
        # Step 1: Keep current LiDAR rays within the search sector.
        half_angle = math.radians(self.p["view_half_angle_deg"])
        view = np.sort(angles[np.abs(angles) <= half_angle])
        if len(points) == 0 or len(view) == 0:
            log["selection_reason"] = "no-valid-points" if len(points) == 0 else "no-rays-in-view"
            return None
        # Step 2: Test spaced-out rays, plus rays nearest the goal and straight ahead.
        step = math.radians(self.p["candidate_step_deg"])
        candidates = []
        for angle in view:
            if not candidates or angle - candidates[-1] >= step:
                candidates.append(float(angle))
        goal_error = (view - goal_bearing + np.pi) % (2 * np.pi) - np.pi
        # Our 3-degree shortlist may skip these useful angles, so include both explicitly.
        closest_to_goal = float(view[np.argmin(np.abs(goal_error))])
        closest_to_forward = float(view[np.argmin(np.abs(view))])
        candidates.append(closest_to_goal) # Also test the ray nearest the virtual goal.
        candidates.append(closest_to_forward) # Also test the ray nearest straight ahead.
        candidates = sorted(set(candidates))
        # Step 3: Try preferred clearance first, then the minimum robot radius.
        preferred = max(self.robot_radius, self.p["preferred_clearance_m"])
        tolerance = max(0.0, self.p["similar_score_tolerance"])
        for radius in dict.fromkeys((preferred, self.robot_radius)):
            choices = []
            for angle in candidates:
                # Step 4: Reject paths that cannot fit the whole robot.
                if not self.path_is_clear(points, angle, distance, radius=radius):
                    continue
                # Step 5: Score the clear paths using depth and goal alignment.
                depth = self.path_depth(points, angle)
                score = (self.p["depth_weight"] * depth / self.p["depth_cap_m"]
                         + self.p["virtual_goal_weight"] * math.cos(angle - goal_bearing))
                choices.append((angle, score, depth))
            if not choices:
                continue
            # Step 6: Among similar scores, prefer depth, then score and less turning.
            best_score = max(score for angle, score, depth in choices)
            near_best = [c for c in choices if best_score - c[1] <= tolerance + 1e-12]
            selected = max(near_best, key=lambda c: (c[2], c[1], -abs(c[0])))[0]
            # Step 7: Return this scan's direction and use the same radius for movement.
            last_radius = log.get("radius")
            if last_radius is not None and last_radius != radius:
                self.report_event("CLEARANCE", f"{last_radius:.2f} -> {radius:.2f}m")
            log["radius"] = self._selection_radius = radius
            return selected
        log["selection_reason"] = "all-paths-blocked"
        return None

    def stop(self, reason="requested"):
        """Publish zero velocity and explain why we stopped."""
        self.log_data().update(direction=None, reason=reason)
        self.move_2D()
        self.report_status("STOP", reason)

    def drive_toward(self, points, selected, distance):
        """Recheck the current scan's direction; turn for large angles or blocked curves."""
        clearance = self._selection_radius
        self.log_data()["direction"] = math.degrees(selected) # Current command display only.
        error_deg = abs(math.degrees(selected))
        turn_limit = min(self.p["max_turn_rps"], 2 * self.max_translate_velocity)
        turn = float(np.clip(self.p["turn_gain"] * selected, -turn_limit, turn_limit))
        speed = min(self.p["forward_speed_mps"], self.max_translate_velocity)
        speed *= max(0.0, math.cos(selected))
        if clearance < max(self.robot_radius, self.p["preferred_clearance_m"]):
            speed = min(speed, self.p["reduced_clearance_speed_mps"])
        # Only the direction from this scan and its collision check decide movement.
        if error_deg < self.p["moving_turn_limit_deg"]:
            if self.motion_is_clear(points, speed, turn, distance, radius=clearance):
                self.log_data()["reason"] = "curve-clear"
                self.move_2D(x=speed, turn=turn)
                self.report_status("DRIVE", "curve-clear")
                return
        # A tiny turn can fail the padded curve check even when straight ahead is clear.
        # Near forward, check the exact straight path at the SAME clearance before turning.
        if error_deg <= self.p["straight_drive_tolerance_deg"]:
            if self.path_is_clear(points, 0.0, distance, radius=clearance):
                self.log_data()["reason"] = "straight-clear"
                self.move_2D(x=speed, turn=0.0)
                self.report_status("DRIVE", "straight-clear")
                return
            # Already nearly aligned: forcing minimum-speed left/right turns will not
            # establish a clear forward path. Stop and reassess on the next scan instead.
            self.stop("near-forward-path-blocked")
            return
        if not self.path_is_clear(points, 0.0, 0.0):
            self.stop("footprint-blocked")
            return
        reason = "large-angle" if error_deg >= self.p["moving_turn_limit_deg"] else "curve-blocked"
        # Flag minimum-speed clamping: a tiny remaining error may otherwise overshoot.
        if 0 < abs(turn) < self.p["min_turn_rps"]:
            reason += "/min-turn"
        self.log_data()["reason"] = reason
        turn = float(np.sign(turn) * np.clip(abs(turn), self.p["min_turn_rps"], turn_limit))
        self.move_2D(turn=turn)
        self.report_status("TURN", reason)

    def timer_callback(self):
        """Measure once, update the segment, select a direction, then check movement."""
        # Step 1: Stop if either sensor is missing, invalid or no longer updating.
        now = time.monotonic()
        if (self.pose is None or not np.all(np.isfinite(self.pose))
                or self.last_scan is None
                or now - self.last_pose_time > self.p["sensor_timeout_s"]
                or now - self.last_scan_time > self.p["sensor_timeout_s"]):
            if self.pose is None or not np.all(np.isfinite(self.pose)):
                reason = "pose-missing-or-invalid"
            elif self.last_scan is None:
                reason = "scan-missing"
            elif now - self.last_pose_time > self.p["sensor_timeout_s"]:
                reason = f"pose-stale:{now - self.last_pose_time:.2f}s"
            else:
                reason = f"scan-stale:{now - self.last_scan_time:.2f}s"
            self.stop(reason)
            return
        # Step 2: Calibration always publishes zero velocity, including in simulation.
        if self.calibrate:
            self.stop("calibration")
            self._ticks += 1
            if self._ticks % 10 == 0 and self.raw_pose is not None:
                self.get_logger().info(f"Optitrack: {self.raw_pose}; maze: {self.pose}")
            return
        # Step 3: Convert valid readings to robot-centred obstacle positions.
        points, angles = self.scan_geometry()
        if len(points) == 0:
            self.stop("no-valid-scan-returns")
            return
        # Step 4: Wait for both side measurements before driving. Save them once.
        if not self.measurement_starting_width(points, angles):
            self.stop("waiting-for-side-measurements")
            return
        # Step 5: Update A/B using exit and re-entry thresholds in the starting frame.
        self.update_segment()
        # Step 6: Select a clear direction toward the current virtual goal.
        distance = self.p["check_distance_m"]
        goal_bearing = math.radians(self.virtual_bearing_deg())
        selected = self.select_direction(points, angles, goal_bearing, distance)
        if selected is None:
            self.stop(self.log_data()["selection_reason"])
            return
        # Step 7: Check the actual motion before publishing its speeds.
        self.drive_toward(points, selected, distance)
        ###### INSERT CODE HERE ######

def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course")
    rclpy.init(args=ros_args)
    lidar = obstacle = None
    try:
        # Use lidar scan length to determine if in simulation or not.
        lidar = LidarNode()
        while rclpy.ok() and lidar.is_simulation is None:
            rclpy.spin_once(lidar, timeout_sec=0.2)
        if rclpy.ok():
            obstacle = ObstacleCourseNode(lidar.is_simulation, cli_args.calibrate)
            lidar.destroy_node()
            lidar = None
            rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if obstacle is not None:
            obstacle.timer.cancel()
            if rclpy.ok():
                obstacle.move_2D() # Send a stop command before destroying the publisher.
            obstacle.destroy_node()
        if lidar is not None:
            lidar.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
