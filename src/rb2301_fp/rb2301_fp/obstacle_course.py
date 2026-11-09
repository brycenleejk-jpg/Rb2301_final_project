import argparse
import configparser
import json
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
from std_msgs.msg import String

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG

# Default: stationary preview. Use --step for one half-lookahead advance.
PARAMS = {
    "goal_a_deg": -45.0,                # Fixed course heading, not relative to startup.
    "goal_b_deg": 0.0,                  # Fixed course +x direction.
    "switch_distance_multiplier": 3.0,  # Switch after this * initial right-wall distance rightward.
    "wall_sector_deg": 5.0,             # Measure around fixed course-right (-course y).
    "candidate_step_deg": 3.0,          # Sample the current -120 to +120 degree search.
    "check_distance_m": 0.3,            # Distance the robot centre would travel.
    "robot_radius_m": 0.11,             # Hard clearance used to reject paths.
    "soft_clearance_m": 0.14,           # Prefer this clearance when available.
    "curve_clearance_m": 0.16,          # Moving-curve radius only; clamped to robot_radius_m.
    "soft_clearance_bonus": 0.3,       # Score bonus; hard-only paths still compete.
    "side_clearance_weight": 0.50,      # Extra reward from hard to soft radius; 0 disables it.
    "depth_weight": 2.5,
    "virtual_goal_weight": 0.75,
    "depth_cap_m": 3.0,
    "similar_score_tolerance": 0.03,    # Within this score band, choose more depth.
    "forward_speed_mps": 0.30,
    "max_turn_rps": 0.4,
    "sim_min_turn_rps": 0.15,          # Gazebo only: avoid ineffective tiny turn commands.
    "heading_tolerance_deg": 3.0,      # Inside this angle, drive straight.
    "moving_turn_limit_deg": 45.0,     # Larger errors need a turn in place first.
    "motion_prediction_s": 0.5,        # Recheck this short moving curve every timer tick.
    "trail_spacing_m": 0.05,           # Save a position after this much displacement.
    "trail_radius_m": 0.10,            # Width of remembered red routes, measured from their lines.
    "visited_check_start_m": 0.20,     # Start comparing travelled history this far from the robot.
    "visited_check_distance_m": 0.80,  # End the comparison this far from the robot, unless obstructed.
    "visited_radius_m": 0.10,          # A sample counts as visited within this distance of an old position.
    "visited_penalty_weight": 2.20,    # Subtract weight * visited fraction at selection; 0 disables it.
    "red_return_angle_deg": 45.0,       # Reject travel within this angle toward a red segment's failed end.
    "turn_clearance_m": 0.21,           # Encloses Gazebo geometry around its rearward pose origin.
    "recheck_turn_space_during_turn": False,  # Initial check always runs; False ignores later clearance changes.
    "turn_timeout_s": 20.0,             # Allow a complete half-turn, including slow final alignment.
    "blockage_confirm_s": 0.5,         # No front candidate for this long confirms blockage.
    "retreat_speed_mps": 0.2,         # Straight reverse speed for one-dot / multi-dot tests.
    "retreat_reach_m": 0.01,           # Stop within 1 cm of that dot (not the trail circle).
    "retreat_timeout_s": 4.0,          # Maximum time per dot; resets when the target changes.
    "retreat_turn_limit_deg": 45.0,    # Curved test: larger rear-heading errors align in place.
    "retreat_curve_timeout_s": 15.0,    # Curved test: per-dot allowance for aligning and reversing.
    "sensor_timeout_s": 0.5,           # Movement stops if scan or pose stops updating.
    "lidar_yaw_deg": 180.0,
    "lidar_x_m": 0.11423,
    "lidar_y_m": 0.0,
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
    def __init__(self, is_simulation:bool=True, calibrate:bool=False, move:bool=False, single_step:bool=False,
                 backtrack_one:bool=False, backtrack_dots:int=0, retreat_curves:bool=False,
                 recover:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        self.p = PARAMS.copy()
        self.robot_radius = self.p["robot_radius_m"]
        self.last_scan = None
        self.start_pose = None
        self.relative_position = np.zeros(2)
        self.trail = []  # Ordered course-frame positions; revisits remain separate entries.
        self.visited_positions = []  # All travelled positions; recovery never trims exploration history.
        self.visited_display = {}  # Fresh overlap diagnostics; never used to select a heading.
        self.right_distance = None
        self.switch_x = None  # Threshold along the viewer's rightward/lateral x axis.
        # Course segment chooses the scoring goal: A = -45 deg, B = 0 deg by default.
        # Current position selects A or B, including when recovery crosses back into A.
        self.segment = "A"
        self.motion_enabled = move or single_step or backtrack_one or backtrack_dots > 0 or recover
        self.single_step = single_step
        self.backtrack_one = backtrack_one  # Preserve the existing single-dot test flag.
        self.retreat_limit = 1 if backtrack_one else backtrack_dots  # 0: detect only; N: reach up to N dots.
        self.retreat_curves = retreat_curves or recover
        self.recover = recover  # Reverse, turn around where clear, then resume front scoring.
        self.red_segments = []  # Course-frame (older, failed-side) endpoints; retained across recoveries.
        self.red_last_position = None  # Last position recorded during this reverse episode.
        self.turn_heading = self.turn_started = None
        self.blocked_since = None  # Time the first no-candidate result was seen; None means not waiting.
        self.recovery_status = ""  # Remember why a held stop happened for later timer/display updates.
        self.retreat_index = None  # Current target; decreases by one after each reached dot.
        self.retreat_started = None  # Start time for the CURRENT dot's timeout.
        self.trail_reached = []  # Reached retreat targets, drawn red; the trail stays intact.
        # SELECT a heading, then TURN / CURVE / DRIVE until half-lookahead progress.
        # --move repeats from SELECT; --step finishes in DONE. Default is preview only.
        self.phase = "SELECT"
        self.locked_heading = None  # Fixed course heading kept through TURN / CURVE / DRIVE.
        self.drive_start = None  # Course position saved when forward driving first starts.
        self.advance_progress = 0.0  # Metres travelled along the locked course heading.
        self.completed_steps = 0  # Display/log counter; does not choose directions.
        self.last_scan_time = self.last_pose_time = -math.inf

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, qos_profile_sensor_data) # Subscribe to LiDAR scan data

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
                PoseStamped,
                f'/vrpn_mocap/bingda_{self.robot_number:03d}/pose',
                self.optitrack_callback,
                qos_profile
                )

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node
        self.preview_publisher = self.create_publisher(String, 'navigation/paths', 2)
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None

    def sub_scan_callback(self, msg):
        """Keep the ranges together with their actual scan angles."""
        self.last_scan = msg
        self.last_scan_time = time.monotonic()

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.raw_pose = np.array((x,y,heading))
        x, y, heading = optitrack_to_maze(x, y, heading, self.frame) # CA2 maze frame, see optitrack_variables.config
        self.pose = np.array((x,y,heading))
        self.last_pose_time = time.monotonic()
        return self.pose

    def to_course(self, pose=None):
        '''Convert a pose [x, y, heading] (default: self.pose) from the CA2 maze frame to the final-project course frame. Optional; see [course] in optitrack_variables.config'''
        pose = self.pose if pose is None else pose
        return np.array(maze_to_course(pose[0], pose[1], pose[2], self.course)) if self.course else np.array(pose)

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        self.last_pose_time = time.monotonic()
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    ################ Navigation logic ################
    def scan_geometry(self):
        """Convert LiDAR returns to obstacle points relative to the robot centre."""
        scan = self.last_scan
        ranges = np.asarray(scan.ranges, dtype=float)
        angles = (scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
                  + math.radians(self.p["lidar_yaw_deg"]))
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        # isfinite rejects +inf, -inf and NaN; also reject nonpositive/out-of-range returns.
        # An inf ray is omitted, rather than replaced with a large free-space distance.
        valid = (np.isfinite(ranges) & (ranges > 0)
                 & (ranges >= scan.range_min) & (ranges <= scan.range_max))
        # Apply the SAME mask to ranges and angles so each remaining pair stays matched.
        ranges, angles = ranges[valid], angles[valid]
        points = np.column_stack((ranges * np.cos(angles) + self.p["lidar_x_m"],
                                  ranges * np.sin(angles) + self.p["lidar_y_m"]))
        return points, angles

    def measure_start(self, points):
        """Save the start and fixed right-wall distance (-course y), once."""
        if self.start_pose is not None:
            return True
        pose = self.to_course()
        heading = math.radians(pose[2])
        course_x = math.cos(heading) * points[:, 0] - math.sin(heading) * points[:, 1]
        course_y = math.sin(heading) * points[:, 0] + math.cos(heading) * points[:, 1]
        rightward = -course_y
        wall = ((rightward > 0)
                & (np.abs(np.arctan2(course_x, rightward)) <= math.radians(self.p["wall_sector_deg"])))
        if not np.any(wall):
            return False
        self.start_pose = pose.copy()
        self.right_distance = float(np.median(rightward[wall]))
        self.switch_x = self.p["switch_distance_multiplier"] * self.right_distance
        return True

    def update_segment(self):
        """Choose the virtual goal from the current side of the lateral boundary."""
        pose = self.to_course()
        self.relative_position = pose[:2] - self.start_pose[:2]
        rightward = -self.relative_position[1]
        previous = self.segment
        self.segment = "B" if rightward >= self.switch_x else "A"
        if self.segment != previous:
            self.get_logger().info(
                f"SEGMENT: {previous} -> {self.segment} | rightward={rightward:.3f}m "
                f"switch_x={self.switch_x:.3f}m goal={self.virtual_heading_deg():+.1f}deg")

    def virtual_heading_deg(self):
        return self.p["goal_a_deg"] if self.segment == "A" else self.p["goal_b_deg"]

    def record_trail(self):
        """Save actual course positions during translation; turning adds no dots."""
        position = self.to_course()[:2]
        # Separate from the active backtracking trail: keep forward AND reverse history.
        # A repeated visit may add a position, but overlap counts each sample only once.
        if (not self.visited_positions or
                (self.phase in ("DRIVE", "CURVE", "RETREAT") and
                 np.linalg.norm(position - self.visited_positions[-1]) >= self.p["trail_spacing_m"])):
            self.visited_positions.append(position.copy())
        if self.recover and self.phase == "RETREAT":
            self.record_red_position(position)
            return
        if not self.trail:
            self.trail.append(position.copy())
        elif self.phase in ("DRIVE", "CURVE"):
            distance = float(np.linalg.norm(position - self.trail[-1]))
            if distance >= self.p["trail_spacing_m"]:
                self.trail.append(position.copy())

    def virtual_bearing_deg(self):
        """Express the fixed course heading relative to the robot's current nose."""
        return wrap_deg(self.virtual_heading_deg() - self.to_course()[2])

    def path_is_clear(self, points, heading, distance, radius=None):
        """Check whether the robot's circle can move along a straight path."""
        if len(points) == 0:
            return False
        radius = self.robot_radius if radius is None else max(self.robot_radius, radius)
        c, s = math.cos(heading), math.sin(heading)
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        # Find each obstacle's closest point on the proposed centre-line segment.
        # Every obstacle must be farther than the radius from that whole segment.
        closest = np.clip(along, 0.0, distance)
        return bool(np.all(np.hypot(along - closest, side) > radius))

    def path_side_clearance(self, points, heading, distance):
        """Smallest perpendicular side distance along this segment, capped at soft radius."""
        c, s = math.cos(heading), math.sin(heading)
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        # Use obstacles beside the checked segment, not behind it or beyond its end.
        relevant = (along >= 0.0) & (along <= distance)
        cap = max(self.robot_radius, self.p["soft_clearance_m"]) # cap the side clearance at soft clearance !
        if not np.any(relevant):
            return cap  # No measured side restriction within this segment.
        return float(min(cap, np.min(np.abs(side[relevant]))))

    def path_depth(self, points, heading):
        """Estimate centre travel before touching an obstacle, using the hard radius."""
        c, s = math.cos(heading), math.sin(heading)
        along = points[:, 0] * c + points[:, 1] * s
        side = -points[:, 0] * s + points[:, 1] * c
        relevant = (along > 0.0) & (np.abs(side) <= self.robot_radius)
        if not np.any(relevant):
            # No measured obstacle in this corridor: return zero depth conservatively.
            return 0.0
        contact = along[relevant] - np.sqrt(np.maximum(
            0.0, self.robot_radius**2 - side[relevant]**2))
        return float(np.clip(np.min(contact), 0.0, self.p["depth_cap_m"]))

    def motion_centres(self, speed, turn, distance):
        """Robot-centre positions sampled along the constant-command path.

        Coordinates are robot-local: x forward, y left. distance is positive arc length.
        The sign of speed selects forward or reverse; turn keeps its usual yaw sign.
        These are positions ON the curve, not the centre of the turning circle.
        """
        count = max(1, math.ceil(distance / 0.01))
        travel = np.linspace(0.0, distance, count + 1) * np.sign(speed)
        if abs(turn) < 1e-6:
            return np.column_stack((travel, np.zeros_like(travel)))
        # Signed travel / signed speed is positive elapsed time, including in reverse.
        angle = turn * travel / speed
        radius = speed / turn  # Signed turning radius, not the robot footprint radius.
        return np.column_stack((radius * np.sin(angle), radius * (1 - np.cos(angle))))

    def motion_is_clear(self, points, speed, turn, distance, radius=None):
        """Sweep the robot circle along the actual commanded straight line or arc."""
        radius = self.robot_radius if radius is None else max(self.robot_radius, radius)
        if speed == 0.0:
            return self.path_is_clear(points, 0.0, 0.0, radius)
        if abs(turn) < 1e-6:
            return self.path_is_clear(points, 0.0 if speed > 0 else math.pi, distance, radius)
        if len(points) == 0:
            return False
        centres = self.motion_centres(speed, turn, distance)
        separation = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
        # Half a sample spacing covers the spaces between sampled centre positions.
        padding = distance / (2 * (len(centres) - 1))
        return bool(np.all(separation > radius + padding))

    def choose_motion(self, points, error, remaining):
        """Choose an actual command for the held heading; never score new directions."""
        # run_motion holds the target; this helper chooses HOW to approach it each update.
        # Return values: (phase, forward speed in m/s, turn speed in rad/s, explanation).
        if self.path_returns_to_red(self.straight_centres(error, remaining)):
            return "STOP", 0.0, 0.0, "committed path returns toward red failed trail"
        degrees = abs(math.degrees(error))
        turn = float(np.clip(error, -self.p["max_turn_rps"], self.p["max_turn_rps"]))
        if degrees <= self.p["heading_tolerance_deg"]:
            # DRIVE: already aligned (within 3 deg by default), so try straight translation.
            # Hard clearance lets an aligned robot leave an already-tight soft margin.
            if (self.path_is_clear(points, 0.0, remaining)
                    and not self.path_returns_to_red(self.straight_centres(0.0, remaining))):
                return "DRIVE", self.p["forward_speed_mps"], 0.0, "aligned; straight path clear"
            reason = "straight path blocked"
        elif degrees < self.p["moving_turn_limit_deg"]:
            # CURVE: for a moderate error, move and turn together if the predicted arc is clear.
            # Slow translation as the error approaches the turn-in-place boundary.
            speed = self.p["forward_speed_mps"] * (1 - degrees / self.p["moving_turn_limit_deg"])
            distance = min(remaining, speed * self.p["motion_prediction_s"])
            curve_radius = max(self.robot_radius, self.p["curve_clearance_m"])
            returns_to_red = self.path_returns_to_red(self.motion_centres(speed, turn, distance))
            if (self.motion_is_clear(points, speed, turn, distance, radius=curve_radius)
                    and not returns_to_red):
                return "CURVE", speed, turn, f"moving curve clear (curve radius={curve_radius:.3f}m)"
            reason = ("moving curve would return to red trail; align first" if returns_to_red else
                      f"moving curve blocked; align first (curve radius={curve_radius:.3f}m)")
        else:
            reason = "large heading error; align first"

        # A circular footprint does not sweep extra area during an in-place turn.
        # Keep aligning only if the remaining target path still has hard clearance.
        # TURN is the fallback for a large error or a blocked moving arc. Otherwise STOP.
        if abs(error) < 1e-6 or not self.path_is_clear(points, error, remaining):
            return "STOP", 0.0, 0.0, "committed path blocked"
        minimum = (min(self.p["sim_min_turn_rps"], self.p["max_turn_rps"])
                   if self.is_simulation else 0.0)
        turn = float(np.sign(error) * max(abs(turn), minimum))
        return "TURN", 0.0, turn, reason

    def select_direction(self, paths):
        """Use the same score band and depth tie-break for normal and recovery candidates."""
        eligible = paths["eligible"]
        if not eligible:
            return None
        best = max(paths["scores"][i] for i in eligible)
        similar = [i for i in eligible
                   if best - paths["scores"][i] <= self.p["similar_score_tolerance"] + 1e-12]
        return max(similar, key=lambda i: (paths["depths"][i], paths["scores"][i], -abs(paths["angles"][i])))

    def visited_overlap(self, points, angle, visited_body):
        """Measure the fraction of proposed samples near travelled positions.

        Both inputs use current robot axes. Compare the configured start-to-end
        window, shortened at the first hard-circle contact.
        Return (fraction or None, checked end distance, visited mask).
        """
        limit = self.p["visited_check_distance_m"]
        start = self.p["visited_check_start_m"]
        direction = np.array([math.cos(angle), math.sin(angle)])
        along = points @ direction
        side = points @ np.array([-direction[1], direction[0]])
        relevant = (along > 0) & (np.abs(side) <= self.robot_radius)
        if len(points) == 0 or np.any(np.linalg.norm(points, axis=1) <= self.robot_radius):
            limit = 0.0  # No usable geometry, or already inside the hard footprint.
        elif np.any(relevant):
            contact = along[relevant] - np.sqrt(np.maximum(
                0.0, self.robot_radius**2 - side[relevant]**2))
            limit = float(np.clip(np.min(contact), 0.0, limit))
        if limit <= start:
            return None, limit, []  # Obstruction before the comparison window: not "0% visited".
        travel = np.linspace(start, limit, max(1, math.ceil((limit - start) / 0.01)) + 1)
        samples = travel[:, None] * direction
        # Exclude remembered points beyond the obstruction, even when their circles
        # would reach backwards into the checked window. This is a local scan check,
        # not a map of whole rooms or openings hidden around corners.
        history_along = visited_body @ direction
        history_side = visited_body @ np.array([-direction[1], direction[0]])
        radius = self.p["visited_radius_m"]
        # Only dots near this comparison segment can cover a sample. This also
        # keeps the live display inexpensive after recording a long route.
        history = visited_body[(history_along <= limit + 1e-9)
                               & (history_along >= start - radius - 1e-9)
                               & (np.abs(history_side) <= radius + 1e-9)]
        mask = np.zeros(len(samples), dtype=bool)
        if len(history):
            distances = np.linalg.norm(samples[:, None, :] - history[None, :, :], axis=2)
            mask = np.any(distances <= self.p["visited_radius_m"] + 1e-9, axis=1)
        return float(np.mean(mask)), limit, mask.tolist()

    def visited_preview(self, points, scan_angles):
        """Calculate live overlap only, including while the movement heading is held."""
        pose = self.to_course()
        heading = math.radians(pose[2])
        c, s = math.cos(heading), math.sin(heading)
        visited_body = (np.asarray(self.visited_positions).reshape(-1, 2) - pose[:2]) @ np.array([[c, -s], [s, c]])
        visited_body = visited_body[np.linalg.norm(visited_body, axis=1) <=
                                    self.p["visited_check_distance_m"] + self.p["visited_radius_m"]]
        angles, fractions, ends, masks = [], [], [], []
        for degrees in np.arange(-120.0, 120.0 + 1e-6, self.p["candidate_step_deg"]):
            angle = math.radians(degrees)
            ray_errors = (scan_angles - angle + np.pi) % (2 * np.pi) - np.pi
            if not np.any(np.abs(ray_errors) <= abs(self.last_scan.angle_increment) / 2 + 1e-6):
                continue
            fraction, end, mask = self.visited_overlap(points, angle, visited_body)
            angles.append(float(degrees))
            fractions.append(fraction)
            ends.append(end)
            masks.append(mask)
        return {"visited_angles": angles, "visited_fractions": fractions,
                "visited_ends": ends, "visited_masks": masks}

    def score_directions(self, points, scan_angles):
        """Rank hard-clear directions by depth, goal alignment and clearance rewards.

        points and scan_angles come from scan_geometry(), after invalid rays are removed.
        Angles used in geometry/cosine are radians; displayed candidate angles are degrees.
        """
        # Columns of one candidate table: index i refers to the SAME path in each.
        # Keeping the columns separate also lets publish_preview send them to live_map.py.
        candidates = []  # Local heading in degrees: negative = right, positive = left.
        clear = []       # True if the hard robot radius fits over the full lookahead.
        soft = []        # True if the larger, preferred clearance radius also fits.
        red_blocked = [] # True if this direction would return toward a remembered failed end.
        depths = []      # Estimated centre travel in metres; None for a blocked path.
        side_clearances = []  # Perpendicular distance, capped at soft radius; metres.
        side_scores = []      # Gradual reward within the hard-to-soft clearance range.
        scores = []      # Depth + goal + clearance rewards - visited penalty; None if blocked.
        visited = self.visited_preview(points, scan_angles)  # Same supported-angle order as below.
        visited_penalties = []
        # Convert the fixed course goal to a bearing relative to the robot's nose.
        goal = math.radians(self.virtual_bearing_deg())
        distance = self.p["check_distance_m"]
        soft_radius = max(self.robot_radius, self.p["soft_clearance_m"])
        # Score the front 240 degrees; clearance still uses all valid obstacle points.
        for degrees in np.arange(-120.0, 120.0 + 1e-6, self.p["candidate_step_deg"]):
            angle = math.radians(degrees)
            # abs(scan_angles - angle) gives each valid ray's angular distance to this candidate.
            # Accept a ray within half the scan spacing; 1e-6 radians allows rounding error.
            # np.any is True if at least one ray matches. If none match, skip this candidate.
            # Thus an omitted inf/NaN return alone cannot create a candidate direction.
            ray_errors = (scan_angles - angle + np.pi) % (2 * np.pi) - np.pi
            if not np.any(np.abs(ray_errors) <= abs(self.last_scan.angle_increment) / 2 + 1e-6):
                continue
            candidates.append(float(degrees))
            fraction = visited["visited_fractions"][len(candidates) - 1]
            # N/A means no measurable window: there is no evidence to penalise.
            # A penalty is a preference, so equally visited/sole routes remain usable.
            visited_penalties.append(self.p["visited_penalty_weight"] * fraction
                                     if fraction is not None else 0.0)
            clear.append(self.path_is_clear(points, angle, distance))
            red_blocked.append(clear[-1] and self.path_returns_to_red(self.straight_centres(angle, distance)))
            # [-1] means the entry just appended for this candidate. Hard clearance is required.
            soft.append(clear[-1] and self.path_is_clear(points, angle, distance, soft_radius))
            depth = self.path_depth(points, angle) if clear[-1] else None
            depths.append(depth)
            side = self.path_side_clearance(points, angle, distance) if clear[-1] else None
            side_clearances.append(side)
            # 0 at the hard radius, 1 at the soft radius; extra open space adds no more.
            # Equal hard/soft settings leave no interval to reward, so use zero.
            side_fraction = 0.0
            if clear[-1] and soft_radius > self.robot_radius:
                side_fraction = (side - self.robot_radius) / (soft_radius - self.robot_radius)
            side_scores.append(self.p["side_clearance_weight"] * side_fraction
                               if clear[-1] else None)
            # Normalise depth by depth_cap_m. cos(angle - goal) rewards goal alignment:
            # +1 facing the goal, 0 at 90 degrees, -1 facing the opposite direction.
            # Only hard-clear paths receive a score; scoring cannot override a collision.
            scores.append((self.p["depth_weight"] * depth / self.p["depth_cap_m"]
                           + self.p["virtual_goal_weight"] * math.cos(angle - goal)
                           + (self.p["soft_clearance_bonus"] if soft[-1] else 0.0)
                           + side_scores[-1] - visited_penalties[-1])
                          if clear[-1] else None)
        # enumerate(clear) yields (index, boolean), e.g. [False, True] -> (0, False), (1, True).
        # fits is just our variable name for that boolean, not a Python function.
        # Keep indices with True so all other columns can be accessed using the same i.
        # Soft clearance adds a preference; it never excludes a hard-clear doorway.
        eligible = [i for i, fits in enumerate(clear) if fits and not red_blocked[i]]
        paths = {"angles": candidates, "clear": clear, "soft_clear": soft,
                "depths": depths, "scores": scores, "eligible": eligible, "red_blocked": red_blocked,
                "side_clearances": side_clearances, "side_scores": side_scores,
                "visited_penalties": visited_penalties,
                "inside": points[np.linalg.norm(points, axis=1) <= self.robot_radius].tolist()}
        paths["selected_index"] = self.select_direction(paths)
        paths.update(visited)
        return paths



    def record_red_position(self, position, finish=False):
        """Remember actual reverse travel. Each segment points BACK toward its failed end."""
        if self.red_last_position is None:
            self.red_last_position = position.copy()
            return
        distance = float(np.linalg.norm(position - self.red_last_position))
        if distance > 1e-6 and (finish or distance >= self.p["trail_spacing_m"]):
            # We moved from the failed side to the older side, so reverse that order.
            self.red_segments.append((position.copy(), self.red_last_position.copy()))
            self.red_last_position = position.copy()

    def straight_centres(self, angle, distance):
        """Sample a straight proposed path in robot coordinates, like motion_centres()."""
        travel = self.motion_centres(1.0, 0.0, distance)[:, 0]
        return np.column_stack((travel * math.cos(angle), travel * math.sin(angle)))

    def path_returns_to_red(self, centres):
        """Reject following a nearby red segment toward failure; allow outward travel/crossing."""
        if not self.red_segments or len(centres) < 2:
            return False
        pose = self.to_course()
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        world = centres @ np.array([[c, s], [-s, c]]) + pose[:2]
        steps = np.diff(world, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        # Cover the spaces between samples, just as the physical curve checker does.
        radius = self.p["trail_radius_m"] + float(np.max(lengths)) / 2
        alignment = math.cos(math.radians(self.p["red_return_angle_deg"]))
        for older, failed in self.red_segments:
            direction = failed - older
            size = float(np.linalg.norm(direction))
            toward = (lengths > 1e-9) & (steps @ direction >= alignment * lengths * size)
            if not np.any(toward):
                continue
            along = np.clip((world[1:] - older) @ direction / size**2, 0.0, 1.0)
            closest = older + along[:, None] * direction
            nearby = np.linalg.norm(world[1:] - closest, axis=1) <= radius
            if np.any(toward & nearby):
                return True
        return False

    def try_turnaround(self, points):
        """At a reached dot, start turning only if the enclosing turn circle fits."""
        radius = max(self.robot_radius, self.p["turn_clearance_m"])
        if not self.path_is_clear(points, 0.0, 0.0, radius):
            self.get_logger().info(
                f"TURN SPACE: radius={radius:.3f}m blocked; continue to older dot")
            return False
        pose = self.to_course()
        if self.retreat_index > 0:
            outward = self.trail[self.retreat_index - 1] - pose[:2]
        else:
            # At the start there is no older dot; face opposite the last reverse segment.
            older, failed = self.red_segments[-1]
            outward = older - failed
        self.turn_heading = math.degrees(math.atan2(outward[1], outward[0]))
        self.turn_started = time.monotonic()
        self.phase = "TURN_AROUND"
        self.get_logger().info(
            f"TURN_AROUND: space clear; target course={self.turn_heading:+.1f}deg "
            f"radius={radius:.3f}m red_segments={len(self.red_segments)} "
            f"space_recheck={'on' if self.p['recheck_turn_space_during_turn'] else 'off'}")
        self.run_turnaround(points)
        return True

    def run_turnaround(self, points):
        """Hold one outward heading; after alignment resume normal candidate scoring."""
        radius = max(self.robot_radius, self.p["turn_clearance_m"])
        # try_turnaround() always checked this circle before starting. Optionally
        # recheck during rotation; disabling this does not disable stale/timeout stops.
        if (self.p["recheck_turn_space_during_turn"]
                and not self.path_is_clear(points, 0.0, 0.0, radius)):
            self.stop_attempt("turning space became blocked", hold=True)
            return
        error = math.radians(wrap_deg(self.turn_heading - self.to_course()[2]))
        if abs(math.degrees(error)) <= self.p["heading_tolerance_deg"]:
            # Remove the failed tail from the ACTIVE route. Red history retains its coordinates.
            self.trail = self.trail[:self.retreat_index + 1]
            self.trail[-1] = self.to_course()[:2].copy()
            self.trail_reached = []
            self.retreat_index = self.retreat_started = self.red_last_position = None
            self.turn_heading = self.turn_started = None
            self.phase = "SELECT"
            self.move_2D()
            self.get_logger().info("RECOVERY COMPLETE: outward heading reached; resume front 240; new trail green")
            self.publish_preview({}, "RECOVERY COMPLETE: front scoring resumes next update; new trail green")
            return
        if time.monotonic() - self.turn_started >= self.p["turn_timeout_s"]:
            self.stop_attempt("turnaround timed out", hold=True)
            return
        minimum = min(self.p["sim_min_turn_rps"], self.p["max_turn_rps"]) if self.is_simulation else 0.0
        turn = float(np.sign(error) * np.clip(abs(error), minimum, self.p["max_turn_rps"]))
        self.move_2D(turn=turn)
        self.publish_preview({"command_mps": 0.0, "command_rps": turn},
            f"TURN_AROUND: target course={self.turn_heading:+.1f}deg | "
            f"error={math.degrees(error):+.1f}deg w={turn:+.3f}rad/s; "
            f"space_recheck={'on' if self.p['recheck_turn_space_during_turn'] else 'off'}; red history retained")

    def handle_no_candidate(self, points, angles, paths):
        """Reuse the front scorer's result: stop immediately, then confirm persistence."""
        # Called by run_motion() only when score_directions() returns selected_index=None. Handlng backtracking!
        # points: valid obstacle [x, y] positions in robot axes, in metres.
        # angles: valid scan-ray angles in radians. paths: the scorer's table/results.
        # This function returns no value; it changes the phase and publishes stop/status.
        # The old forward attempt is over, so discard its heading and progress origin.
        self.locked_heading = self.drive_start = None # Reset both attributes to None. Locked heading is the direction held during previous foward attempt, drive_start is the position used to measure attempt's progress
        self.advance_progress = 0.0
        # A nonempty inside list means an obstacle is already within the hard circle.
        # Every proposed sweep includes that starting circle, so ordinary retreat is refused.
        if paths["inside"]: # checks for obstacles inside robot footprint
            self.stop_attempt("current hard footprint violated; no automatic retreat", hold=True, paths=paths)
            return
        # No supported candidate angles means missing front scan coverage, rather than
        # measured paths that were checked and found blocked. Do not start recovery.
        if not paths["angles"]: # checking candidate-angle list is empty / no supported front directions were generated
            self.stop_attempt("no supported front scan directions", paths=paths)
            return
        now = time.monotonic()
        # Start the waiting timer once. Keeping this value across callbacks measures
        # one continuous spell of no candidates; finding a candidate resets it below.
        if self.blocked_since is None:
            self.blocked_since = now
            cause = "red-return memory" if any(paths["clear"]) else "physical clearance"
            self.get_logger().info(f"WAIT_BLOCKED: no clear candidate; confirming front blockage; cause={cause}")
        # Confirm persistence using incoming scans, not repeated timer calls on one old scan.
        # Both timestamps use the same monotonic clock. max() keeps the first value at zero
        # if the initial scan arrived slightly before this waiting timer started.
        elapsed = max(0.0, self.last_scan_time - self.blocked_since)
        self.phase = "WAIT_BLOCKED"
        self.move_2D()  # No arguments publishes zero linear and angular velocity NOW.
        if elapsed < self.p["blockage_confirm_s"]:
            # Still confirming: stay stopped; a later callback will score the front again.
            self.publish_preview(paths,
                f"WAIT_BLOCKED: no clear candidate for {elapsed:.2f}/{self.p['blockage_confirm_s']:.2f}s")
        elif self.retreat_limit == 0 and not self.recover:
            # --move / --step: confirmation ends this detection test with a held stop.
            self.stop_attempt("front blockage confirmed; detection-only test; restart to continue", hold=True, paths=paths)
        else:
            # A bounded retreat test is enabled: confirmation may start reversing.
            self.start_retreat(points, angles)

    def start_retreat(self, points, angles):
        """Reverse through older dots; automatic recovery records red segments as it moves."""
        position = self.to_course()[:2]
        if self.recover:
            self.trail_reached = []  # Per-episode indices; red coordinate history is separate.
            self.red_last_position = position.copy()  # Includes the blocked tip, even between dots.
        for index in range(len(self.trail) - 1, -1, -1):
            if np.linalg.norm(position - self.trail[index]) > self.p["retreat_reach_m"]:
                self.retreat_index = index
                self.retreat_started = time.monotonic()
                self.phase = "RETREAT"
                self.blocked_since = None
                limit = "until turning space" if self.recover else f"{self.retreat_limit} dots"
                self.get_logger().info(
                    f"RETREAT: front blockage confirmed; target trail dot #{index}; limit={limit}")
                self.run_retreat(points, angles)
                return
        self.stop_attempt("no earlier breadcrumb available", hold=True)

    def curved_retreat_command(self, local_x, local_y, distance):
        """Return signed speed, yaw rate, checked arc length and an action for one dot."""
        # run_retreat() already handles arrival, so distance exceeds the configured tolerance.
        # Measure the target angle FROM the rear axis: a left-side target has negative error.
        rear_error = math.atan2(-local_y, -local_x)
        if abs(math.degrees(rear_error)) > self.p["retreat_turn_limit_deg"]:
            # Rotate the nose opposite the target's rear-side offset before translating.
            minimum = (min(self.p["sim_min_turn_rps"], self.p["max_turn_rps"])
                       if self.is_simulation else 0.0)
            turn = float(np.sign(rear_error) * np.clip(
                abs(rear_error), minimum, self.p["max_turn_rps"]))
            return 0.0, turn, 0.0, "align rear toward breadcrumb"
        # This circle passes through the current centre and the target (pure pursuit).
        # yaw rate = signed speed * curvature. Reverse-left needs a NEGATIVE yaw rate.
        curvature = 2.0 * local_y / distance**2
        speed = self.p["retreat_speed_mps"]
        arc_length = distance
        if abs(curvature) > 1e-6:
            # Slow down for a tight bend so the required yaw rate stays within its limit.
            speed = min(speed, self.p["max_turn_rps"] / abs(curvature))
            arc_length = abs(2.0 * rear_error / curvature)
        turn = -speed * curvature
        action = "reverse curve toward breadcrumb" if abs(turn) > 1e-6 else "straight reverse toward breadcrumb"
        return -speed, turn, arc_length, action

    def run_retreat(self, points, angles):
        """Follow ordered dots up to the test limit, using the chosen straight/curved mode."""
        # Called on every valid timer update while phase is RETREAT, and once at its start.
        # No return value: publish a reverse/stop command and update the phase/display.
        pose = self.to_course()
        # Target minus current position gives the offset TO the fixed dot in course axes.
        delta = self.trail[self.retreat_index] - pose[:2]
        distance = float(np.linalg.norm(delta))
        # axis=1 calculates one centre distance per [x, y] obstacle row.
        # The boolean mask keeps only points within the current hard footprint.
        inside = points[np.linalg.norm(points, axis=1) <= self.robot_radius].tolist()
        paths = {"inside": inside}  # Display data, including any red obstacle points.
        if inside:
            # Stop on a starting-footprint violation rather than trying to reverse through it.
            self.stop_attempt("current hard footprint violated during retreat", hold=True, paths=paths)
            return
        if distance <= self.p["retreat_reach_m"]:
            self.trail_reached.append(self.retreat_index)  # Viewer colours this dot/circle red.
            self.move_2D()  # Stop at the dot before the next segment is checked.
            reached = len(self.trail_reached)
            progress = f"{reached} dots" if self.recover else f"{reached}/{self.retreat_limit} dots"
            if self.recover:
                self.record_red_position(pose[:2], finish=True)
                if self.try_turnaround(points):
                    return
            if (not self.recover and reached >= self.retreat_limit) or self.retreat_index == 0:
                # End at the requested count, or at the trail start if fewer dots exist.
                self.phase = "RETREAT_DONE"
                reason = ("requested dots reached" if not self.recover and reached >= self.retreat_limit
                          else "trail start reached")
                self.recovery_status = f"RETREAT_DONE: {reason}; reached {progress}; test stopped"
                self.get_logger().info(self.recovery_status)
                self.publish_preview(paths, self.recovery_status)
            else:
                previous = self.retreat_index
                self.retreat_index -= 1  # Strict reverse order; never search by global distance.
                self.retreat_started = time.monotonic()  # Give the next dot its own timeout.
                status = (f"RETREAT: reached trail dot #{previous}; {progress}; "
                          f"next target #{self.retreat_index}; checking next segment")
                self.get_logger().info(status)
                self.publish_preview(paths, status)
                # Keep phase RETREAT. The NEXT fresh timer update checks the new
                # target, scan support and actual reverse sweep before moving again.
            return
        timeout = self.p["retreat_curve_timeout_s"] if self.retreat_curves else self.p["retreat_timeout_s"]
        if time.monotonic() - self.retreat_started >= timeout:
            # Bound this test even if fresh odometry keeps reporting no actual movement.
            self.stop_attempt(f"retreat to trail dot #{self.retreat_index} timed out", hold=True, paths=paths)
            return
        heading = math.radians(pose[2])  # sin/cos expect radians, not degrees.
        # Convert the target offset from course axes to the current robot axes.
        # local_x < 0 means behind the robot. local_y is its sideways offset.
        local_x = delta[0] * math.cos(heading) + delta[1] * math.sin(heading)
        local_y = -delta[0] * math.sin(heading) + delta[1] * math.cos(heading)
        if not self.retreat_curves and (local_x >= 0.0 or abs(local_y) > self.p["retreat_reach_m"]):
            # This first test can only reverse straight. A target ahead or more than
            # 1 cm sideways needs steering/turning, which is left for the next stage.
            self.stop_attempt("previous breadcrumb is not on the straight reverse line; bends are a later stage",
                              hold=True, paths=paths)
            return
        if self.retreat_curves and local_x > self.p["retreat_reach_m"]:
            self.stop_attempt("retreat target is ahead of the robot; reverse test stopped", hold=True, paths=paths)
            return
        # Require a valid rear-facing ray; wrapped scan angles may use either +/-pi.
        # pi - abs(angle) is zero exactly behind and pi exactly ahead.
        # np.any means at least one valid ray matches within half the scan spacing.
        # The tiny 1e-6 addition allows numerical rounding at the boundary.
        rear_supported = np.any(np.pi - np.abs(angles) <= abs(self.last_scan.angle_increment) / 2 + 1e-6)
        if not rear_supported:
            self.stop_attempt("no supported rear scan direction", hold=True, paths=paths)
            return
        if self.retreat_curves:
            # As with a normal candidate, require a valid ray toward the proposed target.
            target_bearing = math.atan2(local_y, local_x)
            ray_errors = (angles - target_bearing + np.pi) % (2 * np.pi) - np.pi
            if not np.any(np.abs(ray_errors) <= abs(self.last_scan.angle_increment) / 2 + 1e-6):
                self.stop_attempt("no supported scan direction toward retreat target", hold=True, paths=paths)
                return
            speed, turn, travel, action = self.curved_retreat_command(local_x, local_y, distance)
        else:
            speed, turn, travel = -self.p["retreat_speed_mps"], 0.0, float(-local_x)
            action = "straight reverse to previous dot"
        # Signed-speed geometry checks the ACTUAL reverse line/arc, not a forward ray.
        # A zero-speed alignment checks the current circular footprint instead.
        if not self.motion_is_clear(points, speed, turn, travel):
            reason = "reverse curve blocked" if abs(turn) > 1e-6 and speed != 0 else "reverse path blocked"
            centres = self.motion_centres(speed, turn, travel) if speed != 0 else np.zeros((1, 2))
            nearest = (float(np.min(np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)))
                       if len(points) else math.inf)
            padding = travel / (2 * (len(centres) - 1)) if speed != 0 and abs(turn) > 1e-6 else 0.0
            reason += (f" (dot=#{self.retreat_index} target={distance:.3f}m "
                       f"nearest_sample={nearest:.3f}m hard_radius={self.robot_radius:.3f}m "
                       f"padding={padding:.3f}m v={speed:.3f}m/s w={turn:+.3f}rad/s)")
            self.stop_attempt(reason, hold=True, paths=paths)
            return
        self.move_2D(x=speed, turn=turn)
        # These fields draw the checked reverse motion and report the command to live_map.py.
        # They do not choose a path or change the stored trail.
        paths.update(command_mps=speed, command_rps=turn,
                     motion_curve=self.motion_centres(speed, turn, travel).tolist() if speed != 0 else [],
                     motion_reason=action)
        self.publish_preview(paths,
            f"RETREAT: target trail dot #{self.retreat_index} remaining={distance:.3f}m | "
            f"v={speed:.3f}m/s w={turn:+.3f}rad/s; {action}; hard-clear")

    def publish_preview(self, paths, status):
        """Display only: publish one snapshot for live_map.py, separate from navigation."""
        paths = {"angles": [], "clear": [], "soft_clear": [], "depths": [], "scores": [],
                 "side_clearances": [], "side_scores": [],
                 "eligible": [], "selected_index": None, "inside": [], "red_blocked": [],
                 **paths, **self.visited_display}
        paths.update(stage_preview=True, status=status, segment=self.segment,
                     visited_check_start_m=self.p["visited_check_start_m"],
                     visited_check_distance_m=self.p["visited_check_distance_m"],
                     visited_radius_m=self.p["visited_radius_m"],
                     visited_penalty_weight=self.p["visited_penalty_weight"],
                     visited_relative=(np.asarray(self.visited_positions) - self.start_pose[:2]).tolist()
                     if self.visited_positions else [],
                     course_pose=None if self.pose is None else self.to_course().tolist(),
                     relative_position=None if self.pose is None or self.start_pose is None
                     else self.relative_position.tolist(),
                     trail_relative=(np.asarray(self.trail) - self.start_pose[:2]).tolist()
                     if self.trail else [], trail_radius_m=self.p["trail_radius_m"],
                     # Bounded tests colour reached indices; automatic recovery retains red coordinates separately.
                     trail_reached_indices=self.trail_reached, backtrack_one=self.backtrack_one,
                     recover=self.recover, turn_heading_deg=self.turn_heading,
                     turn_radius=max(self.robot_radius, self.p["turn_clearance_m"]),
                     turn_space_recheck=self.p["recheck_turn_space_during_turn"],
                     red_segments_relative=(np.asarray(self.red_segments) - self.start_pose[:2]).tolist()
                     if self.red_segments else [],
                     retreat_limit=self.retreat_limit,
                     retreat_curves=self.retreat_curves,
                     retreat_target_index=self.retreat_index,
                     # Send the target relative to the start in COURSE axes. The viewer
                     # converts it to fixed-map and robot-relative display coordinates.
                     retreat_target_relative=None if self.retreat_index is None
                     else (self.trail[self.retreat_index] - self.start_pose[:2]).tolist(),
                     retreat_remaining_m=None if self.retreat_index is None or self.pose is None
                     else float(np.linalg.norm(self.trail[self.retreat_index] - self.to_course()[:2])),
                     right_distance=self.right_distance, switch_x=self.switch_x,
                     virtual_heading_deg=self.virtual_heading_deg(),
                     virtual_bearing_deg=None if self.pose is None else self.virtual_bearing_deg(),
                     distance=self.p["check_distance_m"], radius=self.robot_radius,
                     soft_radius=max(self.robot_radius, self.p["soft_clearance_m"]),
                     motion_enabled=self.motion_enabled, single_step=self.single_step,
                     phase=self.phase if self.motion_enabled else "PREVIEW",
                     locked_heading_deg=self.locked_heading,
                     locked_bearing_deg=None if self.locked_heading is None or self.pose is None
                     else wrap_deg(self.locked_heading - self.to_course()[2]),
                     advance_target_m=self.p["check_distance_m"] / 2,
                     advance_progress_m=self.advance_progress, completed_steps=self.completed_steps)
        self.preview_publisher.publish(String(data="Paths: " + json.dumps(paths, allow_nan=False)))
        self.get_logger().info(status, throttle_duration_sec=1.0)

    def stop_attempt(self, reason, hold=False, paths=None):
        """Discard an interrupted attempt; hold=True latches a stop until restart."""
        self.move_2D()
        if hold or self.phase != "SELECT":
            self.get_logger().info("STOP: " + reason)
        # hold=False: normal selection may retry next update. hold=True: run_motion()
        # keeps publishing zero velocity in BLOCKED until the controller restarts.
        self.phase = "BLOCKED" if hold else "SELECT"
        self.blocked_since = None  # A later attempt must begin a fresh confirmation period.
        self.recovery_status = "STOP: " + reason if hold else ""
        self.locked_heading = self.drive_start = None
        self.advance_progress = 0.0
        self.publish_preview(paths or {}, "STOP: " + reason)  # Keep any supplied obstacle/table display data.

    def run_motion(self, points, angles):
        """Hold one course heading through TURN/CURVE/DRIVE for half the lookahead."""
        # MOTION FLOW: SELECT -> TURN / CURVE / DRIVE -> complete half-lookahead advance.
        # This function runs every 0.05 s. Saved attributes keep the attempt between calls.
        # --step: completion enters DONE and holds a stop until the node restarts.
        # --move: completion returns to SELECT and can issue the next command in THIS call.

        # Recovery dispatch happens before normal scoring/progress. Held stops do not
        # resume when a new gap appears; RETREAT keeps its saved breadcrumb target.
        if self.phase in ("BLOCKED", "RETREAT_DONE"):
            self.move_2D()
            self.publish_preview({"inside": points[np.linalg.norm(points, axis=1) <= self.robot_radius].tolist()},
                                 self.recovery_status)
            return
        if self.phase == "RETREAT":
            self.run_retreat(points, angles)
            return
        if self.phase == "TURN_AROUND":
            self.run_turnaround(points)
            return
        # 1. DONE holds zero velocity. It does not search for more candidates.
        if self.phase == "DONE":
            self.move_2D()
            self.publish_preview({}, "DONE: half-lookahead advance complete; restart for another step")
            return

        pose = self.to_course()  # Current [course x metres, course y metres, heading degrees].
        previous_phase = self.phase  # Used only to log phase changes below.
        # 2. Measure progress BEFORE choosing this update's movement command.
        # A 0.40 m lookahead gives a 0.20 m advance target.
        target = self.p["check_distance_m"] / 2
        if self.drive_start is not None:
            # Progress starts at the FIRST translation, including a moving turn.
            # Project displacement onto the locked heading: sideways travel contributes zero.
            # The saved start stays unchanged through CURVE, DRIVE and any later realignment.
            direction = math.radians(self.locked_heading)
            dx, dy = pose[:2] - self.drive_start
            self.advance_progress = max(0.0, float(dx * math.cos(direction) + dy * math.sin(direction)))
            # 3. Complete once progress reaches the target (within 1e-6 m rounding tolerance).
            # This is the same completion idea as remaining <= 1e-6; overshoot also completes.
            if self.advance_progress >= target - 1e-6:
                self.completed_steps += 1
                self.get_logger().info(
                    f"ADVANCE COMPLETE: {self.advance_progress:.3f}/{target:.3f}m; count={self.completed_steps}")
                if self.single_step:
                    # --step completes exactly one advance, then holds the DONE stop.
                    self.phase = "DONE"
                    self.move_2D()
                    self.publish_preview({}, f"DONE: advanced {self.advance_progress:.3f}/{target:.3f}m")
                    return
                # --move: clear the old attempt and continue to selection below in THIS update.
                # Choose and check the next command without a forced stop between advances.
                self.phase = "SELECT"
                self.locked_heading = self.drive_start = None
                self.advance_progress = 0.0

        # 4. SELECT scores candidates only when starting a fresh attempt.
        # Existing TURN/CURVE/DRIVE attempts keep their heading and skip this selection block.
        paths = {}
        if self.phase in ("SELECT", "WAIT_BLOCKED"):
            # Rank all hard-clear paths by score, including the soft-clearance bonus.
            paths = self.score_directions(points, angles)
            selected = paths["selected_index"]
            if selected is None:
                # Reuse the existing scorer's no-candidate result for Stage 5.
                self.handle_no_candidate(points, angles, paths)
                return
            self.blocked_since = None  # A usable path appeared, so blockage was not continuous.
            # Store a fixed course heading = current heading + chosen robot-relative angle.
            # This prevents the target rotating with the robot or changing on every scan.
            self.locked_heading = wrap_deg(pose[2] + paths["angles"][selected])
            self.drive_start = None
            self.advance_progress = 0.0
            clearance = "soft" if paths["soft_clear"][selected] else "hard-only"
            self.get_logger().info(
                f"SELECT: locked course={self.locked_heading:+.1f}deg "
                f"score={paths['scores'][selected]:.3f} clearance={clearance} "
                f"red_returns_ignored={sum(paths['red_blocked'])}")
            # Log the winner and two alternatives at selection only, not every timer tick.
            ranked = sorted(paths["eligible"], key=lambda i: paths["scores"][i], reverse=True)
            compared = [selected] + [i for i in ranked if i != selected][:2]
            for i in compared:
                goal_score = self.p["virtual_goal_weight"] * math.cos(math.radians(
                    paths["angles"][i] - self.virtual_bearing_deg()))
                depth_score = self.p["depth_weight"] * paths["depths"][i] / self.p["depth_cap_m"]
                soft_score = self.p["soft_clearance_bonus"] if paths["soft_clear"][i] else 0.0
                fraction = paths["visited_fractions"][i]
                visited_text = "N/A" if fraction is None else f"{100 * fraction:.1f}%"
                self.get_logger().info(
                    f"CANDIDATE{' selected' if i == selected else ''}: local={paths['angles'][i]:+.0f}deg "
                    f"total={paths['scores'][i]:.3f} goal={goal_score:.3f} depth_score={depth_score:.3f} "
                    f"soft={soft_score:.3f} side={paths['side_scores'][i]:.3f} "
                    f"side_clearance={paths['side_clearances'][i]:.3f}m (capped) "
                    f"visited={visited_text} visited_penalty={paths['visited_penalties'][i]:.3f} "
                    f"visited_end={paths['visited_ends'][i]:.3f}m")
        # 5. Recalculate distance left and the turn from the current nose to the saved heading.
        remaining = max(0.0, target - self.advance_progress)
        error = math.radians(wrap_deg(self.locked_heading - pose[2]))
        # choose_motion checks fresh obstacle points and picks TURN, CURVE, DRIVE or STOP.
        # It keeps the locked target; it never scores a new set of candidates.
        phase, speed, turn, reason = self.choose_motion(points, error, remaining)
        if phase == "STOP":
            # Abort a blocked attempt, send zero velocity, and reset to SELECT for a later call.
            self.stop_attempt(reason)
            return
        self.phase = phase
        if speed > 0.0:
            if self.drive_start is None:
                # Save the position only when translation first begins (CURVE or DRIVE).
                # Turning in place beforehand does not start the progress measurement.
                self.drive_start = pose[:2].copy()
            # Display only: sample the short predicted motion for the viewer's curve overlay.
            distance = min(remaining, speed * self.p["motion_prediction_s"])
            paths["motion_curve"] = self.motion_centres(speed, turn, distance).tolist()
        # 6. Publish the chosen command and display its phase, heading and measured progress.
        paths.update(command_mps=speed, command_rps=turn, motion_reason=reason)
        self.move_2D(x=speed, turn=turn)
        if self.phase != previous_phase:
            self.get_logger().info(f"{self.phase}: {reason}")
        self.publish_preview(paths,
            f"{self.phase}: {reason} | locked course={self.locked_heading:+.1f}deg "
            f"error={math.degrees(error):+.1f}deg v={speed:.3f}m/s w={turn:+.3f}rad/s "
            f"advance={self.advance_progress:.3f}/{target:.3f}m")

    def timer_callback(self):
        """Measure once, update the segment, then preview or execute one attempt."""
        self.visited_display = {}  # A missing/stale input must not reuse an earlier overlay.
        if self.pose is None or not np.all(np.isfinite(self.pose)):
            self.pose = None
            if self.motion_enabled and self.phase in ("RETREAT", "TURN_AROUND"):
                # Losing pose during reverse holds a stop, even if pose returns later.
                self.stop_attempt("pose unavailable during retreat", hold=True)
                return
            self.blocked_since = None
            self.move_2D()
            self.publish_preview({}, "PREVIEW: waiting for pose; motors stopped")
            return

        if self.calibrate and self.raw_pose is not None:
            self.move_2D()
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            return # Calibration mode: look, don't drive

        if self.last_scan is None:
            if self.motion_enabled and self.phase in ("RETREAT", "TURN_AROUND"):
                # A missing scan cannot justify continuing the reverse test.
                self.stop_attempt("scan unavailable during retreat", hold=True)
                return
            self.blocked_since = None
            self.move_2D()
            self.publish_preview({}, "PREVIEW: waiting for scan; motors stopped")
            return
        if self.motion_enabled and self.phase not in ("DONE", "BLOCKED", "RETREAT_DONE"):
            now = time.monotonic()
            pose_age = now - self.last_pose_time  # Real seconds since the callback received a pose.
            scan_age = now - self.last_scan_time
            if (pose_age > self.p["sensor_timeout_s"]
                    or scan_age > self.p["sensor_timeout_s"]):
                # Stale data cancels pending confirmation. During RETREAT it also
                # latches the stop, so returning data cannot restart reverse movement.
                self.stop_attempt(
                    f"pose or scan stale (pose_age={pose_age:.3f}s scan_age={scan_age:.3f}s "
                    f"limit={self.p['sensor_timeout_s']:.3f}s)",
                    hold=self.phase in ("RETREAT", "TURN_AROUND"))
                return
        points, angles = self.scan_geometry()
        if not self.measure_start(points):
            self.move_2D()
            self.publish_preview({}, "PREVIEW: waiting for initial right-side wall; motors stopped")
            return
        self.update_segment()
        # The current phase describes the previous command whose displacement we now see.
        self.record_trail()
        if self.motion_enabled:
            # Keep percentages live during held movement, recovery and --step's DONE.
            # SELECT/WAIT_BLOCKED already include them in score_directions(). This
            # helper calculates no scores and cannot replace the committed heading.
            if self.phase not in ("SELECT", "WAIT_BLOCKED"):
                self.visited_display = self.visited_preview(points, angles)
            self.run_motion(points, angles)
            return
        self.move_2D()  # Default preview remains stationary.
        paths = self.score_directions(points, angles)
        selected = paths["selected_index"]
        choice = "none" if selected is None else f"{paths['angles'][selected]:+.1f}deg"
        status = (f"PREVIEW: motors stopped | segment={self.segment} "
                  f"relative(right,forward)=({-self.relative_position[1]:+.2f}, {self.relative_position[0]:+.2f})m "
                  f"switch_x={self.switch_x:.2f}m | course goal={self.virtual_heading_deg():.0f}deg "
                  f"selected local={choice}")
        self.publish_preview(paths, status)


def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    mode = arg_parser.add_mutually_exclusive_group()
    mode.add_argument('--calibrate', action='store_true', help="Print Optitrack and maze poses without driving")
    mode.add_argument('--step', action='store_true', help="Advance half the lookahead once, steering as needed, then stop")
    mode.add_argument('--move', action='store_true',
                      help="Repeat half-lookahead advances; hold a stop after confirmed front blockage")
    mode.add_argument('--backtrack-one', action='store_true',
                      help="Drive normally; after confirmed front blockage, reverse to one trail dot and hold a stop")
    mode.add_argument('--backtrack-dots', type=int, metavar='N',
                      help="Drive normally; after confirmed front blockage, reverse through up to N straight trail dots and stop")
    mode.add_argument('--backtrack-curves', type=int, metavar='N',
                      help="Drive normally; after confirmed front blockage, steer backwards through up to N trail dots and stop")
    mode.add_argument('--recover', action='store_true',
                      help="Reverse after blockage, turn where there is space, then resume front scoring with red-route memory")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    for flag, count in (("--backtrack-dots", cli_args.backtrack_dots),
                        ("--backtrack-curves", cli_args.backtrack_curves)):
        if count is not None and count < 1:
            arg_parser.error(f'{flag} must be a positive integer')
    print("Starting obstacle course")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate, cli_args.move, cli_args.step,
                                 backtrack_one=cli_args.backtrack_one,
                                 backtrack_dots=cli_args.backtrack_dots or cli_args.backtrack_curves or 0,
                                 retreat_curves=cli_args.backtrack_curves is not None,
                                 recover=cli_args.recover)
    try:
        rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down")
    finally:
        obstacle.timer.cancel()
        if rclpy.ok():
            obstacle.move_2D()
        obstacle.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
