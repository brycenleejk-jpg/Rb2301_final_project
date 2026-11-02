"""Follow the best scan region at the lookahead distance, without opening-edge targeting."""
import argparse
import configparser
import math
import os
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG


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
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
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
        self.get_logger().info("Starting ObstacleCourseNode (region midpoint only)")

        self.is_simulation = is_simulation
        self.calibrate = calibrate # Real robot: only print poses, never drive
        self.raw_pose = None # Pose exactly as Optitrack reported it (real robot only)
        self.frame = self.course = None
        self._ticks = 0

        if is_simulation:
            self.max_translate_velocity = 1.4
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

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
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        # ===== TUNING SETTINGS =====
        self.sim_lidar_yaw_deg = 180.0  # Mounting correction for simulator and current robot
        self.view_half_angle_deg = 100  # Half of the total viewing angle
        self.jump_threshold = 0.3      # Distance change separating regions (m)
        self.min_region_rays = 2        # Minimum readings per region
        self.clearance_percentile = 50  # Representative depth
        self.log_interval = 1.0         # Seconds between log messages
        self.robot_length = 0.05 # Metres
        self.robot_width = 0.05   # Metres
        self.safety_margin = 0.05  # Extra clearance around the robot (m)
        self.lookahead_distance = 0.5  # Preferred carrot distance (m)

        # Circle covering the robot's corners, plus clearance.
        self.robot_radius = np.hypot(
            self.robot_length / 2,
            self.robot_width / 2,
        ) + self.safety_margin

        self.turn_penalty = 0.0  # Larger values discourage bigger turns
        self.depth_weight = 1   # Remaining weight goes to angular width

        self.sensor_timeout = 0.5       # Maximum time without sensor updates (s)
        self.turn_gain = 1.0            # Angular response to carrot angle
        self.min_turn_speed = 0.15  # rad/s
        self.max_turn_speed = 0.30      # rad/s
        self.heading_tolerance_deg = 50 # Align before moving towards the target
        self.forward_speed = 0.3  # m/s

        # ===================================
        self.pose = None
        self.scan_msg = None
        self.last_scan_time = 0.0
        self.last_pose_time = 0.0
        self.carrot = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.scan_msg = msg
        self.last_scan = np.array(msg.ranges)
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

    def path_is_clear(self, angles, ranges, heading, target_distance):
        # Invalid rays are unknown, not obstacle positions. They still break
        # candidate regions below, but must not veto every other direction.
        valid = np.isfinite(angles) & np.isfinite(ranges) & (ranges > 0.0)
        if not np.any(valid):
            self.path_reason = "no usable scan readings"
            return False
        angles = angles[valid]
        ranges = ranges[valid]

        # Obstacle positions relative to the robot.
        # Simulator LiDAR is 0.11423 m ahead of the robot origin.
        x = ranges * np.cos(angles) + 0.11423
        y = ranges * np.sin(angles)

        # Express obstacles relative to the proposed direction.
        along = x * np.cos(heading) + y * np.sin(heading) # X-axis relative to robot
        side = -x * np.sin(heading) + y * np.cos(heading) # Y-axis relative to robot

        # Find each obstacle's distance from the proposed carrot path.
        closest = np.clip(along, 0.0, target_distance)
        distance_to_path = np.hypot(along - closest, side)

        i = int(np.argmin(distance_to_path))
        nearest = float(distance_to_path[i])

        self.path_reason = (
            f"along={along[i]:.3f}, side={side[i]:.3f}, "
            f"closest={closest[i]:.3f}, "
            f"nearest={nearest:.3f} m, "
            f"required>{self.robot_radius:.3f} m"
        )
        return bool(nearest > self.robot_radius)

    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        scan_stale = (
            time.monotonic() - self.last_scan_time
            > self.sensor_timeout
        )

        if self.scan_msg is None or scan_stale:
            self.carrot = None
            self.move_2D()
            return

        if self.calibrate and self.raw_pose is not None:
            self.move_2D()
            self._ticks += 1
            if self._ticks % 10 == 0: # twice a second
                ox, oy, oh = self.raw_pose
                self.get_logger().info(f"optitrack ({ox:+.3f}, {oy:+.3f}, {oh:+7.1f} deg)  ->  maze ({self.pose[0]:+.3f}, {self.pose[1]:+.3f}, {self.pose[2]:+7.1f} deg)")
            # Continue calculating regions for stationary diagnostics. The
            # calibration guard below still prevents all driving commands.

        ###### INSERT CODE HERE ######

        scan = self.scan_msg
        ranges = np.array(scan.ranges, dtype=float)

        # Calculate each ray's direction.
        angles = (
            scan.angle_min
            + np.arange(len(ranges)) * scan.angle_increment
        )

        # Rotate LiDAR readings into the robot frame on both current setups.
        angles += np.radians(self.sim_lidar_yaw_deg)

        # Wrap into [-pi, pi), then arrange from right to left.
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        order = np.argsort(angles)
        angles = angles[order]
        ranges = ranges[order]

        # Prepare distances for gap detection.
        ranges[np.isposinf(ranges)] = scan.range_max #make inf into max range

        valid = ( # Filter for valid readings, marks usable measurements
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges <= scan.range_max)
            & (ranges > 0.0)
        )
        ranges[~valid] = np.nan

        # Examine the configured viewing angle.
        view = np.abs(angles) <= np.radians(self.view_half_angle_deg)
        view_angles = angles[view]
        view_ranges = ranges[view]

        # Split regions when neighbouring distances differ substantially.
        jump_threshold = self.jump_threshold  # Difference in meters that indicates a gap between obstacles. This is a tunable parameter.
        regions = [] # List of (start, end) indices of contiguous regions of similar distance
        start = None
        self.path_reason = "no usable forward rays"

        for i, distance in enumerate(view_ranges):
            direction_clear = (
                np.isfinite(distance)
                and self.path_is_clear(
                    angles,
                    ranges,
                    view_angles[i],
                    self.lookahead_distance,
                )
            )

            if not direction_clear:
                # An unsafe direction ends the current region.
                if start is not None:
                    regions.append((start, i))
                start = None

            elif start is None:
                start = i

            elif abs(distance - view_ranges[i - 1]) > jump_threshold:
                regions.append((start, i))
                start = i

        # Save the last region.
        if start is not None:
            regions.append((start, len(view_ranges)))

        descriptions = []

        best_score = -1.0
        self.carrot = None

        for start, end in regions: # Iterate through each detected region
            if end - start < self.min_region_rays:
                continue  # Ignore isolated rays.

            # Convert the start and end angles of the region from radians to degrees
            lower = np.degrees(view_angles[start]) #for logs and viewer
            upper = np.degrees(view_angles[end - 1]) #same

            # Representative depth across this region
            clearance = np.percentile(view_ranges[start:end], self.clearance_percentile)

            middle = (view_angles[start] + view_angles[end - 1]) * 0.5
            target_distance = self.lookahead_distance

            clear = self.path_is_clear(
                angles, ranges, middle, target_distance
            )

            score = 0.0

            if clear:
                width = view_angles[end - 1] - view_angles[start]

                # Score region depth and angular width, then penalise turning.
                depth_score = clearance / scan.range_max
                width_score = width / np.radians(2 * self.view_half_angle_deg)
                turn_cost = abs(middle) / np.radians(self.view_half_angle_deg)

                score = (
                    self.depth_weight * depth_score
                    + (1.0 - self.depth_weight) * width_score
                ) / (
                    1.0 + self.turn_penalty * turn_cost
                )

                if score > best_score:
                    best_score = score
                    self.carrot = (
                        target_distance * np.cos(middle),
                        target_distance * np.sin(middle),
                    )


            descriptions.append(
                f"[{lower:+.1f} to {upper:+.1f} deg, "
                f"depth={clearance:.2f} m, "
                f"target={target_distance:.2f} m, "
                f"{self.path_reason}, "
                f"score={score:.6f}, "
                f"{'PATH_CLEAR' if clear else 'PATH_BLOCKED'}]"
            )

        if self.carrot is None:
            message = "Carrot: none"
        else:
            x, y = self.carrot
            message = f"Carrot: x={x:+.2f} m, y={y:+.2f} m"

        self.get_logger().info(
            message,
            throttle_duration_sec=self.log_interval,
        )

        self.get_logger().info(
            "Regions: " + ("; ".join(descriptions) or "none"),
            throttle_duration_sec=self.log_interval,
        )

        if self.carrot is None:
            self.get_logger().warning(
                f"Stopped: no usable target; "
                f"scan usable={np.count_nonzero(valid)}/{len(ranges)}, "
                f"forward usable={np.count_nonzero(np.isfinite(view_ranges))}/{len(view_ranges)}; "
                f"last check: {self.path_reason}",
                throttle_duration_sec=self.log_interval,
            )

        if self.calibrate or self.carrot is None:
            self.move_2D()
            return

        x, y = self.carrot
        error = np.arctan2(y, x)

        turn = np.clip(
            self.turn_gain * error,
            -self.max_turn_speed,
            self.max_turn_speed,
        )

        aligned = abs(error) <= np.radians(self.heading_tolerance_deg)

        if aligned and self.path_is_clear(
                angles, ranges, 0.0, self.lookahead_distance):
            self.move_2D(x=self.forward_speed, turn=turn)
        else:
            turn = np.sign(turn) * np.clip(
                abs(turn),
                self.min_turn_speed,
                self.max_turn_speed,
            )
            self.move_2D(turn=turn)
        ###### INSERT CODE HERE ######


def main(args=None):
    arg_parser = argparse.ArgumentParser(description="RB2301 final project")
    arg_parser.add_argument('--calibrate', action='store_true', help="Real robot: only print the Optitrack pose and the maze-frame pose twice a second, never drive")
    cli_args, ros_args = arg_parser.parse_known_args(args=args)
    print("Starting obstacle course without opening-entry adjustment")
    rclpy.init(args=ros_args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation, cli_args.calibrate)
    try:
        rclpy.spin(obstacle)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down")
    finally:
        if rclpy.ok():
            obstacle.timer.cancel()
            obstacle.move_2D()
            time.sleep(0.05)  # Give the final stop message time to leave.
        obstacle.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
