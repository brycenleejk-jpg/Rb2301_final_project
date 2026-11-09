"""Live LiDAR viewer with virtual-goal and selected-path aiming compass.

Laptop: python3 live_map.py --robot 14 --connect bingda@192.168.1.214
Gazebo: python3 live_map.py --simulation
Only the robot-side --stream process uses ROS when connecting over SSH.
SSH uses this local script for the collector unless --remote-script is supplied.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time


PLOT_HZ = 2.0
LIDAR_X = 0.11423
LIDAR_YAW_DEG = 180.0  # Match the current obstacle_course.py mounting correction.
VIEW_RADIUS = 2.0
MAX_SCAN_POINTS = 360
MAX_SCAN_RAYS = 120
NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
CARROT = re.compile(r"Carrot: x=(" + NUMBER + r") m, y=(" + NUMBER + r") m")


GOAL = re.compile(r"Goal: x=(" + NUMBER + r") m, y=(" + NUMBER + r") m, radius=(" + NUMBER + r") m")
VIRTUAL_GOAL = re.compile(r"Goal: bearing=(" + NUMBER + r") deg, segment=(\d+), net_right=("
                          + NUMBER + r") deg, radius=(" + NUMBER + r") m")


class LatestValues:
    """Bounded storage: keep the latest scan and selected-direction log."""

    def __init__(self):
        self.lock = threading.Lock()
        self.values = {}
        self.counts = {"scan": 0, "carrot": 0, "goal": 0, "paths": 0, "status": 0, "rosout": 0}
        self.loggers = set()

    def receive(self, kind, msg):
        if kind == "log":
            with self.lock:
                self.counts["rosout"] += 1
                if len(self.loggers) < 8:
                    self.loggers.add(msg.name)
            if msg.name != "obstaclecourse":
                return
            msg = msg.msg
            kind = "plot"
        if kind == "plot":
            if msg.startswith("Carrot:"):
                kind, msg = "carrot", msg
            elif msg.startswith("Goal:"):
                kind, msg = "goal", msg
            elif msg.startswith("Paths:"):
                kind, msg = "paths", msg
            elif msg.startswith(("STOP:", "ALIGN:", "DRIVE", "DONE:", "MONITOR:", "PREVIEW:")):
                kind = "status"
            else:
                return
        with self.lock:
            now = time.monotonic()
            self.values[kind] = (now, msg)
            # Keep brief forward stops visible for five seconds after resuming.
            if kind == "status" and msg.startswith(("STOP:", "ALIGN:")):
                self.values["last_stop"] = (now, msg)
            self.counts[kind] += 1

    def packet(self):
        with self.lock:
            values = dict(self.values)
            counts, loggers = dict(self.counts), sorted(self.loggers)
        now = time.monotonic()
        packet = {"type": "frame", "scan": None, "carrot": None, "goal": None, "paths": None, "status": None, "last_stop": None,
                  "received": counts, "loggers": loggers,
                  "domain": os.environ.get("ROS_DOMAIN_ID", "0")}
        for kind, timeout in (("scan", 0.5), ("carrot", 1.5), ("goal", 1.5), ("paths", 1.5), ("status", 2.5), ("last_stop", 5.0)):
            stamp, value = values.get(kind, (0.0, None))
            if value is None or now - stamp > timeout:
                continue
            if kind == "scan":
                step = max(1, math.ceil(len(value.ranges) / MAX_SCAN_POINTS))
                packet[kind] = {
                    "ranges": [float(r) if math.isfinite(r) else None for r in value.ranges[::step]],
                    "angle_min": float(value.angle_min),
                    "angle_increment": float(value.angle_increment) * step,
                    "range_min": float(value.range_min), "range_max": float(value.range_max),
                }
            else:
                packet[kind] = value
        return packet


class RosReceiver:
    """Receive continuously; callbacks never draw or write files."""

    def __init__(self):
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import String
        from rcl_interfaces.msg import Log

        self.ros = rclpy
        self.values = LatestValues()
        self.stop = threading.Event()
        self.error = None
        rclpy.init()
        self.node = Node("course_map_observer", enable_rosout=False, start_parameter_services=False)
        self.node.create_subscription(LaserScan, "/scan", lambda msg: self.values.receive("scan", msg), qos_profile_sensor_data)
        self.node.create_subscription(Log, "/rosout", lambda msg: self.values.receive("log", msg), 100)
        self.node.create_subscription(String, "/navigation/plot",
                                      lambda msg: self.values.receive("plot", msg.data), 10)
        # Candidate snapshots have their own topic so they do not flood logs.
        self.node.create_subscription(String, "/navigation/paths",
                                      lambda msg: self.values.receive("paths", msg.data), 2)

        def receive():
            try:
                while not self.stop.is_set() and rclpy.ok():
                    rclpy.spin_once(self.node, timeout_sec=0.05)
            except ExternalShutdownException:
                pass
            except Exception as error:
                self.error = error

        self.thread = threading.Thread(target=receive, daemon=True)
        self.thread.start()

    def packet(self):
        if self.error is not None:
            raise RuntimeError(f"ROS reception failed: {self.error}")
        if not self.ros.ok():
            raise RuntimeError("ROS reception stopped.")
        return self.values.packet()

    def close(self):
        self.stop.set()
        self.thread.join()
        self.node.destroy_node()
        if self.ros.ok():
            self.ros.shutdown()


class SshReceiver:
    """Read JSON only. This laptop process does not import ROS."""

    def __init__(self, host, robot, remote_script, rate):
        if remote_script is None:
            script = "-"  # Read this laptop's script from SSH standard input.
        elif remote_script.startswith("~/"):
            script = '"$HOME"/' + shlex.quote(remote_script[2:])
        else:
            script = shlex.quote(remote_script)
        command = (
            "source /opt/ros/foxy/setup.bash && "
            f"export ROS_DOMAIN_ID={robot} RMW_IMPLEMENTATION=rmw_fastrtps_cpp && "
            f"exec python3 {script} --robot {robot} --stream --plot-hz {rate:g}"
        )
        self.process = subprocess.Popen(
            ["ssh", "-T", "--", host, "bash", "-lc", shlex.quote(command)],
            stdin=subprocess.PIPE if remote_script is None else None,
            stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        if remote_script is None:
            self.process.stdin.write(Path(__file__).read_text(encoding="utf-8"))
            self.process.stdin.close()
        self.lock = threading.Lock()
        self.latest = None
        self.received_at = 0.0
        self.error = None
        self.stop = threading.Event()

        def receive():
            try:
                for line in self.process.stdout:
                    if self.stop.is_set():
                        return
                    try:
                        packet = json.loads(line)
                    except ValueError:
                        print(line.rstrip(), file=sys.stderr)
                        continue
                    if isinstance(packet, dict) and packet.get("type") == "frame":
                        with self.lock:
                            self.latest, self.received_at = packet, time.monotonic()
                if not self.stop.is_set():
                    self.error = "SSH stream closed; check the robot terminal output above."
            except Exception as error:
                if not self.stop.is_set():
                    self.error = str(error)

        self.thread = threading.Thread(target=receive, daemon=True)
        self.thread.start()

    def packet(self):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            if time.monotonic() - self.received_at > 2.5:
                return None
            return self.latest

    def close(self):
        self.stop.set()
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=1)
        self.process.stdout.close()


def describe_packet(packet, remote=False):
    if packet is None:
        return "No SSH frames yet: check the SSH/password prompt or remote-script error." if remote else "Waiting for ROS data."
    counts = packet.get("received")
    if counts is None:
        return "SSH frames received, but diagnostics are unavailable. Copy the updated live_map.py to the robot."
    raw_paths = packet.get("paths") or ""
    if raw_paths.startswith("Paths: {"):
        try:
            preview = json.loads(raw_paths[len("Paths: "):])
        except ValueError:
            preview = {}
        if preview.get("stage_preview"):
            scan = "fresh" if packet.get("scan") is not None else "missing / stale"
            return (f"ROS_DOMAIN_ID={packet.get('domain', '?')} | /scan={counts.get('scan', 0)} ({scan}) "
                    f"| stage snapshots={counts.get('paths', 0)} | {preview['status']}")
    parts = [f"Robot ROS_DOMAIN_ID={packet.get('domain', '?')}"]
    for key, title in (("scan", "/scan"), ("carrot", "direction updates"), ("goal", "goal updates"), ("paths", "path updates")):
        state = "fresh" if packet.get(key) is not None else "stale" if counts.get(key, 0) else "NOT RECEIVED"
        parts.append(f"{title}={counts.get(key, 0)} ({state})")
    if packet.get("carrot") == "Carrot: none":
        parts.append("controller reports NO TARGET")
    if counts.get("rosout") and not counts.get("carrot"):
        parts.append("ROS loggers seen: " + ", ".join(packet.get("loggers", [])))
    return " | ".join(parts)


class MinimalMap:
    def __init__(self, plt, lidar_x, lidar_y, lidar_yaw_deg):
        import numpy as np
        from matplotlib.collections import LineCollection, PatchCollection
        from matplotlib.transforms import Affine2D
        self.np, self.plt = np, plt
        self.lidar_x, self.lidar_y = lidar_x, lidar_y
        self.yaw = math.radians(lidar_yaw_deg)
        self.fig, self.ax = plt.subplots(figsize=(12, 9), dpi=90)
        self.fig.subplots_adjust(left=0.02, right=0.72, bottom=0.02, top=0.85)
        self.ax.set(xlim=(-VIEW_RADIUS, VIEW_RADIUS), ylim=(-VIEW_RADIUS, VIEW_RADIUS), aspect="equal")
        # Robot +x is forward and +y is left: display (x, y) as (-y, x).
        # Share this rotation across all geometry; sensor mounting stays separate.
        self.robot_transform = Affine2D().rotate_deg(90) + self.ax.transData
        self.ax.set_axis_off()
        self.show_rays = True
        self.rays = LineCollection([], colors="0.75", linewidths=0.4, alpha=0.3, zorder=0,
                                   transform=self.robot_transform)
        self.ax.add_collection(self.rays)
        self.candidate_paths = LineCollection([], linewidths=1.6, alpha=0.95, zorder=1,
                                              transform=self.robot_transform)
        self.ax.add_collection(self.candidate_paths)
        self.aim_text = self.fig.text(0.74, 0.97, "", va="top", fontsize=11)
        self.aim_ax = self.fig.add_axes([0.745, 0.56, 0.23, 0.25], aspect="equal")
        self.aim_ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
        self.aim_ax.set_axis_off()
        self.aim_ax.text(0, 1.2, "FORWARD", ha="center", va="bottom", fontsize=9, color="navy")
        self.aim_ax.text(-1.15, -0.2, "LEFT", ha="center", va="center", fontsize=8, color="0.4")
        self.aim_ax.text(1.15, -0.2, "RIGHT", ha="center", va="center", fontsize=8, color="0.4")
        self.aim_ax.text(0, -1.2, "Robot-relative aim", ha="center", va="top", fontsize=9)
        self.aim_ax.plot([0], [0], "o", color="navy", markersize=6, zorder=5)
        self.aim_ax.annotate("", xy=(0, 0.3), xytext=(0, 0),
                             arrowprops={"arrowstyle": "->", "color": "navy", "lw": 1.2})
        self.virtual_compass_arrow = self.aim_ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), visible=False,
            arrowprops={"arrowstyle": "->", "color": "purple", "lw": 3,
                        "mutation_scale": 18})
        self.selected_compass_arrow = self.aim_ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), visible=False,
            arrowprops={"arrowstyle": "->", "color": "darkorange", "lw": 3,
                        "mutation_scale": 16, "linestyle": "--"})
        self.path_scores = self.fig.text(0.74, 0.51, "", va="top", fontsize=11)
        self.score_labels = [self.ax.annotate(
            "", xy=(0, 0), xycoords=self.robot_transform, annotation_clip=True,
            xytext=(8, 8 + 18 * i), textcoords="offset points",
            color="green", fontsize=10, visible=False, zorder=8,
            arrowprops={"arrowstyle": "-", "color": "green", "lw": 0.7},
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1})
            for i in range(3)]
        self.blockers, = self.ax.plot([], [], "o", color="red", markersize=5, zorder=6,
                                     transform=self.robot_transform)
        self.legend_text = self.ax.text(
            0.02, 0.02,
            "Ray colour: straight-path check only\n"
            "Green: best 3 | grey: other rays\n"
            "Purple: virtual goal | orange: selected path\n"
            "Red dots: inside clearance",
            transform=self.ax.transAxes,
            fontsize=12,
            color="0.4",
        )
        self.scan, = self.ax.plot([], [], ".", color="0.6", markersize=2,
                                 transform=self.robot_transform)
        self.ax.plot([0], [0], "b.", markersize=7, zorder=10, transform=self.robot_transform)
        self.lidar_trail, = self.ax.plot([], [], ".", color="seagreen", markersize=4,
                                        alpha=0.85, zorder=4, transform=self.robot_transform)
        self.lidar_reached, = self.ax.plot([], [], ".", color="crimson", markersize=6,
                                          zorder=5, transform=self.robot_transform)
        self.lidar_visited, = self.ax.plot([], [], ".", color="0.7", markersize=3,
                                         zorder=2, transform=self.robot_transform)
        self.visited_new, = self.ax.plot([], [], ".", color="deepskyblue", markersize=5,
                                       zorder=9, transform=self.robot_transform)
        self.visited_old, = self.ax.plot([], [], "o", color="darkviolet", markersize=4,
                                       markerfacecolor="white", markeredgewidth=1.2,
                                       zorder=9, transform=self.robot_transform)
        self.visited_end, = self.ax.plot([], [], "x", color="black", markersize=7,
                                       zorder=10, transform=self.robot_transform)
        self.visited_text = self.ax.text(
            0.02, 0.96, "", va="top", transform=self.ax.transAxes, fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"}, zorder=12)
        self.inspected_visited_angle = None  # Clicking changes this display only.
        self.last_packet = None
        self.ax.annotate("", xy=(0.4, 0), xytext=(0, 0),
                         xycoords=self.robot_transform,
                         arrowprops={"arrowstyle": "->", "color": "navy", "lw": 1.5})
        self.front_label = self.ax.text(0.48, 0.12, "Front 0°", ha="right", va="center", color="navy", fontsize=8,
                                        transform=self.robot_transform)
        self.left_label = self.ax.text(0, VIEW_RADIUS * 0.94, "Left +90°", ha="left", va="center", color="0.5", fontsize=8,
                                       transform=self.robot_transform, clip_on=True)
        self.right_label = self.ax.text(0, -VIEW_RADIUS * 0.94, "Right -90°", ha="right", va="center", color="0.5", fontsize=8,
                                        transform=self.robot_transform, clip_on=True)
        self.target_line, = self.ax.plot([], [], "--", color="darkorange", linewidth=2.6, zorder=7,
                                        transform=self.robot_transform)
        self.motion_curve, = self.ax.plot([], [], ".-", color="teal", linewidth=3,
                                         markersize=4, zorder=9, transform=self.robot_transform)
        self.carrot, = self.ax.plot([], [], "o", color="darkorange", markersize=12,
                                   markeredgecolor="black", markeredgewidth=0.8, zorder=10,
                                   transform=self.robot_transform)
        self.target_angle = self.ax.annotate(
            "", xy=(0, 0), xycoords=self.robot_transform, annotation_clip=True,
            xytext=(12, 12), textcoords="offset points",
            color="darkorange", fontsize=10, visible=False, zorder=11,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2})
        from matplotlib.patches import Circle
        self.goal, = self.ax.plot([], [], "X", color="purple", markersize=11, zorder=5,
                                 transform=self.robot_transform)
        self.goal_label = self.ax.text(0, 0, "", color="purple", fontsize=9, visible=False,
                                       transform=self.robot_transform)
        self.virtual_goal = self.ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), visible=False, zorder=5,
            xycoords=self.robot_transform, annotation_clip=True,
            arrowprops={"arrowstyle": "->", "color": "purple", "lw": 3, "mutation_scale": 18})
        self.clearance = Circle((0, 0), 0, fill=False, color="crimson",
                                linestyle=":", linewidth=1.2, visible=False,
                                transform=self.robot_transform)
        self.ax.add_patch(self.clearance)
        self.soft_clearance = Circle((0, 0), 0, fill=False, color="goldenrod",
                                     linestyle="--", linewidth=1.3, visible=False,
                                     transform=self.robot_transform)
        self.ax.add_patch(self.soft_clearance)
        self.turn_clearance = Circle((0, 0), 0, fill=False, color="steelblue",
                                     linestyle=":", linewidth=1.5, transform=self.robot_transform)
        self.ax.add_patch(self.turn_clearance)
        self.maze_ax = self.fig.add_axes([0.765, 0.075, 0.20, 0.22], aspect="equal")
        self.maze_ax.set_title("FIXED COURSE: 0° UP", fontsize=10)
        self.maze_ax.set_xlabel("x / rightward (m)", fontsize=8)
        self.maze_ax.set_ylabel("y / forward (m)", fontsize=8)
        self.maze_ax.tick_params(labelsize=7)
        self.maze_ax.grid(alpha=0.2)
        self.maze_ax.plot([0], [0], "ko", markersize=3, label="Start (0, 0)", zorder=4)
        self.maze_trail, = self.maze_ax.plot([], [], ".", color="seagreen",
                                           markersize=3, label="Trail", zorder=3)
        self.maze_reached, = self.maze_ax.plot([], [], ".", color="crimson",
                                             markersize=5, label="Red history", zorder=4)
        self.maze_visited, = self.maze_ax.plot([], [], ".", color="0.7", markersize=2, zorder=2)
        self.maze_retreat_target, = self.maze_ax.plot([], [], "x", color="darkorange",
                                                    markersize=7, zorder=6)
        self.trail_circles = PatchCollection([], facecolor="none", edgecolor="seagreen",
                                            linestyle=":", linewidth=0.7, alpha=0.4, zorder=1)
        self.maze_ax.add_collection(self.trail_circles)
        self.maze_robot, = self.maze_ax.plot([], [], "o", color="navy", markersize=5, zorder=5)
        self.maze_wall = self.maze_ax.axvline(0, color="0.5", linestyle=":", label="Initial right wall")
        self.maze_switch = self.maze_ax.axvline(0, color="seagreen", linestyle="--", label="Lateral switch")
        self.maze_heading = self.maze_ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "navy", "lw": 1.5})
        self.maze_goal = self.maze_ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "purple", "lw": 2})
        self.maze_locked = self.maze_ax.annotate(
            "", xy=(0, 0), xytext=(0, 0), visible=False,
            arrowprops={"arrowstyle": "->", "color": "darkorange", "lw": 2, "linestyle": "--"})
        self.maze_ax.legend(loc="best", fontsize=6, framealpha=0.85)
        self.maze_ax.set_visible(False)
        self.waiting = self.ax.text(0.02, 0.98, "Waiting for data", transform=self.ax.transAxes,
                                    va="top", color="0.5", fontsize=12)
        self.status_text = self.fig.text(0.03, 0.98, "", va="top", fontsize=12)
        self.last_stop_text = self.fig.text(0.03, 0.91, "", va="top", fontsize=7, color="firebrick")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.update_aim_compass()

    def on_click(self, event):
        """Inspect a candidate in the plot. No ROS command or controller setting changes."""
        toolbar = self.fig.canvas.toolbar
        if (event.inaxes is self.ax and event.button == 1 and event.xdata is not None
                and event.ydata is not None and (toolbar is None or not toolbar.mode)):
            # The plot displays robot (x forward, y left) as (-y, x).
            self.inspected_visited_angle = math.degrees(math.atan2(-event.xdata, event.ydata))
            self.update(self.last_packet)

    def draw_visited_preview(self, paths):
        """Render overlap and its score penalty; held movement is never reselected here."""
        np = self.np
        fractions = paths.get("visited_fractions", [])
        if not fractions:
            return  # Older snapshots, or unavailable scan/pose data.
        angles = np.asarray(paths.get("visited_angles", paths["angles"]))
        selected = paths.get("selected_index")
        aim = self.inspected_visited_angle
        if aim is None:
            aim = paths.get("locked_bearing_deg")
        if aim is None and paths.get("phase") == "TURN_AROUND" and paths.get("course_pose") is not None:
            aim = (paths["turn_heading_deg"] - paths["course_pose"][2] + 180) % 360 - 180
        if aim is None:
            inspected = selected if selected is not None else int(np.argmin(np.abs(angles)))
        else:
            inspected = int(np.argmin(np.abs(angles - aim)))
        begin = paths.get("visited_check_start_m", paths["visited_check_distance_m"] / 2)
        end = paths["visited_ends"][inspected]
        mask = np.asarray(paths["visited_masks"][inspected], dtype=bool)
        angle = math.radians(angles[inspected])
        if len(mask):
            travel = np.linspace(begin, end, len(mask))
            samples = travel[:, None] * np.array([math.cos(angle), math.sin(angle)])
            self.visited_new.set_data(samples[~mask, 0], samples[~mask, 1])
            self.visited_old.set_data(samples[mask, 0], samples[mask, 1])
        self.visited_end.set_data([end * math.cos(angle)], [end * math.sin(angle)])
        # Fixed probes make a left/right comparison possible even if the top scoring
        # candidates are all within a few degrees of each other.
        rows = []
        probes = {int(np.argmin(np.abs(angles - a))) for a in (-90, -45, 0, 45, 90)}
        for i in sorted(probes, key=lambda i: angles[i], reverse=True):
            fraction = fractions[i]
            percentage = "N/A" if fraction is None else f"{100 * fraction:.0f}%"
            rows.append(f"{angles[i]:+4.0f}°  {percentage:>4}  end {paths['visited_ends'][i]:.2f} m")
        fraction = fractions[inspected]
        percentage = "N/A: blocked before window" if fraction is None else f"{100 * fraction:.0f}% visited"
        weight = paths.get("visited_penalty_weight", 0.0)
        penalty = weight * fraction if fraction is not None else 0.0
        self.visited_text.set_text(
            f"VISITED OVERLAP: penalty weight {weight:.2f}\n"
            f"Check {begin:.2f}–{paths['visited_check_distance_m']:.2f} m | radius {paths['visited_radius_m']:.2f} m\n"
            + "\n".join(rows) + f"\nInspect {angles[inspected]:+.0f}°: {percentage}\n"
            + f"Visited penalty: -{penalty:.3f} (applied at selection)\n"
            + (f"Shortened at obstruction: {end:.2f} m\n" if end < paths["visited_check_distance_m"] - 1e-6 else "")
            + "Cyan: unvisited | violet: visited | x: check end\nClick a ray to inspect; grey dots: all travelled history")

    def update_aim_compass(self, virtual=None, selected=None, segment=None, net_right=None):
        """Display controller-reported bearings; never infer an aim from raw scans."""
        def label(angle):
            if angle is None:
                return "none / stale"
            return "straight ahead" if abs(angle) < 0.05 else f"{abs(angle):.1f}° {'left' if angle > 0 else 'right'}"

        segment_label = str(segment) if segment is not None else "—"
        turn_label = f"{net_right:+.1f}°" if net_right is not None else "—"
        self.aim_text.set_text(
            f"AIMING\nPurple — virtual: {label(virtual)}\n"
            f"Orange — selected: {label(selected)}\n"
            f"Segment: {segment_label}\nNet right turn: {turn_label}")
        for arrow, bearing, length in ((self.virtual_compass_arrow, virtual, 1.0),
                                       (self.selected_compass_arrow, selected, 0.7)):
            arrow.set_visible(bearing is not None)
            if bearing is not None:
                radians = math.radians(bearing)
                # Robot +x is forward and +y is left. The compass faces upward.
                arrow.xy = (-length * math.sin(radians), length * math.cos(radians))

    def on_key(self, event):
        if event.key == "q":
            self.plt.close(self.fig)
        elif event.key == "r":
            self.show_rays = not self.show_rays
            self.rays.set_visible(self.show_rays)
            self.candidate_paths.set_visible(self.show_rays)
            for label in self.score_labels:
                label.set_visible(self.show_rays and bool(label.get_text()))
            self.fig.canvas.draw_idle()

    def draw_scan(self, scan):
        """Use the same mounting correction in the legacy and stage-preview views."""
        np = self.np
        ranges = np.asarray(scan["ranges"], dtype=float)
        indices = np.flatnonzero(np.isfinite(ranges) & (ranges >= scan["range_min"])
                                 & (ranges <= scan["range_max"]) & (ranges > 0))
        angles = scan["angle_min"] + indices * scan["angle_increment"] + self.yaw
        hits = np.column_stack((self.lidar_x + ranges[indices] * np.cos(angles),
                                self.lidar_y + ranges[indices] * np.sin(angles)))
        self.scan.set_data(hits[:, 0], hits[:, 1])
        ray_step = max(1, math.ceil(len(hits) / MAX_SCAN_RAYS))
        endpoints = hits[::ray_step]
        starts = np.tile([self.lidar_x, self.lidar_y], (len(endpoints), 1))
        self.rays.set_segments(np.stack((starts, endpoints), axis=1))
        self.rays.set_visible(self.show_rays)
        return hits, angles

    def draw_stage_preview(self, packet, paths):
        """Show the controller's stage snapshot; this viewer never selects a path."""
        np = self.np
        extent = max(1.0, 2 * paths["distance"])
        self.ax.set_xlim(-extent, extent)
        self.ax.set_ylim(-extent, extent)
        self.left_label.set_position((0, extent * 0.94))
        self.right_label.set_position((0, -extent * 0.94))
        if packet.get("scan") is not None:
            self.draw_scan(packet["scan"])
        status = paths["status"]
        moving = paths.get("motion_enabled", False)
        phase = paths.get("phase", "PREVIEW")
        retreat_limit = paths.get("retreat_limit", 1 if paths.get("backtrack_one") else 0)
        retreat_curves = paths.get("retreat_curves", False)
        target = paths.get("advance_target_m", paths["distance"] / 2)
        progress = paths.get("advance_progress_m", 0.0)
        pose, relative = paths["course_pose"], paths["relative_position"]
        self.status_text.set_text(status.replace(" | ", "\n", 1))
        self.status_text.set_fontsize(10)
        self.status_text.set_color("firebrick" if status.startswith("STOP:")
                                   else "darkorange" if phase in ("TURN", "TURN_AROUND", "WAIT_BLOCKED")
                                   else "seagreen" if phase in ("CURVE", "DRIVE", "RETREAT") else "navy")
        self.last_stop_text.set_text("")
        self.legend_text.set_text(
            (f"Movement enabled: {phase}" if moving else "Preview only: zero velocity")
            + "\nGreen: soft-clear | amber: hard-only | red: blocked\n"
            + ("Purple: virtual heading | orange: retreat target\n"
               + ("Teal: checked reverse arc\n" if retreat_curves else "Teal: checked straight reverse segment\n")
               + "Circles: hard (red), soft (gold)"
               if paths.get("retreat_target_relative") is not None else
               "Purple: virtual heading | orange: committed heading\n"
               "Orange: remaining advance | teal: predicted centre path\n"
               "Circles: hard (red), soft (gold)"
               if moving else "Purple: virtual heading | orange: selected candidate\n"
               "Circles: hard (red), soft (gold); candidate length = lookahead")
            + "\nGreen: active trail | red: abandoned-route history\n"
            + "Grey rays: return toward red failed end\nBlue dotted circle: turning envelope")
        self.legend_text.set_fontsize(9)
        if phase == "TURN_AROUND":
            self.legend_text.set_text(
                "TURN_AROUND: rotation only\nBlue dotted circle: turning envelope\n"
                "Orange: saved outward heading | purple: virtual goal\n"
                "Red dots: abandoned history | green: active trail")
        angles = np.asarray(paths["angles"], dtype=float)
        radians = np.radians(angles)
        distance = paths["distance"]
        ends = distance * np.column_stack((np.cos(radians), np.sin(radians)))
        segments = np.stack((np.zeros_like(ends), ends), axis=1)
        self.candidate_paths.set_segments(segments)
        self.candidate_paths.set_color([
            "0.65" if i < len(paths.get("red_blocked", [])) and paths["red_blocked"][i] else
            "#24965d" if soft else "#d49b31" if clear else "#cb7777"
            for i, (clear, soft) in enumerate(zip(paths["clear"], paths["soft_clear"]))])
        self.candidate_paths.set_linewidth(0.9)
        self.candidate_paths.set_visible(self.show_rays)
        self.clearance.set_radius(paths["radius"])
        self.clearance.set_visible(True)
        self.soft_clearance.set_radius(paths["soft_radius"])
        self.soft_clearance.set_visible(True)
        self.turn_clearance.set_radius(paths.get("turn_radius", paths["radius"]))
        self.turn_clearance.set_visible(paths.get("recover", False) and phase in ("RETREAT", "TURN_AROUND"))
        inside = np.asarray(paths["inside"], dtype=float).reshape(-1, 2)
        self.blockers.set_data(inside[:, 0], inside[:, 1])
        curve = np.asarray(paths.get("motion_curve", []), dtype=float).reshape(-1, 2)
        self.motion_curve.set_data(curve[:, 0], curve[:, 1])

        selected = paths["selected_index"]
        selected_bearing = None if selected is None else float(angles[selected])
        if moving and paths.get("locked_bearing_deg") is not None:
            selected_bearing = paths["locked_bearing_deg"]
        retreat_target = paths.get("retreat_target_relative")
        retreat_vector = None
        if retreat_target is not None and pose is not None and relative is not None and phase != "TURN_AROUND":
            offset = np.asarray(retreat_target) - relative
            heading = math.radians(pose[2])
            retreat_vector = np.array([math.cos(heading) * offset[0] + math.sin(heading) * offset[1],
                                       -math.sin(heading) * offset[0] + math.cos(heading) * offset[1]])
            selected_bearing = math.degrees(math.atan2(retreat_vector[1], retreat_vector[0]))
        if phase == "TURN_AROUND" and pose is not None:
            selected_bearing = (paths["turn_heading_deg"] - pose[2] + 180) % 360 - 180
        virtual = paths["virtual_bearing_deg"]
        self.update_aim_compass(virtual, selected_bearing, paths["segment"])
        if virtual is not None:
            bearing = math.radians(virtual)
            self.virtual_goal.xy = (0.8 * math.cos(bearing), 0.8 * math.sin(bearing))
            self.virtual_goal.set_visible(True)
        shown_distance = max(0.0, target - progress) if moving else distance
        if phase == "TURN_AROUND":
            shown_distance = distance
        if retreat_vector is not None:
            shown_distance = float(np.linalg.norm(retreat_vector))
        if phase == "RETREAT_DONE":
            shown_distance = 0.0  # Keep the reached red dot visible instead of covering it with the carrot.
        if selected_bearing is not None and shown_distance > 1e-6:
            angle = math.radians(selected_bearing)
            x, y = shown_distance * math.cos(angle), shown_distance * math.sin(angle)
            self.target_line.set_data([0, x], [0, y])
            self.target_line.set_visible(True)
            self.carrot.set_data([x], [y])
            self.carrot.set_visible(phase != "TURN_AROUND")
            self.target_angle.xy = (x, y)
            self.target_angle.set_text(
                f"Turn toward {selected_bearing:+.1f}°"
                if phase == "TURN_AROUND" else
                f"Retreat dot #{paths['retreat_target_index']}\nremaining {shown_distance:.3f} m"
                if retreat_vector is not None else
                f"Locked {selected_bearing:+.1f}°\nremaining {shown_distance:.2f} m" if moving
                else f"Selected {selected_bearing:+.1f}°\nlookahead {distance:.2f} m")
            self.target_angle.set_visible(True)

        ranked = sorted(paths["eligible"], key=lambda i: paths["scores"][i], reverse=True)
        if selected is not None:
            ranked = [selected] + [i for i in ranked if i != selected]
        rows = []
        for i in ranked[:3]:
            side = paths.get("side_clearances", [])
            side_text = f"{100 * side[i]:.1f}" if i < len(side) and side[i] is not None else "—"
            rows.append(f"{'*' if i == selected else ' '} {angles[i]:+4.0f}°  "
                        f"{paths['scores'][i]:.3f}  {paths['depths'][i]:.2f}  {side_text}")
        mode = "hard-clear" if any(paths["clear"]) else "none"
        self.path_scores.set_fontsize(10)
        self.path_scores.set_text(
            f"Eligible paths ({mode})\n Local°  Score  Depth(m) Side(cm)\n"
            + ("\n".join(rows) if rows else "No clear candidate")
            + f"\n* selected; lookahead {distance:.2f} m\nSide capped at soft radius")
        if moving and paths.get("locked_heading_deg") is not None:
            next_action = ("Step complete; motors stopped" if phase == "DONE"
                           else "Advance complete; choosing again" if phase == "SELECT"
                           else "Heading held; overlap preview live")
            self.path_scores.set_text(
                f"{phase}: chosen heading held\nCourse target: {paths['locked_heading_deg']:+.1f}°\n"
                f"Advance: {progress:.3f} / {target:.3f} m\n"
                f"Remaining: {max(0.0, target - progress):.3f} m\n"
                f"Completed advances: {paths.get('completed_steps', 0)}\n"
                f"v={paths.get('command_mps', 0):.3f} m/s  w={paths.get('command_rps', 0):+.3f} rad/s\n"
                + (paths.get("motion_reason", "") + "\n" if paths.get("motion_reason") else "")
                + next_action)

        if phase in ("WAIT_BLOCKED", "BLOCKED", "RETREAT", "RETREAT_DONE"):
            remaining = paths.get("retreat_remaining_m")
            remaining_text = "not selected" if remaining is None else f"{remaining:.3f} m"
            next_action = ("Test stopped; restart controller" if phase in ("BLOCKED", "RETREAT_DONE")
                           else "Confirming no front candidate" if phase == "WAIT_BLOCKED"
                           else "Reverse steering; no new trail dots" if retreat_curves
                           else "Reverse only; no new trail dots")
            retreat_mode = ("reverse until turn space" if paths.get("recover") else
                            f"{retreat_limit}-dot curved retreat" if retreat_curves else
                            "one-dot retreat" if retreat_limit == 1 else
                            f"{retreat_limit}-dot straight retreat" if retreat_limit > 1 else "detection only")
            reached_count = str(len(paths.get("trail_reached_indices", [])))
            if retreat_limit:
                reached_count += f" / {retreat_limit}"
            self.path_scores.set_text(
                f"{phase}\nMode: {retreat_mode}\n"
                f"Target dot: {paths.get('retreat_target_index')}\nRemaining: {remaining_text}\n"
                f"v={paths.get('command_mps', 0):.3f} m/s  w={paths.get('command_rps', 0):+.3f} rad/s\n"
                + (paths["motion_reason"] + "\n" if paths.get("motion_reason") else "")
                + f"Reached retreat dots: {reached_count}\n" + next_action)
        if phase == "TURN_AROUND":
            self.path_scores.set_text(
                f"TURN_AROUND: translating stopped\nTarget course: {paths['turn_heading_deg']:+.1f}°\n"
                f"Turn envelope: {paths.get('turn_radius', 0):.2f} m\n"
                f"Space recheck during turn: {'ON' if paths.get('turn_space_recheck', True) else 'OFF'}\n"
                f"w={paths.get('command_rps', 0):+.3f} rad/s\n"
                "After alignment: front 240° scoring\nRed history remains; new movement green")
        heading = "waiting" if pose is None else f"{pose[2]:+.1f}°"
        position = "waiting for wall" if relative is None else f"({-relative[1]:+.2f}, {relative[0]:+.2f}) m"
        right = "waiting" if paths["right_distance"] is None else f"{paths['right_distance']:.2f} m"
        switch = "waiting" if paths["switch_x"] is None else f"{paths['switch_x']:.2f} m"
        local = "none" if virtual is None else f"{virtual:+.1f}°"
        chosen = "none" if selected_bearing is None or pose is None else f"{(pose[2] + selected_bearing + 180) % 360 - 180:+.1f}°"
        self.aim_text.set_fontsize(10)
        title = (("RECOVERY TEST" if paths.get("recover") else
                  "CURVED RETREAT TEST" if retreat_curves else
                  "MULTI-DOT TEST" if retreat_limit > 1 else "ONE-DOT TEST" if retreat_limit == 1 else
                  "SINGLE STEP" if paths.get("single_step") else "MOVEMENT") + f": {phase}"
                 if moving else "STAGES 1–2: MOTORS STOPPED")
        self.aim_text.set_text(
            f"{title}\nSegment {paths['segment']} | goal course {paths['virtual_heading_deg']:.0f}°\n"
            f"Relative x(right), y(up): {position}\nRobot course heading: {heading}\n"
            f"Initial right wall: {right}\nSwitch at rightward x: {switch}\n"
            f"Virtual local: {local}\n"
            f"{'Turn target course' if phase == 'TURN_AROUND' else 'Retreat course bearing' if retreat_target is not None else 'Selected course'}: {chosen}")
        if relative is not None:
            from matplotlib.patches import Circle
            # Fixed display rotation: course +x is up and course -y is right.
            # This rotation never follows the robot's heading.
            x, y = -relative[1], relative[0]
            self.maze_ax.set_visible(True)
            self.maze_robot.set_data([x], [y])
            trail = np.asarray(paths.get("trail_relative", []), dtype=float).reshape(-1, 2)
            # Subtract the current position, then rotate course offsets into robot axes.
            # The existing robot_transform draws forward up and left to the left.
            offsets = trail - np.asarray(relative)
            heading = math.radians(pose[2])
            c, s = math.cos(heading), math.sin(heading)
            self.lidar_trail.set_data(c * offsets[:, 0] + s * offsets[:, 1],
                                      -s * offsets[:, 0] + c * offsets[:, 1])
            visited = np.asarray(paths.get("visited_relative", []), dtype=float).reshape(-1, 2)
            visited_offsets = visited - np.asarray(relative)
            self.lidar_visited.set_data(c * visited_offsets[:, 0] + s * visited_offsets[:, 1],
                                       -s * visited_offsets[:, 0] + c * visited_offsets[:, 1])
            visited_xy = np.column_stack((-visited[:, 1], visited[:, 0]))
            self.maze_visited.set_data(visited_xy[:, 0], visited_xy[:, 1])
            trail_xy = np.column_stack((-trail[:, 1], trail[:, 0]))
            trail_radius = paths.get("trail_radius_m", 0.1)
            self.maze_trail.set_data(trail_xy[:, 0], trail_xy[:, 1])
            reached = np.asarray(paths.get("trail_reached_indices", []), dtype=int)
            self.lidar_reached.set_data(c * offsets[reached, 0] + s * offsets[reached, 1],
                                        -s * offsets[reached, 0] + c * offsets[reached, 1])
            self.maze_reached.set_data(trail_xy[reached, 0], trail_xy[reached, 1])
            red = np.asarray(paths.get("red_segments_relative", []), dtype=float).reshape(-1, 2)
            red_xy = np.column_stack((-red[:, 1], red[:, 0]))
            if len(red):
                red_offsets = red - np.asarray(relative)
                red_body = np.column_stack((c * red_offsets[:, 0] + s * red_offsets[:, 1],
                                           -s * red_offsets[:, 0] + c * red_offsets[:, 1]))
                old_red = np.column_stack(self.lidar_reached.get_data())
                all_red = np.vstack((old_red, red_body))
                self.lidar_reached.set_data(all_red[:, 0], all_red[:, 1])
                all_red_xy = np.vstack((trail_xy[reached], red_xy))
                self.maze_reached.set_data(all_red_xy[:, 0], all_red_xy[:, 1])
            self.trail_circles.set_paths([Circle((tx, ty), trail_radius) for tx, ty in np.vstack((trail_xy, red_xy))])
            self.trail_circles.set_edgecolor([
                "crimson" if index in reached else "seagreen" for index in range(len(trail))] + ["crimson"] * len(red))
            if retreat_target is not None and phase not in ("RETREAT_DONE", "TURN_AROUND"):
                self.maze_retreat_target.set_data([-retreat_target[1]], [retreat_target[0]])
            self.maze_ax.set_title(
                f"FIXED COURSE: 0° UP\nTrail: {len(trail)} dots | circle radius {trail_radius:.2f} m",
                fontsize=9)
            self.maze_wall.set_xdata([paths["right_distance"]] * 2)
            self.maze_switch.set_xdata([paths["switch_x"]] * 2)
            self.maze_heading.xyann = (x, y)
            self.maze_goal.xyann = (x, y)
            for arrow, degrees in ((self.maze_heading, pose[2]),
                                   (self.maze_goal, paths["virtual_heading_deg"])):
                radians = math.radians(degrees)
                arrow.xy = (x - 0.2 * math.sin(radians), y + 0.2 * math.cos(radians))
            fixed_choice = (pose[2] + selected_bearing if phase == "TURN_AROUND" and selected_bearing is not None
                            else paths.get("locked_heading_deg"))
            if moving and fixed_choice is not None:
                radians = math.radians(fixed_choice)
                self.maze_locked.xyann = (x, y)
                self.maze_locked.xy = (x - 0.25 * math.sin(radians), y + 0.25 * math.cos(radians))
                self.maze_locked.set_visible(True)
            # Keep the entire recorded route visible, including each dotted circle.
            map_points = np.vstack(([[0.0, 0.0], [x, y]], trail_xy, red_xy, visited_xy))
            margin = max(0.25, trail_radius)
            self.maze_ax.set_xlim(min(-0.15, map_points[:, 0].min() - margin),
                                  max(paths["switch_x"], map_points[:, 0].max()) + margin)
            self.maze_ax.set_ylim(min(-0.3, map_points[:, 1].min() - margin),
                                  max(0.4, map_points[:, 1].max() + margin))
        self.waiting.set_text(status if selected_bearing is None else "")
        self.waiting.set_visible(selected_bearing is None and phase not in ("DONE", "RETREAT_DONE"))
        self.draw_visited_preview(paths)
        if paths.get("visited_fractions"):
            self.waiting.set_visible(False)  # The status above already reports no candidate.
        self.fig.canvas.draw_idle()

    def update(self, packet):
        np = self.np
        self.last_packet = packet
        virtual_bearing = selected_bearing = segment_number = net_right_turn = None
        self.update_aim_compass()
        status = (packet or {}).get("status")
        last_stop = (packet or {}).get("last_stop")
        self.status_text.set_text((status or "Status unavailable / stale").replace(" | ", "\n", 1))
        self.status_text.set_color("firebrick" if status and status.startswith("STOP:")
                                   else "darkorange" if status and status.startswith("ALIGN:")
                                   else "black")
        self.last_stop_text.set_text("Last forward stop: " + last_stop.split("|")[0].strip()
                                    if last_stop and last_stop != status else "")
        self.scan.set_data([], [])
        self.lidar_trail.set_data([], [])
        self.lidar_reached.set_data([], [])
        self.lidar_visited.set_data([], [])
        self.maze_visited.set_data([], [])
        self.visited_new.set_data([], [])
        self.visited_old.set_data([], [])
        self.visited_end.set_data([], [])
        self.visited_text.set_text("")
        self.rays.set_segments([])
        self.candidate_paths.set_segments([])
        self.path_scores.set_text("")
        for label in self.score_labels:
            label.set_text("")
            label.set_visible(False)
        self.blockers.set_data([], [])
        self.carrot.set_visible(False)
        self.target_line.set_visible(False)
        self.motion_curve.set_data([], [])
        self.target_angle.set_visible(False)
        self.goal.set_visible(False)
        self.virtual_goal.set_visible(False)
        self.goal_label.set_visible(False)
        self.clearance.set_visible(False)
        self.soft_clearance.set_visible(False)
        self.turn_clearance.set_visible(False)
        self.maze_ax.set_visible(False)
        self.maze_locked.set_visible(False)
        self.maze_trail.set_data([], [])
        self.maze_reached.set_data([], [])
        self.maze_retreat_target.set_data([], [])
        self.trail_circles.set_paths([])
        if packet is None:
            self.waiting.set_text("Waiting for data")
            self.waiting.set_visible(True)
            self.fig.canvas.draw_idle()
            return
        raw_paths = packet.get("paths") or ""
        if raw_paths.startswith("Paths: {"):
            try:
                paths = json.loads(raw_paths[len("Paths: "):])
            except ValueError:
                paths = {}
            if paths.get("stage_preview"):
                self.draw_stage_preview(packet, paths)
                return
        virtual_match = VIRTUAL_GOAL.search(packet.get("goal") or "")
        if virtual_match is not None:
            bearing, segment, net_right, radius = map(float, virtual_match.groups())
            if all(map(math.isfinite, (bearing, segment, net_right, radius))) and radius > 0:
                virtual_bearing, segment_number, net_right_turn = bearing, int(segment), net_right
                # Fixed display length: the virtual goal has a bearing, not a distance.
                px = 0.8 * math.cos(math.radians(bearing))
                py = 0.8 * math.sin(math.radians(bearing))
                self.virtual_goal.xy = (px, py)
                self.virtual_goal.set_visible(True)
                self.goal_label.set_position((px, py))
                self.goal_label.set_text(
                    f"Virtual goal {bearing:+.0f}°\nSegment {int(segment)}, net right {net_right:+.1f}°")
                self.goal_label.set_visible(True)
                self.clearance.set_radius(radius)
                self.clearance.set_visible(True)
        goal_match = GOAL.search(packet.get("goal") or "")
        if goal_match is not None:
            gx, gy, radius = map(float, goal_match.groups())
            if all(map(math.isfinite, (gx, gy, radius))) and radius > 0:
                # Expand the plot to show the actual goal, never a clipped proxy.
                extent = max(VIEW_RADIUS, abs(gx) + 0.5, abs(gy) + 0.5)
                self.ax.set_xlim(-extent, extent)
                self.ax.set_ylim(-extent, extent)
                self.goal.set_data([gx], [gy])
                self.goal.set_visible(True)
                self.goal_label.set_ha("left" if gx > 0 else "right")
                self.goal_label.set_position((gx - 0.08 if gx > 0 else gx + 0.08, gy + 0.08))
                self.goal_label.set_text(f"G ({gx:+.2f}, {gy:+.2f}) m\nDistance {math.hypot(gx, gy):.2f} m")
                self.goal_label.set_visible(True)
                self.clearance.set_radius(radius)
                self.clearance.set_visible(True)
        scan = packet.get("scan")
        if scan is not None:
            hits, angles = self.draw_scan(scan)
        # Draw the controller's decisions, not a separate viewer approximation.
        raw_paths = packet.get("paths") or ""
        if scan is not None and raw_paths.startswith("Paths: {"):
            try:
                paths = json.loads(raw_paths[len("Paths: "):])
                directions = np.asarray(paths["angles"], dtype=float)
                clear = np.asarray(paths["clear"], dtype=bool)
                scores = np.asarray(paths.get("scores", [None] * len(directions)), dtype=float)
                distance, radius = float(paths["distance"]), float(paths["radius"])
                inside = np.asarray(paths["inside"], dtype=float).reshape(-1, 2)
                if (directions.ndim != 1 or clear.shape != directions.shape
                        or scores.shape != directions.shape
                        or not np.all(np.isfinite(directions))
                        or not np.all(np.isfinite(inside))
                        or not math.isfinite(distance) or distance < 0
                        or not math.isfinite(radius) or radius <= 0):
                    raise ValueError("Invalid path snapshot")
                ranked = np.flatnonzero(clear & np.isfinite(scores))
                ranked = ranked[np.argsort(-scores[ranked])][:3]
                if len(hits) and len(directions):
                    # Colour the measured ray nearest each tested direction.
                    difference = (np.radians(directions)[:, None] - angles[None, :]
                                  + np.pi) % (2 * np.pi) - np.pi
                    nearest = np.argmin(np.abs(difference), axis=1)
                    matched = np.abs(difference[np.arange(len(directions)), nearest]) <= (
                        abs(scan["angle_increment"]) / 2 + 1e-5)
                    shown = np.zeros_like(clear)
                    shown[ranked] = True
                    drawn = matched & shown
                    ends = hits[nearest[drawn]]
                    starts = np.tile([self.lidar_x, self.lidar_y], (len(ends), 1))
                    self.candidate_paths.set_segments(np.stack((starts, ends), axis=1))
                    self.candidate_paths.set_color("#00a651")
                    self.candidate_paths.set_visible(self.show_rays)
                    labelled = sorted((i for i in ranked if matched[i]), key=lambda i: directions[i], reverse=True)
                    for label, i in zip(self.score_labels, labelled):
                        label.xy = tuple(hits[nearest[i]])
                        label.set_text(f"{scores[i]:.3f}")
                        label.set_visible(self.show_rays)
                rows = [f"{directions[i]:+6.1f}°   {scores[i]:.3f}" for i in ranked]
                self.path_scores.set_text(
                    "Clear paths (top 3)\nAngle       Score\n"
                    + ("\n".join(rows) if rows else "No scores available"))
                self.blockers.set_data(inside[:, 0], inside[:, 1])
                self.clearance.set_radius(radius)
                self.clearance.set_visible(True)
            except (ValueError, TypeError, KeyError, AttributeError):
                pass  # Ignore malformed/older viewer data; keep rendering scan.
        match = CARROT.search(packet.get("carrot") or "")
        if match is not None:
            x, y = map(float, match.groups())
            if math.isfinite(x) and math.isfinite(y) and math.hypot(x, y) > 1e-9:
                selected_bearing = math.degrees(math.atan2(y, x))
                self.carrot.set_data([x], [y])
                self.carrot.set_visible(True)
                self.target_line.set_data([0, x], [0, y])
                self.target_line.set_visible(True)
                self.target_angle.xy = (x, y)
                self.target_angle.set_text(
                    f"Selected path {math.degrees(math.atan2(y, x)):+.1f}°\n"
                    f"{math.hypot(x, y):.2f} m")
                self.target_angle.set_visible(True)
        self.update_aim_compass(virtual_bearing, selected_bearing, segment_number, net_right_turn)
        missing = [name for key, name in (("scan", "scan"), ("carrot", "direction update"))
                   if packet.get(key) is None]
        no_target = packet.get("carrot") == "Carrot: none"
        self.waiting.set_visible(bool(missing) or no_target)
        self.waiting.set_text("Missing: " + ", ".join(missing) if missing
                              else "No selected travel direction" if no_target else "")
        self.fig.canvas.draw_idle()


def check_hardware(parser, robot):
    if os.environ.get("ROS_DISTRO") != "foxy":
        parser.error("Robot-side ROS reception requires Foxy. For laptop rendering use --robot NUMBER --connect USER@IP.")
    if os.environ.get("ROS_DOMAIN_ID") != str(robot):
        parser.error(f"Set ROS_DOMAIN_ID={robot} on the robot.")
    middleware = os.environ.get("RMW_IMPLEMENTATION") or "rmw_fastrtps_cpp"
    if middleware != "rmw_fastrtps_cpp":
        parser.error("Set RMW_IMPLEMENTATION=rmw_fastrtps_cpp on the robot.")
    os.environ["RMW_IMPLEMENTATION"] = middleware


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--robot", type=int, choices=range(1, 100), metavar="NUMBER")
    mode.add_argument("--simulation", action="store_true")
    mode.add_argument("--demo", action="store_true")
    parser.add_argument("--demo-segment", choices=("A", "B"), default="A",
                        help="Which synthetic stage snapshot to render with --demo")
    parser.add_argument("--demo-visited", choices=("right-travelled", "nearby-dots", "wall", "short-window", "all-visited", "penalty-off", "penalty-on"),
                        help="With --demo: render an overlap test without ROS or movement")
    parser.add_argument("--connect", metavar="USER@IP", help="Render on this laptop; receive JSON over SSH from the robot")
    parser.add_argument("--stream", action="store_true", help="Robot-side headless ROS-to-JSON sender")
    parser.add_argument("--diagnose", action="store_true", help="Print reception counts without opening a plot")
    parser.add_argument("--remote-script", help="Optional robot-side script; by default send this local file over SSH")
    parser.add_argument("--plot-hz", type=float, default=PLOT_HZ)
    parser.add_argument("--lidar-x", type=float, default=LIDAR_X)
    parser.add_argument("--lidar-y", type=float, default=0.0)
    parser.add_argument("--lidar-yaw-deg", type=float, default=LIDAR_YAW_DEG)
    parser.add_argument("--output", type=Path, help="Optionally save one final map.png in this folder; no continuous logging")
    args = parser.parse_args()
    if not math.isfinite(args.plot_hz) or args.plot_hz <= 0:
        parser.error("--plot-hz must be positive and finite.")
    if (args.connect or args.stream) and args.robot is None:
        parser.error("--connect and --stream require --robot NUMBER.")
    if args.connect and (args.stream or args.connect.startswith("-")):
        parser.error("Use --connect USER@IP on the laptop and --stream only on the robot.")
    if args.diagnose and (args.stream or args.demo):
        parser.error("Use --diagnose with --robot/--connect or --simulation.")
    if args.demo_visited and not args.demo:
        parser.error("Use --demo-visited with --demo.")
    if not all(math.isfinite(value) for value in (args.lidar_x, args.lidar_y, args.lidar_yaw_deg)):
        parser.error("LiDAR mounting values must be finite.")
    if args.robot is not None and not args.connect:
        check_hardware(parser, args.robot)
    if args.stream:
        receiver = RosReceiver()
        try:
            while receiver.ros.ok():
                print(json.dumps(receiver.packet(), allow_nan=False), flush=True)
                time.sleep(1 / args.plot_hz)
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        finally:
            receiver.close()
        return
    if args.diagnose:
        receiver = None
        try:
            receiver = (SshReceiver(args.connect, args.robot, args.remote_script, args.plot_hz)
                        if args.connect else RosReceiver())
            while True:
                print(describe_packet(receiver.packet(), bool(args.connect)), flush=True)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        except (RuntimeError, OSError) as error:
            print(f"Viewer stopped: {error}", file=sys.stderr)
            raise SystemExit(1)
        finally:
            if receiver is not None:
                receiver.close()
        return
    if args.demo:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not args.demo and plt.get_backend().lower() == "agg":
        parser.error("No graphical Matplotlib backend on this computer. Use a laptop desktop terminal.")
    view = MinimalMap(plt, args.lidar_x, args.lidar_y, args.lidar_yaw_deg)
    receiver = None
    try:
        if args.demo:
            # Saved outputs from the actual controller with synthetic room scans.
            case = args.demo_visited or args.demo_segment
            filename = "visited_overlap_demo.json" if args.demo_visited else "stage_preview_demo.json"
            examples = json.loads(Path(__file__).with_name(filename).read_text())
            view.inspected_visited_angle = examples[case].get("inspect_angle_deg")
            view.update(examples[case])
            view.status_text.set_text(f"DEMO: {case}; synthetic data, motors stopped\n" + view.status_text.get_text())
            args.output = args.output or Path(__file__).resolve().parent / "runs" / f"stage-preview-{case}"
        else:
            receiver = (SshReceiver(args.connect, args.robot, args.remote_script, args.plot_hz)
                        if args.connect else RosReceiver())
            print(f"Minimal map: laptop drawing at {args.plot_hz:g} Hz; r toggles LiDAR rays, q closes.", flush=True)
            plt.ion()
            plt.show(block=False)
            next_status = time.monotonic() + 5
            while plt.fignum_exists(view.fig.number):
                packet = receiver.packet()
                view.update(packet)
                if time.monotonic() >= next_status:
                    print(describe_packet(packet, bool(args.connect)), flush=True)
                    next_status = time.monotonic() + 5
                plt.pause(max(0.001, 1 / args.plot_hz))
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError) as error:
        print(f"Viewer stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if receiver is not None:
            receiver.close()
        if args.output is not None:
            args.output.mkdir(parents=True, exist_ok=True)
            view.fig.savefig(args.output / "map.png", dpi=130)
            print(f"Saved: {args.output / 'map.png'}")
        plt.close(view.fig)


if __name__ == "__main__":
    main()
