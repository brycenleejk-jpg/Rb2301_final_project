"""Live LiDAR and selected-direction viewer (orange carrot).

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
            elif msg.startswith(("STOP:", "ALIGN:", "DRIVE", "DONE:")):
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
        from matplotlib.collections import LineCollection
        self.np, self.plt = np, plt
        self.lidar_x, self.lidar_y = lidar_x, lidar_y
        self.yaw = math.radians(lidar_yaw_deg)
        self.fig, self.ax = plt.subplots(figsize=(12, 9), dpi=90)
        self.fig.subplots_adjust(left=0.02, right=0.72, bottom=0.02, top=0.85)
        # Reverse both display axes to rotate the map by 180 degrees.
        self.ax.set(xlim=(VIEW_RADIUS, -VIEW_RADIUS), ylim=(VIEW_RADIUS, -VIEW_RADIUS), aspect="equal")
        self.ax.set_axis_off()
        self.show_rays = True
        self.rays = LineCollection([], colors="0.75", linewidths=0.4, alpha=0.3, zorder=0)
        self.ax.add_collection(self.rays)
        self.candidate_paths = LineCollection([], linewidths=1.6, alpha=0.95, zorder=1)
        self.ax.add_collection(self.candidate_paths)
        self.path_scores = self.fig.text(0.74, 0.80, "", va="top", fontsize=11)
        self.score_labels = [self.ax.annotate(
            "", xy=(0, 0), xytext=(8, 8 + 18 * i), textcoords="offset points",
            color="green", fontsize=10, visible=False, zorder=8,
            arrowprops={"arrowstyle": "-", "color": "green", "lw": 0.7},
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1})
            for i in range(3)]
        self.blockers, = self.ax.plot([], [], "o", color="red", markersize=5, zorder=6)
        self.ax.text(
            0.02, 0.02,
            "Ray colour: straight-path check only\n"
            "Green: best 3 | grey: other rays\n"
            "Orange circle: carrot | red dots: inside clearance",
            transform=self.ax.transAxes,
            fontsize=12,
            color="0.4",
        )
        self.scan, = self.ax.plot([], [], ".", color="0.6", markersize=2)
        self.ax.plot([0], [0], "b.", markersize=7)
        self.ax.annotate("", xy=(0.4, 0), xytext=(0, 0),
                         arrowprops={"arrowstyle": "->", "color": "navy", "lw": 1.5})
        self.ax.text(0.42, -0.12, "Front 0°", ha="right", color="navy", fontsize=8)
        self.ax.text(0, VIEW_RADIUS * 0.94, "Left +90°", ha="center", va="bottom", color="0.5", fontsize=8)
        self.ax.text(0, -VIEW_RADIUS * 0.94, "Right -90°", ha="center", va="top", color="0.5", fontsize=8)
        self.target_line, = self.ax.plot([], [], "--", color="darkorange", linewidth=2.6, zorder=7)
        self.carrot, = self.ax.plot([], [], "o", color="darkorange", markersize=12,
                                   markeredgecolor="black", markeredgewidth=0.8, zorder=10)
        self.target_angle = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            color="darkorange", fontsize=10, visible=False, zorder=11,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2})
        from matplotlib.patches import Circle
        self.goal, = self.ax.plot([], [], "X", color="purple", markersize=11, zorder=5)
        self.goal_label = self.ax.text(0, 0, "", color="purple", fontsize=9, visible=False)
        self.clearance = Circle((0, 0), 0, fill=False, color="crimson",
                                linestyle=":", linewidth=1.2, visible=False)
        self.ax.add_patch(self.clearance)
        self.waiting = self.ax.text(0.02, 0.98, "Waiting for data", transform=self.ax.transAxes,
                                    va="top", color="0.5", fontsize=12)
        self.status_text = self.fig.text(0.03, 0.98, "", va="top", fontsize=12)
        self.last_stop_text = self.fig.text(0.03, 0.91, "", va="top", fontsize=7, color="firebrick")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

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

    def update(self, packet):
        np = self.np
        status = (packet or {}).get("status")
        last_stop = (packet or {}).get("last_stop")
        self.status_text.set_text((status or "Status unavailable / stale").replace(" | ", "\n", 1))
        self.status_text.set_color("firebrick" if status and status.startswith("STOP:")
                                   else "darkorange" if status and status.startswith("ALIGN:")
                                   else "black")
        self.last_stop_text.set_text("Last forward stop: " + last_stop.split("|")[0].strip()
                                    if last_stop and last_stop != status else "")
        self.scan.set_data([], [])
        self.rays.set_segments([])
        self.candidate_paths.set_segments([])
        self.path_scores.set_text("")
        for label in self.score_labels:
            label.set_text("")
            label.set_visible(False)
        self.blockers.set_data([], [])
        self.carrot.set_visible(False)
        self.target_line.set_visible(False)
        self.target_angle.set_visible(False)
        self.goal.set_visible(False)
        self.goal_label.set_visible(False)
        self.clearance.set_visible(False)
        if packet is None:
            self.waiting.set_text("Waiting for data")
            self.waiting.set_visible(True)
            self.fig.canvas.draw_idle()
            return
        goal_match = GOAL.search(packet.get("goal") or "")
        if goal_match is not None:
            gx, gy, radius = map(float, goal_match.groups())
            if all(map(math.isfinite, (gx, gy, radius))) and radius > 0:
                # Expand the plot to show the actual goal, never a clipped proxy.
                extent = max(VIEW_RADIUS, abs(gx) + 0.5, abs(gy) + 0.5)
                self.ax.set_xlim(extent, -extent)
                self.ax.set_ylim(extent, -extent)
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
            if math.isfinite(x) and math.isfinite(y):
                self.carrot.set_data([x], [y])
                self.carrot.set_visible(True)
                self.target_line.set_data([0, x], [0, y])
                self.target_line.set_visible(True)
                self.target_angle.xy = (x, y)
                self.target_angle.set_text(
                    f"Carrot {math.degrees(math.atan2(y, x)):+.1f}°\n"
                    f"{math.hypot(x, y):.2f} m")
                self.target_angle.set_visible(True)
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
            # Synthetic finite scan to exercise the same rendering path as live data.
            view.update({"scan": {"ranges": [1.5] * 360, "angle_min": -math.pi,
                                  "angle_increment": math.pi / 180,
                                  "range_min": 0.05, "range_max": 10.0},
                         "carrot": "Carrot: x=+0.433 m, y=+0.250 m",
                         "goal": "Goal: x=+2.250 m, y=-1.700 m, radius=0.245 m",
                         "paths": "Paths: " + json.dumps({"angles": list(range(-90,91,5)),
                             "clear": [a >= 20 for a in range(-90,91,5)],
                             "scores": [math.cos(math.radians(a - 30)) + 0.1 if a >= 20 else None
                                        for a in range(-90,91,5)],
                             "distance": 0.5, "radius": 0.245, "inside": []})})
            view.waiting.set_text("DEMO: illustrative values, not a robot run")
            view.waiting.set_visible(True)
            args.output = args.output or Path(__file__).resolve().parent / "runs/minimal-demo"
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
