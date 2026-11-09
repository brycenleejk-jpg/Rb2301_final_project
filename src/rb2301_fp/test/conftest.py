"""Controller fixture without ROS nodes, publishers or driving hardware."""
import time
from types import SimpleNamespace

import pytest

from rb2301_fp.obstacle_course import ObstacleCourseNode, PARAMS


@pytest.fixture
def node():
    # Use real controller methods, replacing only ROS communication and plotting.
    n = ObstacleCourseNode.__new__(ObstacleCourseNode)
    n.p = PARAMS.copy()
    # Isolate the established selector tests with a forward virtual bearing.
    n.p.update(virtual_goal_initial_deg=0.0, virtual_goal_weight=1.0,
               preferred_clearance_m=0.13, check_distance_m=0.3)
    n.robot_radius = n.p["robot_radius_m"]
    n.max_translate_velocity = 0.3
    n.scan_msg = object()
    n.heading_deg = 0.0
    n._last_sensor_yaw_deg = None
    n.last_scan_time = n.last_heading_time = time.monotonic()
    n._last_heading_stamp_ns = 0
    n._heading_fault = None
    n.segment = 1
    n._switch_confirmations = 0
    n.net_right_turn_deg = 0.0
    n.calibrate = False
    n.selected_heading_deg = None
    n._alignment_active = False
    n.commands, n.logs, n.snapshots = [], [], []
    n.publisher_ = SimpleNamespace(publish=lambda msg: n.commands.append(
        (msg.linear.x, msg.linear.y, msg.angular.z)))
    n.get_logger = lambda: SimpleNamespace(info=lambda message, **kw: n.logs.append(message))
    n.plot_goal = lambda: None
    n.plot_direction = lambda *args, **kwargs: None
    n.plot_paths = lambda *args, **kwargs: n.snapshots.append((args, kwargs))
    return n

