"""Stage preview and half-lookahead movement without ROS nodes or real commands."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import LaserScan

from rb2301_fp.obstacle_course import ObstacleCourseNode, PARAMS


@pytest.fixture
def preview_node():
    n = ObstacleCourseNode.__new__(ObstacleCourseNode)
    n.p = PARAMS.copy()
    # Explicit scenario settings keep these geometry checks independent of routine tuning.
    n.p.update(goal_a_deg=45.0, goal_b_deg=0.0, check_distance_m=0.3,
               depth_weight=1.5, virtual_goal_weight=1.5,
               soft_clearance_bonus=0.15, side_clearance_weight=0.0,
               switch_distance_multiplier=2.0,
               moving_turn_limit_deg=30.0, motion_prediction_s=0.5,
               forward_speed_mps=0.1, robot_radius_m=0.11, soft_clearance_m=0.2,
               curve_clearance_m=0.2,
               trail_spacing_m=0.05, trail_radius_m=0.1,
               visited_check_start_m=0.2, visited_check_distance_m=0.4,
               visited_radius_m=0.1, visited_penalty_weight=0.0,
               recheck_turn_space_during_turn=True)
    n.robot_radius = n.p["robot_radius_m"]
    n.is_simulation = True
    n.max_translate_velocity = 0.3
    n.pose = n.last_scan = n.start_pose = None
    n.course = None
    n.relative_position = np.zeros(2)
    n.trail = []
    n.visited_positions = []
    n.visited_display = {}
    n.right_distance = n.switch_x = None
    n.segment, n.calibrate, n.raw_pose, n._ticks = "A", False, None, 0
    n.motion_enabled = n.single_step = False
    n.backtrack_one = False
    n.retreat_limit = 0
    n.retreat_curves = False
    n.recover = False
    n.red_segments = []
    n.red_last_position = None
    n.turn_heading = n.turn_started = None
    n.blocked_since = n.retreat_index = n.retreat_started = None
    n.recovery_status = ""
    n.trail_reached = []
    n.phase, n.locked_heading, n.drive_start = "SELECT", None, None
    n.advance_progress = 0.0
    n.completed_steps = 0
    n.last_pose_time = n.last_scan_time = -math.inf
    n.commands, n.frames, n.logs = [], [], []
    n.publisher_ = SimpleNamespace(publish=lambda msg: n.commands.append(
        (msg.linear.x, msg.linear.y, msg.angular.z)))
    n.preview_publisher = SimpleNamespace(publish=lambda msg: n.frames.append(
        json.loads(msg.data[len("Paths: "):])))
    n.get_logger = lambda: SimpleNamespace(info=lambda msg, **kwargs: n.logs.append(msg))
    return n


def box_scan(pose, right=0.4):
    """Room with its right wall at course y=-right, including LiDAR offset."""
    heading = math.radians(pose[2])
    sensor = pose[:2] + np.array([
        PARAMS["lidar_x_m"] * math.cos(heading),
        PARAMS["lidar_x_m"] * math.sin(heading)])
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    world_angles = angles + math.radians(PARAMS["lidar_yaw_deg"]) + heading
    dx, dy = np.cos(world_angles), np.sin(world_angles)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (np.where(dx > 0, 2.0, -2.0) - sensor[0]) / dx
        ty = (np.where(dy > 0, 2.0, -right) - sensor[1]) / dy
    msg = LaserScan()
    msg.angle_min, msg.angle_increment = -math.pi, math.pi / 180
    msg.range_min, msg.range_max = 0.05, 10.0
    msg.ranges = np.minimum(tx, ty).tolist()
    return msg


def body_points(world_offsets, heading):
    radians = math.radians(heading)
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray(world_offsets) @ np.array([[c, -s], [s, c]])


def initialise(n, heading=90.0):
    n.pose = np.array([0.0, 0.0, heading])
    n.sub_scan_callback(box_scan(n.pose))
    n.timer_callback()
    return n.frames[-1]


@pytest.mark.parametrize("heading", [-90.0, 0.0, 45.0, 90.0, 179.0])
def test_wall_measurement_uses_fixed_course_right_despite_robot_rotation(preview_node, heading):
    n = preview_node
    n.pose = np.array([2.0, 3.0, heading])
    points = body_points([[-0.02, -0.4], [0, -0.4], [0.02, -0.4], [2, 0], [0.1, 0.5]], heading)
    assert n.measure_start(points)
    assert n.right_distance == pytest.approx(0.4)
    assert n.switch_x == pytest.approx(0.8)
    n.pose[:] = [4, 5, -40]
    assert n.measure_start(np.empty((0, 2)))
    assert n.start_pose == pytest.approx([2, 3, heading])
    assert n.right_distance == pytest.approx(0.4)


def test_missing_wall_does_not_capture_the_start(preview_node):
    n = preview_node
    n.pose = np.array([0.0, 0.0, 90.0])
    assert not n.measure_start(np.empty((0, 2)))
    assert n.start_pose is None and n.switch_x is None


@pytest.mark.parametrize("points", [[[0.4, 0.0]], [[0.0, 0.4]]])
def test_front_or_left_wall_is_not_mistaken_for_the_right_wall(preview_node, points):
    n = preview_node
    n.pose = np.array([0.0, 0.0, 0.0])
    assert not n.measure_start(np.asarray(points))
    assert n.start_pose is None and n.right_distance is None


def test_lateral_position_switches_segment_in_both_directions(preview_node):
    n = preview_node
    n.pose = np.array([2.0, 3.0, 90.0])
    assert n.measure_start(body_points([[0, -0.4]], 90))
    # Travelling forward several metres must not trigger the lateral switch.
    n.pose = np.array([5.0, 3.0, -90.0])
    n.update_segment()
    assert n.relative_position == pytest.approx([3.0, 0.0])
    assert n.segment == "A"
    # The same distance leftward must not count as rightward travel either.
    n.pose[1] = 3.81
    n.update_segment()
    assert n.segment == "A"
    n.pose[1] = 2.21
    n.update_segment()
    assert n.segment == "A"
    n.pose[1] = 2.19
    n.update_segment()
    assert n.segment == "B"
    assert n.virtual_heading_deg() == 0
    n.pose[1] = 3
    n.update_segment()
    assert n.segment == "A" and n.virtual_heading_deg() == 45
    assert any("SEGMENT: B -> A" in line for line in n.logs)


def test_switch_includes_its_boundary(preview_node):
    n = preview_node
    initialise(n)
    n.pose[1] = -n.switch_x
    n.update_segment()
    assert n.segment == "B"


@pytest.mark.parametrize("heading", [-179, 0, 45, 90, 179])
@pytest.mark.parametrize("segment,goal", [("A", 45), ("B", 0)])
def test_virtual_goals_are_fixed_maze_headings(preview_node, heading, segment, goal):
    n = preview_node
    n.pose = np.array([0.0, 0.0, float(heading)])
    n.segment = segment
    assert n.virtual_heading_deg() == goal
    assert (n.virtual_bearing_deg() + heading - goal + 180) % 360 - 180 == pytest.approx(0)


def test_scan_geometry_keeps_metadata_and_filters_invalid_ranges(preview_node):
    n = preview_node
    n.p.update(lidar_yaw_deg=0.0, lidar_x_m=0.1, lidar_y_m=0.0)
    msg = LaserScan()
    msg.angle_min, msg.angle_increment = 0.0, math.pi / 2
    msg.range_min, msg.range_max = 0.05, 5.0
    msg.ranges = [1.0, float("inf"), 0.0, 2.0]
    n.sub_scan_callback(msg)
    assert n.last_scan is msg
    points, angles = n.scan_geometry()
    assert points == pytest.approx(np.array([[1.1, 0], [0.1, -2]]))
    assert angles == pytest.approx([0, -math.pi / 2])


@pytest.mark.parametrize("step,expected", [(3.0, 81), (6.0, 41)])
def test_front_240_sampling_is_configurable(preview_node, step, expected):
    n = preview_node
    n.p["candidate_step_deg"] = step
    frame = initialise(n)
    assert len(frame["angles"]) == expected
    assert frame["angles"][0] == -120 and frame["angles"][-1] == 120
    assert np.diff(frame["angles"]) == pytest.approx(np.full(expected - 1, step))


def test_missing_scan_returns_do_not_become_candidate_directions(preview_node):
    n = preview_node
    initialise(n)
    result = n.score_directions(np.array([[1.0, 0.0]]), np.array([0.0]))
    assert result["angles"] == [0.0]


@pytest.mark.parametrize("side_depth,chosen", [(1.18, 30), (1.10, 0)])
def test_depth_breaks_only_similar_score_ties(preview_node, side_depth, chosen):
    n = preview_node
    n.pose = np.array([0.0, 0.0, 45.0])
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    angles = np.radians([0.0, 30.0])
    ranges = np.array([0.8, side_depth]) + n.robot_radius
    points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
    result = n.score_directions(points, angles)
    assert result["angles"][result["selected_index"]] == chosen


@pytest.mark.parametrize("bonus,chosen", [(0.0, 0), (0.15, 30)])
def test_soft_bonus_can_outweigh_a_small_base_score_difference(preview_node, bonus, chosen):
    n = preview_node
    n.pose = np.array([0.0, 0.0, 45.0])
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p["soft_clearance_bonus"] = bonus
    points = np.array([[0.8, 0], [math.cos(math.pi / 6), 0.5], [0.2, -0.12]])
    result = n.score_directions(points, np.radians([0, 30]))
    assert result["clear"] == [True, True]
    assert result["soft_clear"] == [False, True]
    assert result["eligible"] == [0, 1]
    assert result["angles"][result["selected_index"]] == chosen
    base_soft_score = result["scores"][1] - bonus
    assert result["scores"][0] > base_soft_score


def test_hard_clear_doorway_competes_with_soft_clear_alternative(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p.update(goal_a_deg=0.0, virtual_goal_weight=1.0, check_distance_m=0.4,
               soft_clearance_m=0.2)
    # A 30 cm doorway ahead and an equally deep, wide route to the right.
    points = np.array([[0.25, 0.15], [0.25, -0.15], [2, 0], [0, -2]])
    paths = n.score_directions(points, np.radians([-90, 0]))
    assert paths["clear"] == [True, True]
    assert paths["soft_clear"] == [True, False]
    assert paths["eligible"] == [0, 1]
    assert paths["scores"] == pytest.approx([1.095, 1.945])
    assert paths["angles"][paths["selected_index"]] == 0


def test_soft_bonus_never_allows_a_hard_blocked_path(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p["soft_clearance_bonus"] = 100.0
    paths = n.score_directions(np.array([[0.05, 0]]), np.radians([-90, 0, 90]))
    assert paths["clear"] == [False, False, False]
    assert paths["scores"] == [None, None, None]
    assert paths["eligible"] == [] and paths["selected_index"] is None


@pytest.mark.parametrize("side_distance,chosen", [(0.12, 0), (0.10, None)])
def test_hard_clear_path_remains_eligible_without_soft_margin(preview_node, side_distance, chosen):
    n = preview_node
    n.pose = np.array([0.0, 0.0, 45.0])
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    result = n.score_directions(np.array([[0.8, 0], [0, -side_distance]]), np.array([0.0]))
    assert result["selected_index"] == chosen
    assert result["soft_clear"] == [False]


@pytest.mark.parametrize("enabled,chosen", [(False, 0.0), (True, 3.0)])
@pytest.mark.parametrize("hard,soft,weight", [(0.15, 0.20, 0.5), (0.13, 0.16, 0.3)])
def test_side_reward_overcomes_sideways_goal_bias_and_depth_tie(preview_node, enabled, chosen, hard, soft, weight):
    n = preview_node
    n.p.update(PARAMS, goal_a_deg=0.0, side_clearance_weight=weight if enabled else 0.0,
               robot_radius_m=hard, soft_clearance_m=soft)
    n.robot_radius = n.p["robot_radius_m"]
    n.pose = np.array([0.0, 0.0, 90.0])  # Goal is 90 degrees to the right.
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    # A nearby obstacle on the right; +3 degrees gives about 1 cm more side room.
    # The farther obstacle makes straight ahead slightly deeper, testing the tie rule too.
    angle = math.radians(3)
    points = np.array([[0.2, -(hard + 0.01)], [1.5 * math.cos(angle), 1.5 * math.sin(angle)]])
    paths = n.score_directions(points, np.radians([0, 3]))
    assert paths["clear"] == [True, True] and paths["soft_clear"] == [False, False]
    assert paths["depths"][0] > paths["depths"][1]
    assert paths["side_clearances"][1] - paths["side_clearances"][0] > 0.01
    assert paths["angles"][paths["selected_index"]] == chosen
    if enabled:
        # The wider path must win beyond the existing tie band, not merely by total score.
        assert paths["scores"][1] - paths["scores"][0] > n.p["similar_score_tolerance"]


@pytest.mark.parametrize("points,expected", [
    ([[0.2, 0.16]], 0.16),
    ([[0.2, -0.17]], 0.17),
    ([[0.2, 1.0]], 0.20),
    ([[-0.1, 0.01], [0.6, 0.01]], 0.20),
])
def test_side_preference_uses_segment_projection_and_caps_extra_room(preview_node, points, expected):
    n = preview_node
    n.p.update(robot_radius_m=0.15, soft_clearance_m=0.20)
    n.robot_radius = 0.15
    assert n.path_side_clearance(np.asarray(points), 0.0, 0.4) == pytest.approx(expected)


def test_current_side_reward_keeps_narrow_doorway_eligible_and_selectable(preview_node):
    n = preview_node
    n.p.update(PARAMS, goal_a_deg=0.0, virtual_goal_weight=1.0,
               soft_clearance_bonus=0.3, side_clearance_weight=0.3)
    n.robot_radius = n.p["robot_radius_m"]
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    # Choose a door midway between the currently configured hard and soft diameters.
    half_width = (n.robot_radius + n.p["soft_clearance_m"]) / 2
    points = np.array([[0.25, half_width], [0.25, -half_width], [2, 0], [0, -2]])
    paths = n.score_directions(points, np.radians([-90, 0]))
    assert paths["clear"] == [True, True] and paths["soft_clear"] == [True, False]
    assert paths["angles"][paths["selected_index"]] == 0
    assert paths["side_scores"][1] == pytest.approx(n.p["side_clearance_weight"] / 2)


def test_side_reward_cannot_override_collision_with_segment_end(preview_node):
    n = preview_node
    n.p.update(side_clearance_weight=100.0, check_distance_m=0.4)
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    paths = n.score_directions(np.array([[0.45, 0.0]]), np.radians([0]))
    assert paths["selected_index"] is None
    assert paths["scores"] == [None] and paths["side_scores"] == [None]


def test_equal_hard_and_soft_radii_disable_gradual_side_reward(preview_node):
    n = preview_node
    n.p.update(side_clearance_weight=0.5, soft_clearance_m=n.robot_radius)
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    paths = n.score_directions(np.array([[0.8, 0]]), np.radians([0]))
    assert paths["selected_index"] == 0 and paths["side_scores"] == [0.0]


@pytest.mark.parametrize("state", ["no_pose", "no_scan", "no_wall", "A", "B", "blocked", "calibrate"])
def test_every_preview_state_publishes_zero_velocity(preview_node, state):
    n = preview_node
    if state != "no_pose":
        n.pose = np.array([0.0, 0.0, 90.0])
    if state not in ("no_pose", "no_scan"):
        n.sub_scan_callback(box_scan(n.pose))
    if state in ("B", "blocked"):
        n.timer_callback()
        if state == "B":
            n.pose[1] = -n.switch_x - 0.1
            n.sub_scan_callback(box_scan(n.pose, right=2.0))
        else:
            n.last_scan.ranges = [0.06] * 360
    if state == "no_wall":
        n.last_scan.ranges = [float("inf")] * 360
    if state == "calibrate":
        n.calibrate, n.raw_pose = True, np.array([0.0, 0.0, 90.0])
    n.timer_callback()
    assert n.commands and all(command == (0.0, 0.0, 0.0) for command in n.commands)
    if state != "calibrate":
        assert n.frames[-1]["stage_preview"]
    if state == "blocked":
        assert n.frames[-1]["selected_index"] is None


def test_template_optitrack_conversion_still_supplies_the_pose(preview_node):
    n = preview_node
    n.frame = dict(origin_x=2.0, origin_y=3.0, rotation_deg=90.0, heading_offset_deg=0.0)
    message = SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=2.0, y=4.0),
        orientation=Quaternion(z=math.sin(math.pi / 4), w=math.cos(math.pi / 4))))
    n.optitrack_callback(message)
    assert n.pose == pytest.approx([1.0, 0.0, 0.0], abs=1e-10)


def test_navigation_converts_to_course_before_saving_or_subtracting_positions(preview_node):
    n = preview_node
    n.course = dict(origin_x=2.0, origin_y=3.0, rotation_deg=-90.0)
    n.pose = np.array([2.0, 3.0, -90.0])
    assert n.measure_start(np.array([[0.0, -0.4]]))
    assert n.start_pose == pytest.approx([0.0, 0.0, 0.0])
    assert n.right_distance == pytest.approx(0.4)
    # CA2 down becomes course +x; CA2 right becomes course +y.
    n.pose = np.array([2.2, 2.9, -80.0])
    n.update_segment()
    assert n.relative_position == pytest.approx([0.1, 0.2])
    assert n.virtual_bearing_deg() == pytest.approx(35.0)
    assert n.pose == pytest.approx([2.2, 2.9, -80.0])  # Preserve the template's CA2 pose.
    n.publish_preview({}, "PREVIEW: motors stopped")
    assert n.frames[-1]["course_pose"] == pytest.approx([0.1, 0.2, 10.0])
    n.pose = np.array([2.0, 1.0, -90.0])  # Forward in course coordinates.
    n.update_segment()
    assert n.segment == "A"
    n.pose[0] = 1.1  # Rightward in course coordinates.
    n.update_segment()
    assert n.segment == "B"


@pytest.fixture
def viewer_module():
    import matplotlib
    matplotlib.use("Agg")
    path = Path(__file__).resolve().parents[3] / "visualization/live_map.py"
    spec = importlib.util.spec_from_file_location("stage_viewer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def viewer_packet(n):
    scan = n.last_scan
    return {"paths": "Paths: " + json.dumps(n.frames[-1]),
            "scan": None if scan is None else {
                "ranges": [float(r) if math.isfinite(r) else None for r in scan.ranges],
                "angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                "range_min": scan.range_min, "range_max": scan.range_max}}


def test_viewer_diagnostics_use_stage_snapshots_instead_of_legacy_arrow_topics(preview_node, viewer_module):
    n = preview_node
    initialise(n)
    values = viewer_module.LatestValues()
    values.receive("scan", n.last_scan)
    values.receive("paths", viewer_packet(n)["paths"])
    values.receive("plot", "PREVIEW: motors stopped")
    description = viewer_module.describe_packet(values.packet())
    assert "stage snapshots=1" in description
    assert "direction updates" not in description
    assert "NOT RECEIVED" not in description
    assert values.counts["status"] == 1


def test_viewer_renders_real_controller_snapshots_and_clears_stale_data(preview_node, viewer_module):
    import matplotlib.pyplot as plt
    n = preview_node
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        n.timer_callback()
        view.update(viewer_packet(n))
        assert "waiting for pose" in view.status_text.get_text()
        frame = initialise(n)
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        assert "Segment A | goal course 45°" in view.aim_text.get_text()
        assert "Side(cm)" in view.path_scores.get_text()
        assert view.maze_ax.get_visible()
        assert view.maze_switch.get_xdata() == pytest.approx([frame["switch_x"]] * 2)
        assert len(view.candidate_paths.get_segments()) == len(frame["angles"])
        n.p["goal_a_deg"] = -45.0
        n.pose[2] = 0.0
        n.sub_scan_callback(box_scan(n.pose))
        n.timer_callback()
        view.update(viewer_packet(n))
        assert view.maze_goal.xy[0] > 0 and view.maze_goal.xy[1] > 0
        assert view.maze_heading.xy[0] == pytest.approx(0)
        assert view.maze_heading.xy[1] > 0
        n.pose[1] = -n.switch_x - 0.1
        n.sub_scan_callback(box_scan(n.pose, right=2.0))
        n.timer_callback()
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        assert "Segment B | goal course 0°" in view.aim_text.get_text()
        assert "Relative x(right), y(up): (+0.90, +0.00)" in view.aim_text.get_text()
        assert view.maze_robot.get_xdata()[0] > n.switch_x
        assert view.maze_robot.get_ydata()[0] == pytest.approx(0)
        assert view.maze_goal.xy[0] == pytest.approx(-n.relative_position[1])
        assert view.maze_goal.xy[1] > n.relative_position[0]
        view.update(None)
        assert not view.maze_ax.get_visible()
        assert not view.virtual_goal.get_visible()
        assert not view.target_line.get_visible()
    finally:
        plt.close(view.fig)


@pytest.fixture
def motion_node(preview_node, monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr("rb2301_fp.obstacle_course.time.monotonic", lambda: clock.now)
    n = preview_node
    n.motion_enabled = n.single_step = True
    n.p["goal_a_deg"] = 0.0
    n.pose = np.array([0.0, 0.0, 0.0])
    return n, clock


def motion_update(n, clock, points=((2.0, 0.0),), angles=(0.0,)):
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    points = np.asarray(points, dtype=float)
    if n.start_pose is None:
        # The motion scenarios also need the fixed right wall for startup.
        wall = body_points([[0.0, -2.0]], n.to_course()[2])
        points = np.vstack((points, wall))
    n.scan_geometry = lambda: (points, np.radians(angles))
    n.last_pose_time = n.last_scan_time = clock.now
    n.timer_callback()


def test_side_trial_logs_selected_and_alternatives_only_when_selecting(motion_node):
    n, clock = motion_node
    n.p["side_clearance_weight"] = PARAMS["side_clearance_weight"]
    motion_update(n, clock, [[2, 0], [2, 1], [2, -1]], [-3, 0, 3])
    comparisons = [line for line in n.logs if line.startswith("CANDIDATE")]
    assert len(comparisons) == 3 and "CANDIDATE selected:" in comparisons[0]
    for line in comparisons:
        values = dict(field.split("=", 1) for field in line.split() if "=" in field)
        assert float(values["total"]) == pytest.approx(
            sum(float(values[key]) for key in ["goal", "depth_score", "soft", "side"])
            - float(values["visited_penalty"]), abs=0.003)
        assert "side_clearance=" in line
    motion_update(n, clock)
    assert [line for line in n.logs if line.startswith("CANDIDATE")] == comparisons


def test_turn_then_curve_then_drive_keeps_heading_without_rescoring(motion_node):
    n, clock = motion_node
    n.p["goal_a_deg"] = -45.0
    points = [[2.0, 0.0], [math.sqrt(2), -math.sqrt(2)]]
    motion_update(n, clock, points, [0, -45])
    assert n.phase == "TURN" and n.commands[-1][2] < 0
    assert n.locked_heading == pytest.approx(-45)

    def unexpected_selection(*args):
        pytest.fail("Rescored before completing the committed advance")

    n.score_directions = unexpected_selection
    # The higher-level goal changes, but the current attempt remains committed.
    n.switch_x = 0.0  # Put the current position on B's boundary without translating.
    n.pose[2] = -20
    motion_update(n, clock)
    assert n.phase == "CURVE" and n.commands[-1][0] > 0 and n.commands[-1][2] < 0
    assert n.segment == "B"
    assert n.drive_start == pytest.approx([0, 0]) and n.advance_progress == 0
    assert n.locked_heading == pytest.approx(-45)
    n.pose[2] = -45
    motion_update(n, clock)
    assert n.phase == "DRIVE"
    assert n.commands[-1][0] > 0 and n.commands[-1][2] == 0
    assert n.locked_heading == pytest.approx(-45)


@pytest.mark.parametrize("lookahead", [0.4, 0.6])
def test_advances_half_lookahead_then_stays_stopped(motion_node, lookahead):
    n, clock = motion_node
    n.p["check_distance_m"] = lookahead
    motion_update(n, clock)
    assert n.phase == "DRIVE"
    n.pose[0] = lookahead / 2 - 0.001
    motion_update(n, clock)
    assert n.commands[-1][0] > 0
    n.pose[0] = lookahead / 2
    motion_update(n, clock)
    assert n.phase == "DONE"
    assert n.advance_progress == pytest.approx(lookahead / 2)
    assert n.commands[-1] == (0, 0, 0)
    # A completed single step must not restart, even after a sensor pause.
    clock.now += 1.0
    n.timer_callback()
    assert n.phase == "DONE" and n.commands[-1] == (0, 0, 0)
    for _ in range(3):
        motion_update(n, clock)
        assert n.phase == "DONE" and n.commands[-1] == (0, 0, 0)
    assert n.completed_steps == 1


def test_sideways_displacement_does_not_complete_forward_progress(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    n.pose[1] = 0.4
    motion_update(n, clock)
    assert n.advance_progress == pytest.approx(0)
    assert n.phase == "DRIVE" and n.commands[-1][0] > 0


def test_progress_origin_is_saved_at_first_curved_translation(motion_node):
    n, clock = motion_node
    n.p["goal_a_deg"] = -45.0
    points = [[2.0, 0.0], [math.sqrt(2), -math.sqrt(2)]]
    motion_update(n, clock, points, [0, -45])
    n.pose[:] = [0.1, -0.1, -20]
    motion_update(n, clock)
    assert n.phase == "CURVE" and n.advance_progress == 0
    assert n.drive_start == pytest.approx([0.1, -0.1])
    n.pose[:2] += [0.02, -0.02]
    n.pose[2] = -45
    motion_update(n, clock)
    assert n.drive_start == pytest.approx([0.1, -0.1])
    assert n.advance_progress == pytest.approx(math.sqrt(2) * 0.02)


def test_remaining_step_is_checked_instead_of_another_full_lookahead(motion_node):
    n, clock = motion_node
    n.p["check_distance_m"] = 0.6
    motion_update(n, clock, [[0.73, 0]])  # Hard-radius depth is 0.62 m.
    assert n.phase == "DRIVE"
    n.pose[0] = 0.1
    motion_update(n, clock, [[0.63, 0]])
    assert not n.path_is_clear(np.array([[0.63, 0]]), 0, 0.6)
    assert n.advance_progress == pytest.approx(0.1)
    assert n.phase == "DRIVE" and n.commands[-1][0] > 0
    n.pose[0] = 0.3
    motion_update(n, clock, [[0.43, 0]])
    assert n.phase == "DONE" and n.commands[-1] == (0, 0, 0)


def test_new_obstacle_in_remaining_step_interrupts_the_attempt(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    motion_update(n, clock, [[0.2, 0]])
    assert n.commands[-1] == (0, 0, 0)
    assert n.phase == "SELECT" and n.locked_heading is None
    assert "committed path blocked" in n.frames[-1]["status"]


def test_loss_of_soft_margin_does_not_cancel_a_hard_clear_attempt(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    motion_update(n, clock, [[2.0, 0], [0, -0.12]])
    assert n.phase == "DRIVE" and n.commands[-1][0] > 0


@pytest.mark.parametrize("sensor", ["pose", "scan"])
def test_stale_sensor_stops_movement(motion_node, sensor):
    n, clock = motion_node
    motion_update(n, clock)
    clock.now += 0.6
    if sensor == "pose":
        n.last_scan_time = clock.now
    else:
        n.last_pose_time = clock.now
    n.timer_callback()
    assert n.commands[-1] == (0, 0, 0)
    assert n.phase == "SELECT" and n.drive_start is None


def test_large_heading_drift_realigns_without_resetting_progress_origin(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    start = n.drive_start.copy()
    n.pose[:] = [0.05, 0.0, 40.0]
    motion_update(n, clock)
    assert n.phase == "TURN" and n.commands[-1][0] == 0
    assert n.drive_start == pytest.approx(start)
    n.pose[2] = 0
    motion_update(n, clock)
    assert n.phase == "DRIVE" and n.advance_progress == pytest.approx(0.05)
    n.pose[0] = n.p["check_distance_m"] / 2
    motion_update(n, clock)
    assert n.phase == "DONE"


@pytest.mark.parametrize("simulation,error,expected", [
    (True, 5.2, -0.15),
    (True, -5.2, 0.15),
    (True, 80.0, -0.4),
    (False, 5.2, -math.radians(5.2)),
    (False, -5.2, math.radians(5.2)),
])
def test_minimum_turn_speed_applies_only_in_simulation(motion_node, simulation, error, expected):
    n, clock = motion_node
    n.is_simulation = simulation
    motion_update(n, clock)
    n.pose[2] = error
    # A tight soft margin requires alignment even for a small error.
    points = [[2.0, 0], [0, -0.16]] if abs(error) < 30 else [[2.0, 0]]
    motion_update(n, clock, points)
    assert n.phase == "TURN"
    assert n.commands[-1] == pytest.approx((0, 0, expected))


def test_simulation_minimum_respects_maximum_and_does_not_force_aligned_turns(motion_node):
    n, clock = motion_node
    n.p["max_turn_rps"] = 0.1
    motion_update(n, clock)
    n.pose[2] = 5.2
    motion_update(n, clock, [[2.0, 0], [0, -0.16]])
    assert n.commands[-1][2] == pytest.approx(-0.1)
    n.pose[2] = 2.0
    motion_update(n, clock)
    assert n.phase == "DRIVE"
    assert n.commands[-1][0] > 0 and n.commands[-1][2] == 0


def test_repeat_mode_selects_again_only_after_the_advance(motion_node):
    n, clock = motion_node
    n.single_step = False
    original = n.score_directions
    calls = []
    n.score_directions = lambda *args: (calls.append(True), original(*args))[1]
    motion_update(n, clock)
    assert len(calls) == 1
    n.pose[0] = n.p["check_distance_m"] / 2
    previous_commands = len(n.commands)
    motion_update(n, clock)
    assert n.phase == "DRIVE" and n.commands[-1][0] > 0
    assert len(calls) == 2 and n.completed_steps == 1
    assert n.commands[previous_commands:] == [(n.p["forward_speed_mps"], 0, 0)]
    motion_update(n, clock)
    assert len(calls) == 2 and n.phase == "DRIVE"


def test_advance_uses_course_positions_with_a_nonzero_transform(motion_node):
    n, clock = motion_node
    n.course = dict(origin_x=2.0, origin_y=3.0, rotation_deg=-90.0)
    n.pose[:] = [2, 3, -90]
    motion_update(n, clock)
    assert n.phase == "DRIVE" and n.locked_heading == pytest.approx(0)
    n.pose[1] -= n.p["check_distance_m"] / 2
    motion_update(n, clock)
    assert n.phase == "DONE"
    assert n.advance_progress == pytest.approx(n.p["check_distance_m"] / 2)


def test_invalid_pose_stops_and_publishes_a_usable_snapshot(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    n.pose[0] = math.nan
    n.timer_callback()
    assert n.commands[-1] == (0, 0, 0)
    assert n.frames[-1]["course_pose"] is None
    assert n.frames[-1]["relative_position"] is None


def scan_with_bin(pose):
    """Room scan plus one stationary bin; all ranges come from geometry."""
    scan = box_scan(pose, right=2.0)
    heading = math.radians(pose[2])
    sensor = pose[:2] + PARAMS["lidar_x_m"] * np.array([math.cos(heading), math.sin(heading)])
    angles = (scan.angle_min + np.arange(360) * scan.angle_increment
              + math.radians(PARAMS["lidar_yaw_deg"]) + heading)
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (np.array([0.5, -0.15]) - sensor) / rays
        b = (np.array([0.8, 0.15]) - sensor) / rays
    near, far = np.minimum(a, b).max(axis=1), np.maximum(a, b).min(axis=1)
    bin_ranges = np.where((near > 0) & (far >= near), near, np.inf)
    scan.ranges = np.minimum(np.asarray(scan.ranges), bin_ranges).tolist()
    return scan


def test_two_closed_loop_advances_with_a_bin_keep_their_headings(motion_node):
    n, clock = motion_node
    n.single_step = False
    n.p.update(goal_a_deg=-45.0, check_distance_m=0.4)
    used_curve = False
    for _ in range(600):
        clock.now += 0.05
        n.last_pose_time = clock.now
        n.sub_scan_callback(scan_with_bin(n.pose))
        previous_heading, previous_count = n.locked_heading, n.completed_steps
        n.timer_callback()
        used_curve = used_curve or n.phase == "CURVE"
        if previous_heading is not None and n.completed_steps == previous_count:
            assert n.locked_heading == previous_heading
        speed, _, turn = n.commands[-1]
        heading = math.radians(n.pose[2])
        n.pose[:2] += speed * 0.05 * np.array([math.cos(heading), math.sin(heading)])
        n.pose[2] += math.degrees(turn * 0.05)
        closest = np.clip(n.pose[:2], [0.5, -0.15], [0.8, 0.15])
        assert np.linalg.norm(n.pose[:2] - closest) > n.robot_radius
        if n.completed_steps >= 2:
            break
    assert n.completed_steps == 2
    assert used_curve
    # Reselection can require an in-place turn; it still issues its command immediately.
    assert n.phase in ("TURN", "CURVE", "DRIVE")
    if n.phase == "TURN":
        assert n.commands[-1][0] == 0 and abs(n.commands[-1][2]) > 0
    else:
        assert n.commands[-1][0] > 0


def test_viewer_shows_locked_heading_and_measured_progress(motion_node, viewer_module):
    import matplotlib.pyplot as plt
    n, clock = motion_node
    n.p["goal_a_deg"] = -45
    n.last_pose_time = clock.now
    initialise(n, heading=0)
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        view.update(viewer_packet(n))
        assert "SINGLE STEP: CURVE" in view.aim_text.get_text()
        assert "Heading held; overlap preview live" in view.path_scores.get_text()
        assert view.maze_locked.get_visible()
        expected = np.asarray(n.frames[-1]["motion_curve"])
        assert view.motion_curve.get_xdata() == pytest.approx(expected[:, 0])
        assert view.motion_curve.get_ydata() == pytest.approx(expected[:, 1])
        assert "moving curve clear" in view.path_scores.get_text()
        n.pose[2] = n.locked_heading
        n.sub_scan_callback(box_scan(n.pose))
        n.timer_callback()
        view.update(viewer_packet(n))
        assert "SINGLE STEP: DRIVE" in view.aim_text.get_text()
        direction = math.radians(n.locked_heading)
        n.pose[:2] += n.p["check_distance_m"] / 2 * np.array([math.cos(direction), math.sin(direction)])
        n.sub_scan_callback(box_scan(n.pose))
        n.timer_callback()
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        assert "SINGLE STEP: DONE" in view.aim_text.get_text()
        assert n.commands[-1] == (0, 0, 0)
        view.update(None)
        assert not view.maze_locked.get_visible()
        assert len(view.motion_curve.get_xdata()) == 0
    finally:
        plt.close(view.fig)


def test_deep_straight_path_can_win_despite_virtual_goal(motion_node):
    n, clock = motion_node
    n.p["goal_a_deg"] = 45.0
    points = [[3.11, 0], [0.8 / math.sqrt(2), 0.8 / math.sqrt(2)]]
    motion_update(n, clock, points, [0, 45])
    assert n.locked_heading == pytest.approx(0)
    assert n.phase == "DRIVE"
    paths = n.frames[-1]
    assert all(paths["soft_clear"])
    assert paths["scores"][paths["angles"].index(0)] > paths["scores"][paths["angles"].index(45)]


@pytest.mark.parametrize("sign", [-1, 1])
def test_motion_centres_follow_left_and_right_quarter_circles(preview_node, sign):
    n = preview_node
    # v/w = 0.25 m; a quarter circle ends at (0.25, +/-0.25).
    centres = n.motion_centres(0.1, sign * 0.4, math.pi * 0.25 / 2)
    assert centres[0] == pytest.approx([0, 0])
    assert centres[-1] == pytest.approx([0.25, sign * 0.25])
    # Geometry remains valid when the user tunes the sampling spacing.
    assert np.linalg.norm(centres - [0, sign * 0.25], axis=1) == pytest.approx(0.25)


def test_arc_check_uses_swept_centres_instead_of_straight_ray_or_circle_centre(preview_node):
    n = preview_node
    distance = math.pi * 0.25 / 2
    # A straight-ahead check misses an obstacle at the end of the left-turn arc.
    obstacle = np.array([[0.25, 0.25]])
    assert n.path_is_clear(obstacle, 0, distance)
    assert not n.motion_is_clear(obstacle, 0.1, 0.4, distance)
    # Conversely, an obstacle on the straight ray is well outside the curved sweep.
    obstacle = np.array([[0.35, 0]])
    assert not n.path_is_clear(obstacle, 0, distance)
    assert n.motion_is_clear(obstacle, 0.1, 0.4, distance)
    # The turning-circle centre is not a robot-centre position on the arc.
    assert n.motion_is_clear(np.array([[0, 0.25]]), 0.1, 0.4, distance)


@pytest.mark.parametrize("speed,turn,distance", [(0.1, 0.4, 0.2), (0.1, 0.0, 0.2), (0, 0.4, 0)])
def test_motion_check_honours_hard_and_soft_clearance(preview_node, speed, turn, distance):
    n = preview_node
    points = np.array([[0, -0.15]])
    assert n.motion_is_clear(points, speed, turn, distance)
    assert not n.motion_is_clear(points, speed, turn, distance, radius=0.2)
    assert not n.motion_is_clear(np.empty((0, 2)), speed, turn, distance)


@pytest.mark.parametrize("heading", [-15.0, 15.0, -5.2, 5.2])
def test_small_heading_error_moves_and_steers_without_in_place_minimum(motion_node, heading):
    n, clock = motion_node
    motion_update(n, clock)
    n.pose[2] = heading
    motion_update(n, clock)
    speed, _, turn = n.commands[-1]
    assert n.phase == "CURVE" and 0 < speed < n.p["forward_speed_mps"]
    assert turn == pytest.approx(-math.radians(heading))
    assert "moving curve clear" in n.frames[-1]["status"]
    assert "CURVE: moving curve clear (curve radius=0.200m)" in n.logs


def test_curve_clearance_can_change_without_changing_direction_scoring(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p.update(soft_clearance_m=0.16, curve_clearance_m=0.16)
    # The robot fits beside this return; it can curve left away from it.
    points = np.array([[2.0, 0.0], [0.0, -0.155]])
    angles = np.array([0.0])
    error = math.radians(15)
    selected = n.score_directions(points, angles)
    assert selected["clear"] == [True] and selected["soft_clear"] == [False]
    assert n.choose_motion(points, error, 0.15)[0] == "TURN"

    n.p["curve_clearance_m"] = 0.14
    phase, speed, turn, reason = n.choose_motion(points, error, 0.15)
    assert phase == "CURVE" and speed > 0 and turn > 0
    assert "curve radius=0.140m" in reason
    assert n.score_directions(points, angles) == selected
    assert n.p["soft_clearance_m"] == 0.16

    # Global soft-clearance tuning does not change the moving-curve threshold.
    n.p["soft_clearance_m"] = 0.3
    assert n.choose_motion(points, error, 0.15)[0] == "CURVE"


@pytest.mark.parametrize("curve_radius", [0.0, 0.05])
def test_curve_clearance_cannot_shrink_the_robot_footprint(preview_node, curve_radius):
    n = preview_node
    n.p["curve_clearance_m"] = curve_radius
    points = np.array([[2.0, 0.0], [0.0, -0.1]])
    phase, speed, turn, _ = n.choose_motion(points, math.radians(15), 0.15)
    assert (phase, speed, turn) == ("STOP", 0.0, 0.0)


def test_blocked_curve_aligns_same_target_then_can_leave_tight_soft_margin(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    saved_start = n.drive_start.copy()
    n.pose[2] = -15  # Target remains course 0; steer left toward it.
    points = np.array([[2.0, 0], [0.12, -0.17]])
    assert n.path_is_clear(points, math.radians(15), 0.15)
    assert n.path_is_clear(points, 0, 0, radius=0.2)
    assert not n.motion_is_clear(points, 0.05, math.radians(15), 0.025, radius=0.2)
    n.score_directions = lambda *args: pytest.fail("Blocked curve must keep its selected heading")
    motion_update(n, clock, points)
    assert n.phase == "TURN" and n.commands[-1][0] == 0 and n.commands[-1][2] > 0
    assert n.locked_heading == 0
    assert n.drive_start == pytest.approx(saved_start)
    assert "moving curve blocked; align first" in n.frames[-1]["status"]
    n.pose[2] = 0
    motion_update(n, clock, points)
    assert n.phase == "DRIVE" and n.commands[-1][0] > 0


def test_collision_inside_hard_footprint_stops_a_moving_turn(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    n.pose[2] = 15
    motion_update(n, clock, [[2, 0], [0, 0.1]])
    assert n.commands[-1] == (0, 0, 0)
    assert n.phase == "SELECT" and n.locked_heading is None


def test_curve_prediction_is_short_and_shrinks_to_remaining_advance(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    n.pose[2] = 15
    motion_update(n, clock)
    curve = np.asarray(n.frames[-1]["motion_curve"])
    # At 0.05 m/s, the half-second prediction covers 0.025 m, not another full lookahead.
    assert np.linalg.norm(curve[-1]) <= 0.025
    n.pose[0] = n.p["check_distance_m"] / 2 - 0.001
    motion_update(n, clock)
    curve = np.asarray(n.frames[-1]["motion_curve"])
    assert n.phase == "CURVE" and np.linalg.norm(curve[-1]) <= 0.001 + 1e-12


def test_repeat_handoff_stops_when_next_selection_has_no_clear_path(motion_node):
    n, clock = motion_node
    n.single_step = False
    motion_update(n, clock)
    n.pose[0] = n.p["check_distance_m"] / 2
    motion_update(n, clock, [[0.2, 0]])
    assert n.completed_steps == 1
    assert n.phase == "WAIT_BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert n.locked_heading is None and n.drive_start is None


@pytest.mark.parametrize("speed", [0.1, 0.3])
def test_two_advances_cross_a_hard_clear_narrow_doorway(motion_node, speed):
    n, clock = motion_node
    n.single_step = False
    n.p.update(check_distance_m=0.4, soft_clearance_m=0.2,
               virtual_goal_weight=1.0, forward_speed_mps=speed)
    world_points = np.array([[0.25, 0.15], [0.25, -0.15], [2, 0], [0, -2]])
    for _ in range(200):
        points = body_points(world_points - n.pose[:2], n.pose[2])
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        motion_update(n, clock, points, angles)
        assert n.phase == "DRIVE" and n.locked_heading == pytest.approx(0)
        if n.completed_steps >= 2:
            break
        speed, _, turn = n.commands[-1]
        assert speed > 0 and turn == 0
        n.pose[0] += speed * 0.05
        clock.now += 0.05
        assert np.all(np.linalg.norm(world_points[:2] - n.pose[:2], axis=1) > n.robot_radius)
    assert n.completed_steps == 2
    assert n.pose[0] >= 0.4  # Crossed the doorway at course x=0.25 m.
    assert any("clearance=hard-only" in message for message in n.logs)


def test_trail_saves_start_once_and_copies_positions(preview_node):
    n = preview_node
    initialise(n, heading=0)
    assert np.asarray(n.trail) == pytest.approx(np.array([[0, 0]]))
    n.pose[:] = [0.2, 0.1, 90]
    n.record_trail()  # SELECT is stationary; the start dot must stay unchanged.
    assert np.asarray(n.trail) == pytest.approx(np.array([[0, 0]]))
    assert n.frames[-1]["trail_relative"] == [[0, 0]]


def test_trail_spacing_uses_actual_displacement_in_any_direction(preview_node):
    n = preview_node
    initialise(n, heading=0)
    n.phase = "DRIVE"
    n.pose[:2] = [0.03, 0.039]
    n.record_trail()
    assert len(n.trail) == 1
    n.pose[:2] = [0.03, 0.04]  # A 3-4-5 triangle gives exactly 5 cm displacement.
    n.record_trail()
    assert np.asarray(n.trail) == pytest.approx(np.array([[0, 0], [0.03, 0.04]]))
    n.phase = "CURVE"
    n.pose[:2] = [0.03, 0.049]
    n.record_trail()
    assert len(n.trail) == 2
    n.pose[:2] = [0.03, 0.10]
    n.record_trail()
    assert len(n.trail) == 3


@pytest.mark.parametrize("phase", ["SELECT", "TURN", "DONE"])
def test_turning_or_stopped_phases_do_not_append_trail(preview_node, phase):
    n = preview_node
    initialise(n, heading=0)
    n.phase = phase
    n.pose[:] = [0.06, 0.0, 150]  # Even position noise during a turn cannot add a dot.
    n.record_trail()
    assert len(n.trail) == 1


def test_heading_change_alone_does_not_add_trail(preview_node):
    n = preview_node
    initialise(n, heading=0)
    n.phase = "CURVE"
    for heading in [30, 90, 180, -90, 0]:
        n.pose[2] = heading
        n.record_trail()
    assert len(n.trail) == 1


def test_trail_retains_chronological_order_including_revisits(preview_node):
    n = preview_node
    initialise(n, heading=0)
    n.phase = "DRIVE"
    route = [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0], [0.1, 0]]
    for point in route[1:]:
        n.pose[:2] = point
        n.record_trail()
    assert np.asarray(n.trail) == pytest.approx(np.asarray(route))


def test_timer_records_arrival_before_completing_a_single_step(motion_node):
    n, clock = motion_node
    n.p["check_distance_m"] = 0.4
    motion_update(n, clock)
    n.pose[0] = 0.05
    motion_update(n, clock)
    n.pose[0] = 0.2
    motion_update(n, clock)
    assert n.phase == "DONE"
    assert np.asarray(n.trail) == pytest.approx(np.array([[0, 0], [0.05, 0], [0.2, 0]]))
    assert np.asarray(n.frames[-1]["trail_relative"]) == pytest.approx(np.asarray(n.trail))
    motion_update(n, clock)
    assert len(n.trail) == 3


@pytest.mark.parametrize("fault", ["stale", "invalid"])
def test_unusable_pose_or_sensor_data_does_not_extend_trail(motion_node, fault):
    n, clock = motion_node
    motion_update(n, clock)
    n.pose[0] = 0.1
    if fault == "invalid":
        n.pose[0] = math.nan
    else:
        clock.now += 1.0
    n.timer_callback()
    assert len(n.trail) == 1
    assert n.commands[-1] == (0, 0, 0)


def test_trail_uses_course_conversion_and_survives_segment_switch(motion_node):
    n, clock = motion_node
    n.p["check_distance_m"] = 0.6
    n.course = dict(origin_x=2.0, origin_y=3.0, rotation_deg=-90.0)
    n.pose[:] = [2.4, 3.2, -90]
    motion_update(n, clock)
    start = n.trail[0].copy()
    assert start == pytest.approx([-0.2, 0.4])
    n.pose[1] -= 0.1
    motion_update(n, clock)
    n.switch_x = 0.05
    n.pose[0] -= 0.1
    motion_update(n, clock)
    assert n.segment == "B"
    expected_relative = np.array([[0, 0], [0.1, 0], [0.1, -0.1]])
    assert np.asarray(n.trail) == pytest.approx(start + expected_relative)
    assert np.asarray(n.frames[-1]["trail_relative"]) == pytest.approx(expected_relative)


def test_trail_radius_only_changes_display_data(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    command, heading = n.commands[-1], n.locked_heading
    n.p["trail_radius_m"] = 0.4
    n.pose[0] = 0.05
    motion_update(n, clock)
    assert n.commands[-1] == command and n.locked_heading == heading
    assert n.frames[-1]["trail_radius_m"] == 0.4


def test_viewer_draws_trail_circles_with_fixed_axes_and_keeps_whole_route(preview_node, viewer_module):
    import matplotlib.pyplot as plt
    n = preview_node
    initialise(n, heading=0)
    n.phase = "DRIVE"
    route = np.array([[0, 0], [1, -1], [2, -1], [2, 1], [0, 0]])
    for point in route[1:]:
        n.pose[:2] = point
        n.record_trail()
    n.timer_callback()
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        display = np.column_stack((-route[:, 1], route[:, 0]))
        assert view.maze_trail.get_xdata() == pytest.approx(display[:, 0])
        assert view.maze_trail.get_ydata() == pytest.approx(display[:, 1])
        circles = view.trail_circles.get_paths()
        assert len(circles) == len(route)
        radius = n.p["trail_radius_m"]
        for circle, (x, y) in zip(circles, display):
            bounds = circle.get_extents()
            assert bounds.extents == pytest.approx([x - radius, y - radius, x + radius, y + radius])
            assert view.maze_ax.get_xlim()[0] <= bounds.xmin
            assert view.maze_ax.get_xlim()[1] >= bounds.xmax
            assert view.maze_ax.get_ylim()[0] <= bounds.ymin
            assert view.maze_ax.get_ylim()[1] >= bounds.ymax
        assert "Trail: 5 dots" in view.maze_ax.get_title()
        view.update(None)
        assert len(view.maze_trail.get_xdata()) == 0
        assert view.trail_circles.get_paths() == []
    finally:
        plt.close(view.fig)


@pytest.mark.parametrize("heading,expected", [
    (0, [[-0.3, 0.2], [0, 0.2], [0, 0]]),
    (90, [[0.2, 0.3], [0.2, 0], [0, 0]]),
    (-90, [[-0.2, -0.3], [-0.2, 0], [0, 0]]),
    (180, [[0.3, -0.2], [0, -0.2], [0, 0]]),
])
def test_lidar_trail_follows_robot_translation_and_rotation(preview_node, viewer_module, heading, expected):
    import matplotlib.pyplot as plt
    n = preview_node
    initialise(n, heading=0)
    n.phase = "DRIVE"
    for point in [[0.3, 0], [0.3, -0.2]]:
        n.pose[:2] = point
        n.record_trail()
    n.pose[2] = heading
    n.sub_scan_callback(box_scan(n.pose, right=2.0))
    n.timer_callback()
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        view.update(viewer_packet(n))
        local_trail = np.column_stack(view.lidar_trail.get_data())
        assert local_trail == pytest.approx(np.asarray(expected))
        # The fixed map retains its course axes while the main view follows the robot.
        assert view.maze_trail.get_xdata() == pytest.approx([0, 0, 0.2])
        assert view.maze_trail.get_ydata() == pytest.approx([0, 0.3, 0.3])
        view.update(None)
        assert len(view.lidar_trail.get_xdata()) == 0
    finally:
        plt.close(view.fig)


def test_lidar_trail_handles_older_snapshots_without_trail_data(preview_node, viewer_module):
    import matplotlib.pyplot as plt
    n = preview_node
    initialise(n, heading=0)
    del n.frames[-1]["trail_relative"]
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        assert len(view.lidar_trail.get_xdata()) == 0
        assert len(view.maze_trail.get_xdata()) == 0
    finally:
        plt.close(view.fig)


@pytest.fixture
def blocked_route(motion_node):
    n, clock = motion_node
    n.single_step, n.backtrack_one = False, True
    n.retreat_limit = 1
    n.p.update(check_distance_m=0.4, blockage_confirm_s=0.5,
               retreat_speed_mps=0.05, retreat_reach_m=0.01, retreat_timeout_s=4.0)
    n.pose[:] = [0.2, 0, 0]
    n.start_pose = np.zeros(3)
    n.right_distance, n.switch_x = 0.18, 0.36
    n.trail = [np.array([x, 0.0]) for x in [0, 0.05, 0.10, 0.15, 0.20]]
    return n, clock


def blocked_update(n, clock, points=((0.2, 0), (-1, 0), (0, 0.18), (0, -0.18)), angles=(0, -180)):
    motion_update(n, clock, points, angles)


def begin_one_retreat(n, clock, **kwargs):
    blocked_update(n, clock, **kwargs)
    clock.now += n.p["blockage_confirm_s"]
    blocked_update(n, clock, **kwargs)


def test_detection_stops_immediately_then_latches_after_persistent_no_candidate(blocked_route):
    n, clock = blocked_route
    n.backtrack_one = False
    n.retreat_limit = 0
    blocked_update(n, clock)
    assert n.phase == "WAIT_BLOCKED" and n.commands[-1] == (0, 0, 0)
    clock.now += 0.49
    blocked_update(n, clock)
    assert n.phase == "WAIT_BLOCKED"
    clock.now += 0.02
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and "front blockage confirmed" in n.frames[-1]["status"]
    blocked_update(n, clock, points=((2, 0), (-1, 0)))
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert all(command[0] >= 0 for command in n.commands)


def test_temporary_blockage_clearing_resets_confirmation_timer(blocked_route):
    n, clock = blocked_route
    blocked_update(n, clock)
    clock.now += 0.3
    blocked_update(n, clock, points=((2, 0), (-1, 0)))
    assert n.phase == "DRIVE" and n.blocked_since is None
    clock.now += 1.0
    blocked_update(n, clock)  # Interrupt the committed drive first.
    assert n.phase == "SELECT"
    clock.now += 0.05
    blocked_update(n, clock)
    assert n.phase == "WAIT_BLOCKED" and n.retreat_index is None


@pytest.mark.parametrize("points,angles", [((), ()), (((0.2, 0), (-1, 0)), (-180,))])
def test_missing_front_scan_support_is_not_a_confirmed_blockage(blocked_route, points, angles):
    n, clock = blocked_route
    points = np.asarray(points).reshape(-1, 2)
    begin_one_retreat(n, clock, points=points, angles=angles)
    assert n.phase == "SELECT" and n.blocked_since is None
    assert n.retreat_index is None
    assert "no supported front scan directions" in n.frames[-1]["status"]
    assert all(command[0] == 0 for command in n.commands)


def test_current_footprint_violation_is_reported_without_reverse(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock, points=((0.2, 0), (-1, 0), (0, 0.1)))
    assert n.phase == "BLOCKED" and n.retreat_index is None
    assert "current hard footprint violated" in n.frames[-1]["status"]
    assert all(command == (0, 0, 0) for command in n.commands)


def test_one_retreat_uses_latest_earlier_dot_and_stops_without_rescoring(blocked_route):
    n, clock = blocked_route
    original_trail = np.asarray(n.trail).copy()
    begin_one_retreat(n, clock)
    assert n.phase == "RETREAT" and n.retreat_index == 3
    assert n.commands[-1] == (-0.05, 0, 0)
    n.score_directions = lambda *args: pytest.fail("Retreat must not rescore normal candidates")
    n.pose[0] = 0.159
    blocked_update(n, clock)
    assert n.phase == "RETREAT_DONE" and n.commands[-1] == (0, 0, 0)
    assert n.trail_reached == [3]
    assert np.asarray(n.trail) == pytest.approx(original_trail)
    for _ in range(3):
        blocked_update(n, clock)
        assert n.phase == "RETREAT_DONE" and n.commands[-1] == (0, 0, 0)
    clock.now += 1.0
    n.timer_callback()
    assert n.phase == "RETREAT_DONE" and n.trail_reached == [3]


def test_retreat_target_order_does_not_pick_an_old_spatially_nearer_visit(blocked_route):
    n, clock = blocked_route
    n.trail[0] = np.array([0.199, 0])
    begin_one_retreat(n, clock)
    assert n.retreat_index == 3
    assert n.trail[n.retreat_index] == pytest.approx([0.15, 0])


@pytest.mark.parametrize("rear_angle", [-180, 180])
def test_reverse_scan_support_wraps_at_both_pi_boundaries(blocked_route, rear_angle):
    n, clock = blocked_route
    begin_one_retreat(n, clock, angles=(0, rear_angle))
    assert n.phase == "RETREAT" and n.commands[-1][0] < 0


def test_no_rear_scan_support_refuses_retreat(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock, angles=(0,))
    assert n.phase == "BLOCKED" and "no supported rear scan" in n.frames[-1]["status"]
    assert all(command[0] == 0 for command in n.commands)


def test_obstacle_in_reverse_sweep_refuses_retreat(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock, points=((0.2, 0), (-0.13, 0)))
    assert n.phase == "BLOCKED" and "reverse path blocked" in n.frames[-1]["status"]
    assert all(command[0] == 0 for command in n.commands)


def test_reverse_uses_hard_clearance_in_a_tight_passage(blocked_route):
    n, clock = blocked_route
    points = np.array([[0.2, 0], [-0.22, 0], [0, 0.18], [0, -0.18]])
    assert not n.path_is_clear(points, math.pi, 0.05, radius=0.2)
    begin_one_retreat(n, clock, points=points)
    assert n.phase == "RETREAT" and n.commands[-1][0] < 0


def test_new_rear_obstacle_stops_and_does_not_restart_when_it_clears(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock)
    blocked_update(n, clock, points=((0.2, 0), (-0.13, 0)))
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert n.trail_reached == []


@pytest.mark.parametrize("target", [[0.15, 0.03], [0.25, 0]])
def test_one_dot_test_refuses_bends_or_a_target_ahead(blocked_route, target):
    n, clock = blocked_route
    n.trail[3] = np.asarray(target)
    begin_one_retreat(n, clock)
    assert n.phase == "BLOCKED" and "straight reverse line" in n.frames[-1]["status"]
    assert all(command == (0, 0, 0) for command in n.commands)


def test_no_previous_dot_refuses_retreat(blocked_route):
    n, clock = blocked_route
    n.trail = [n.pose[:2].copy()]
    begin_one_retreat(n, clock)
    assert n.phase == "BLOCKED" and "no earlier breadcrumb" in n.frames[-1]["status"]


@pytest.mark.parametrize("sensor", ["pose", "scan"])
def test_stale_sensor_during_retreat_holds_stop_after_data_returns(blocked_route, sensor):
    n, clock = blocked_route
    begin_one_retreat(n, clock)
    clock.now += 0.6
    if sensor == "pose":
        n.last_scan_time = clock.now
    else:
        n.last_pose_time = clock.now
    n.timer_callback()
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert "stale" in n.frames[-1]["status"]
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)


def test_invalid_pose_during_retreat_holds_stop(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock)
    n.pose[0] = math.nan
    n.timer_callback()
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert n.frames[-1]["course_pose"] is None


def test_retreat_timeout_stops_a_robot_that_has_not_moved(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock)
    clock.now += 4.1
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and "timed out" in n.frames[-1]["status"]
    assert n.commands[-1] == (0, 0, 0) and n.trail_reached == []


def test_retreat_uses_course_positions_with_nonidentity_conversion(blocked_route):
    n, clock = blocked_route
    n.course = dict(origin_x=2, origin_y=3, rotation_deg=-90)
    n.pose[:] = [2, 2.8, -90]  # Course pose is (0.2, 0, 0).
    begin_one_retreat(n, clock)
    assert n.phase == "RETREAT" and n.retreat_index == 3
    n.pose[1] = 2.85
    blocked_update(n, clock)
    assert n.phase == "RETREAT_DONE" and n.trail_reached == [3]


def u_dead_end_scan(pose, front=0.65, half_width=0.18):
    """Narrow corridor with a closed front and a distant rear boundary."""
    heading = math.radians(pose[2])
    sensor = pose[:2] + PARAMS["lidar_x_m"] * np.array([math.cos(heading), math.sin(heading)])
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    world = angles + math.radians(PARAMS["lidar_yaw_deg"]) + heading
    dx, dy = np.cos(world), np.sin(world)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (np.where(dx > 0, front, -2.0) - sensor[0]) / dx
        ty = (np.where(dy > 0, half_width, -half_width) - sensor[1]) / dy
    msg = LaserScan()
    msg.angle_min, msg.angle_increment = -math.pi, math.pi / 180
    msg.range_min, msg.range_max = 0.05, 10.0
    msg.ranges = np.minimum(tx, ty).tolist()
    return msg


@pytest.mark.parametrize("backtrack", [False, True])
@pytest.mark.parametrize("side_weight", [0.0, PARAMS["side_clearance_weight"]])
def test_closed_loop_u_dead_end_detection_and_one_dot_retreat(motion_node, backtrack, side_weight):
    n, clock = motion_node
    n.single_step, n.backtrack_one = False, backtrack
    n.retreat_limit = 1 if backtrack else 0
    n.p.update(check_distance_m=0.4, similar_score_tolerance=0.0,
               side_clearance_weight=side_weight)
    saw_wait = saw_reverse = False
    frozen_trail = None
    for _ in range(240):
        clock.now += 0.05
        n.last_pose_time = clock.now
        n.sub_scan_callback(u_dead_end_scan(n.pose))
        n.timer_callback()
        saw_wait = saw_wait or n.phase == "WAIT_BLOCKED"
        speed, _, turn = n.commands[-1]
        if n.phase == "RETREAT":
            saw_reverse = True
            if frozen_trail is None:
                frozen_trail = np.asarray(n.trail).copy()
            assert speed < 0 and turn == 0
            assert np.asarray(n.trail) == pytest.approx(frozen_trail)
        if n.phase in ("BLOCKED", "RETREAT_DONE"):
            break
        heading = math.radians(n.pose[2])
        n.pose[:2] += speed * 0.05 * np.array([math.cos(heading), math.sin(heading)])
        n.pose[2] += math.degrees(turn * 0.05)
        assert 0.65 - n.pose[0] > n.robot_radius
        assert 0.18 - abs(n.pose[1]) > n.robot_radius
    assert saw_wait
    if backtrack:
        assert saw_reverse and n.phase == "RETREAT_DONE"
        assert len(n.trail_reached) == 1
        assert np.linalg.norm(n.pose[:2] - n.trail[n.retreat_index]) <= n.p["retreat_reach_m"]
    else:
        assert not saw_reverse and n.phase == "BLOCKED"
        assert "front blockage confirmed" in n.frames[-1]["status"]
    assert n.commands[-1] == (0, 0, 0)


def test_stale_data_resets_pending_blockage_confirmation(blocked_route):
    n, clock = blocked_route
    blocked_update(n, clock)
    clock.now += 0.6
    n.timer_callback()
    assert n.phase == "SELECT" and n.blocked_since is None
    blocked_update(n, clock)
    assert n.phase == "WAIT_BLOCKED" and n.retreat_index is None


def test_reusing_one_scan_does_not_confirm_persistent_blockage(blocked_route):
    n, clock = blocked_route
    blocked_update(n, clock)
    clock.now += 0.5
    n.last_pose_time = clock.now
    n.timer_callback()  # Pose updated, but no new scan arrived.
    assert n.phase == "WAIT_BLOCKED" and n.retreat_index is None
    assert n.commands[-1] == (0, 0, 0)
    clock.now += 0.01
    n.last_pose_time = clock.now
    n.timer_callback()
    assert n.phase == "SELECT" and "stale" in n.frames[-1]["status"]
    assert all(command[0] == 0 for command in n.commands)


def test_retreat_stops_if_the_target_was_overshot(blocked_route):
    n, clock = blocked_route
    begin_one_retreat(n, clock)
    n.pose[0] = 0.13
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert n.trail_reached == []


def test_viewer_shows_one_retreat_target_then_red_reached_dot(blocked_route, viewer_module):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    n, clock = blocked_route
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    def packet():
        return {"paths": "Paths: " + json.dumps(n.frames[-1]), "scan": None}
    try:
        blocked_update(n, clock)
        view.update(packet())
        assert "WAIT_BLOCKED" in view.path_scores.get_text()
        clock.now += n.p["blockage_confirm_s"]
        blocked_update(n, clock)
        view.update(packet())
        view.fig.canvas.draw()
        assert "ONE-DOT TEST: RETREAT" in view.aim_text.get_text()
        assert view.carrot.get_xdata() == pytest.approx([-0.05])
        assert view.carrot.get_ydata() == pytest.approx([0])
        assert view.maze_retreat_target.get_xdata() == pytest.approx([0])
        assert view.maze_retreat_target.get_ydata() == pytest.approx([0.15])
        n.pose[0] = 0.159
        blocked_update(n, clock)
        view.update(packet())
        view.fig.canvas.draw()
        assert "RETREAT_DONE" in view.aim_text.get_text()
        assert not view.carrot.get_visible()
        assert view.maze_reached.get_xdata() == pytest.approx([0])
        assert view.maze_reached.get_ydata() == pytest.approx([0.15])
        assert view.lidar_reached.get_xdata() == pytest.approx([-0.009])
        assert view.lidar_reached.get_ydata() == pytest.approx([0])
        assert view.trail_circles.get_edgecolors()[3, :3] == pytest.approx(to_rgba("crimson")[:3])
        assert len(view.maze_retreat_target.get_xdata()) == 0
        view.update(None)
        assert len(view.maze_reached.get_xdata()) == 0
        assert len(view.lidar_reached.get_xdata()) == 0
    finally:
        plt.close(view.fig)


def test_multi_retreat_reaches_three_dots_in_reverse_order_then_holds(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    original = np.asarray(n.trail).copy()
    begin_one_retreat(n, clock)
    n.score_directions = lambda *args: pytest.fail("Multi-dot retreat must not reselect forward headings")
    for count, index in enumerate([3, 2, 1], start=1):
        assert n.phase == "RETREAT" and n.retreat_index == index
        n.pose[:2] = n.trail[index]
        clock.now += 0.5
        blocked_update(n, clock)
        assert n.trail_reached == [3, 2, 1][:count]
        assert n.commands[-1] == (0, 0, 0)
        if count < 3:
            assert n.phase == "RETREAT" and n.retreat_index == index - 1
            blocked_update(n, clock)
            assert n.commands[-1] == (-0.05, 0, 0)
    assert n.phase == "RETREAT_DONE"
    assert np.asarray(n.trail) == pytest.approx(original)
    for _ in range(3):
        blocked_update(n, clock)
        assert n.phase == "RETREAT_DONE" and n.commands[-1] == (0, 0, 0)
        assert n.trail_reached == [3, 2, 1]


def test_multi_retreat_timeout_resets_for_each_target(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    begin_one_retreat(n, clock)
    for index in [3, 2, 1]:
        clock.now += 3.5  # Each leg is under 4 s; the whole retreat is longer than 4 s.
        n.pose[:2] = n.trail[index]
        blocked_update(n, clock)
    assert n.phase == "RETREAT_DONE" and n.trail_reached == [3, 2, 1]


def test_multi_retreat_stops_at_trail_start_when_fewer_dots_exist(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 10
    begin_one_retreat(n, clock)
    for index in [3, 2, 1, 0]:
        n.pose[:2] = n.trail[index]
        blocked_update(n, clock)
    assert n.phase == "RETREAT_DONE" and n.retreat_index == 0
    assert n.trail_reached == [3, 2, 1, 0]
    assert "trail start reached" in n.frames[-1]["status"]
    assert "4/10" in n.frames[-1]["status"]
    blocked_update(n, clock)
    assert n.commands[-1] == (0, 0, 0)


def test_multi_retreat_checks_new_rear_segment_before_moving(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    begin_one_retreat(n, clock)
    n.pose[:2] = n.trail[3]
    blocked_update(n, clock)
    assert n.retreat_index == 2 and n.commands[-1] == (0, 0, 0)
    blocked_update(n, clock, points=((0.2, 0), (-0.13, 0)))
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert n.trail_reached == [3]
    assert "reverse path blocked" in n.frames[-1]["status"]
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and n.trail_reached == [3]


def test_multi_retreat_stops_before_a_later_bend(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    n.trail[2] = np.array([0.1, 0.03])
    begin_one_retreat(n, clock)
    n.pose[:2] = n.trail[3]
    blocked_update(n, clock)
    assert n.retreat_index == 2 and n.trail_reached == [3]
    blocked_update(n, clock)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    assert "straight reverse line" in n.frames[-1]["status"]


def test_multi_retreat_requires_rear_coverage_again_after_arrival(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    begin_one_retreat(n, clock)
    n.pose[:2] = n.trail[3]
    blocked_update(n, clock)
    blocked_update(n, clock, angles=(0,))
    assert n.phase == "BLOCKED" and "no supported rear scan" in n.frames[-1]["status"]
    assert n.commands[-1] == (0, 0, 0) and n.trail_reached == [3]


def test_multi_retreat_does_not_count_arrival_twice(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    begin_one_retreat(n, clock)
    n.pose[:2] = n.trail[3]
    blocked_update(n, clock)
    for _ in range(3):
        blocked_update(n, clock)
        assert n.retreat_index == 2 and n.trail_reached == [3]


def test_multi_retreat_uses_converted_course_positions_for_every_dot(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    n.course = dict(origin_x=2, origin_y=3, rotation_deg=-90)
    n.pose[:] = [2, 2.8, -90]
    begin_one_retreat(n, clock)
    for index in [3, 2, 1]:
        n.pose[1] = 3.0 - n.trail[index][0]
        blocked_update(n, clock)
    assert n.phase == "RETREAT_DONE" and n.trail_reached == [3, 2, 1]


def test_general_retreat_mode_can_request_just_one_dot(blocked_route):
    n, clock = blocked_route
    n.backtrack_one = False
    begin_one_retreat(n, clock)
    n.pose[:2] = n.trail[3]
    blocked_update(n, clock)
    assert n.phase == "RETREAT_DONE" and n.trail_reached == [3]


def test_closed_loop_u_dead_end_reverses_three_dots_and_stops(motion_node):
    n, clock = motion_node
    n.single_step, n.retreat_limit = False, 3
    n.p.update(check_distance_m=0.4, similar_score_tolerance=0.0)
    saw_wait = False
    reverse_targets = []
    frozen_trail = None
    for _ in range(400):
        clock.now += 0.05
        n.last_pose_time = clock.now
        n.sub_scan_callback(u_dead_end_scan(n.pose))
        n.timer_callback()
        saw_wait = saw_wait or n.phase == "WAIT_BLOCKED"
        if n.phase == "RETREAT":
            if frozen_trail is None:
                frozen_trail = np.asarray(n.trail).copy()
            assert np.asarray(n.trail) == pytest.approx(frozen_trail)
            if not reverse_targets or reverse_targets[-1] != n.retreat_index:
                reverse_targets.append(n.retreat_index)
        if n.phase in ("BLOCKED", "RETREAT_DONE"):
            break
        speed, _, turn = n.commands[-1]
        if n.phase == "RETREAT":
            assert speed <= 0 and turn == 0
        heading = math.radians(n.pose[2])
        n.pose[:2] += speed * 0.05 * np.array([math.cos(heading), math.sin(heading)])
        n.pose[2] += math.degrees(turn * 0.05)
        assert 0.65 - n.pose[0] > n.robot_radius
        assert 0.18 - abs(n.pose[1]) > n.robot_radius
    assert saw_wait and n.phase == "RETREAT_DONE"
    assert len(n.trail_reached) == 3 and reverse_targets == n.trail_reached
    assert np.diff(n.trail_reached) == pytest.approx([-1, -1])
    assert n.commands[-1] == (0, 0, 0)


@pytest.mark.parametrize("flag", ["--backtrack-dots", "--backtrack-curves"])
@pytest.mark.parametrize("count", ["0", "-2", "not-a-number"])
def test_cli_refuses_invalid_dot_counts_before_ros_initialisation(monkeypatch, count, flag):
    from rb2301_fp.obstacle_course import main
    monkeypatch.setattr("rb2301_fp.obstacle_course.rclpy.init",
                        lambda **kwargs: pytest.fail("Invalid count must be rejected before ROS starts"))
    with pytest.raises(SystemExit) as error:
        main([flag, count])
    assert error.value.code == 2


def test_viewer_displays_multi_dot_target_and_reached_progress(blocked_route, viewer_module):
    import matplotlib.pyplot as plt
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 3
    begin_one_retreat(n, clock)
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    def packet():
        return {"paths": "Paths: " + json.dumps(n.frames[-1]), "scan": None}
    try:
        view.update(packet())
        assert "MULTI-DOT TEST: RETREAT" in view.aim_text.get_text()
        assert "0 / 3" in view.path_scores.get_text()
        for count, index in enumerate([3, 2, 1], start=1):
            n.pose[:2] = n.trail[index]
            blocked_update(n, clock)
            view.update(packet())
            view.fig.canvas.draw()
            assert f"{count} / 3" in view.path_scores.get_text()
            assert len(view.maze_reached.get_xdata()) == count
            assert len(view.lidar_reached.get_xdata()) == count
            if count < 3:
                assert view.maze_retreat_target.get_ydata() == pytest.approx([n.trail[index - 1][0]])
        assert "RETREAT_DONE" in view.aim_text.get_text()
        assert not view.carrot.get_visible()
    finally:
        plt.close(view.fig)


@pytest.fixture
def curved_route(blocked_route):
    n, clock = blocked_route
    n.retreat_curves = True
    n.p.update(retreat_turn_limit_deg=45.0, retreat_curve_timeout_s=8.0, max_turn_rps=0.4)
    return n, clock


def curve_update(n, clock, target=(-0.05, 0.03),
                 points=((0.2, 0), (-2, 0), (0, 0.3), (0, -0.3)), angles=None):
    """One reverse update with an explicit robot-local target and current scan geometry."""
    n.trail[3] = n.pose[:2] + np.asarray(target)
    n.retreat_index, n.retreat_started, n.phase = 3, clock.now, "RETREAT"
    angles = np.arange(-180, 180) if angles is None else angles
    motion_update(n, clock, points, angles)


@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_geometry_predicts_reverse_quarter_circles(preview_node, sign):
    n = preview_node
    distance = math.pi * 0.25 / 2
    centres = n.motion_centres(-0.1, sign * 0.4, distance)
    assert centres[-1] == pytest.approx([-0.25, -sign * 0.25])
    obstacle = np.array([[-0.25, -sign * 0.25]])
    assert n.path_is_clear(obstacle, math.pi, distance)
    assert not n.motion_is_clear(obstacle, -0.1, sign * 0.4, distance)
    assert not n.path_is_clear(np.array([[-0.35, 0]]), math.pi, distance)
    assert n.motion_is_clear(np.array([[-0.35, 0]]), -0.1, sign * 0.4, distance)


def test_negative_straight_speed_checks_rear_travel_instead_of_only_current_circle(preview_node):
    n = preview_node
    points = np.array([[-0.2, 0]])
    assert n.motion_is_clear(points, 0, 0.2, 0)
    assert n.motion_is_clear(points, 0.05, 0, 0.15)
    assert not n.motion_is_clear(points, -0.05, 0, 0.15)


@pytest.mark.parametrize("target", [(-0.05, 0.03), (-0.05, -0.03), (-0.05, 0.0)])
def test_reverse_pursuit_arc_reaches_target_with_correct_steering_sign(curved_route, target):
    n, _ = curved_route
    distance = math.hypot(*target)
    speed, turn, travel, _ = n.curved_retreat_command(*target, distance)
    assert speed < 0 and abs(speed) <= n.p["retreat_speed_mps"]
    assert abs(turn) <= n.p["max_turn_rps"]
    assert np.sign(turn) == -np.sign(target[1])
    assert n.motion_centres(speed, turn, travel)[-1] == pytest.approx(target)
    if target[1]:
        assert abs(speed) < n.p["retreat_speed_mps"]  # A tight bend needs slower reverse.


def test_curve_alignment_rotates_rear_toward_large_side_offset(curved_route):
    n, _ = curved_route
    speed, turn, travel, action = n.curved_retreat_command(-0.005, 0.05, math.hypot(0.005, 0.05))
    assert speed == 0 and turn < 0 and travel == 0
    assert "align rear" in action


def test_actual_reverse_arc_collision_is_not_missed_by_straight_check(curved_route):
    n, clock = curved_route
    points = np.array([[0.2, 0], [-2, 0], [-0.05, 0.135]])
    assert n.path_is_clear(points, math.pi, 0.06)
    curve_update(n, clock, points=points)
    assert n.phase == "BLOCKED" and "reverse curve blocked" in n.frames[-1]["status"]
    assert n.commands[-1] == (0, 0, 0)


def test_curved_retreat_can_accept_a_safe_arc_when_straight_back_is_blocked(curved_route):
    n, clock = curved_route
    points = np.array([[0.2, 0], [-0.35, 0]])
    assert not n.path_is_clear(points, math.pi, 0.35)
    curve_update(n, clock, target=(-0.25, 0.25), points=points)
    assert n.phase == "RETREAT" and n.commands[-1][0] < 0 and n.commands[-1][2] < 0


def test_curved_mode_still_checks_full_straight_reverse_segment_when_aligned(curved_route):
    n, clock = curved_route
    curve_update(n, clock, target=(-0.05, 0), points=((0.2, 0), (-0.13, 0)))
    assert n.phase == "BLOCKED" and "reverse path blocked" in n.frames[-1]["status"]


def test_curved_mode_requires_scan_support_toward_its_target(curved_route):
    n, clock = curved_route
    curve_update(n, clock, angles=(0, -180))
    assert n.phase == "BLOCKED"
    assert "no supported scan direction toward retreat target" in n.frames[-1]["status"]
    assert n.commands[-1] == (0, 0, 0)


def test_curved_mode_refuses_a_target_more_than_arrival_tolerance_ahead(curved_route):
    n, clock = curved_route
    curve_update(n, clock, target=(0.03, 0.02))
    assert n.phase == "BLOCKED" and "target is ahead" in n.frames[-1]["status"]


def test_curved_mode_uses_its_own_eight_second_timeout(curved_route):
    n, clock = curved_route
    curve_update(n, clock, target=(-0.05, 0))
    clock.now += 4.1
    motion_update(n, clock, ((0.2, 0), (-2, 0)), np.arange(-180, 180))
    assert n.phase == "RETREAT"
    clock.now += 4.0
    motion_update(n, clock, ((0.2, 0), (-2, 0)), np.arange(-180, 180))
    assert n.phase == "BLOCKED" and "timed out" in n.frames[-1]["status"]


def curved_corridor_scan(pose, side=1, radius=0.3, half_width=0.18):
    """Full LiDAR returns from the two circular walls of a bent corridor."""
    heading = math.radians(pose[2])
    sensor = pose[:2] + PARAMS["lidar_x_m"] * np.array([math.cos(heading), math.sin(heading)])
    centre_offset = sensor - np.array([0.0, side * radius])
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    world = angles + math.radians(PARAMS["lidar_yaw_deg"]) + heading
    rays = np.column_stack((np.cos(world), np.sin(world)))
    b = rays @ centre_offset
    ranges = np.full(360, np.inf)
    for wall_radius in [radius - half_width, radius + half_width]:
        discriminant = b**2 - (centre_offset @ centre_offset - wall_radius**2)
        root = np.sqrt(np.maximum(0, discriminant))
        for candidate in [-b - root, -b + root]:
            ranges = np.minimum(ranges, np.where((discriminant >= 0) & (candidate > 0), candidate, np.inf))
    msg = LaserScan()
    msg.angle_min, msg.angle_increment = -math.pi, math.pi / 180
    msg.range_min, msg.range_max = 0.05, 10.0
    msg.ranges = ranges.tolist()
    return msg


def prepare_bent_trail(n, clock, side=1, limit=3):
    n.single_step, n.backtrack_one, n.retreat_limit, n.retreat_curves = False, False, limit, True
    n.p.update(retreat_turn_limit_deg=45.0, retreat_curve_timeout_s=8.0,
               retreat_speed_mps=0.05, max_turn_rps=0.4)
    theta = np.linspace(0, math.pi / 2, 10)
    n.trail = [np.array([0.3 * math.sin(a), side * 0.3 * (1 - math.cos(a))]) for a in theta]
    n.pose[:] = [0.3, side * 0.3, side * 90]
    n.start_pose = np.zeros(3)
    n.right_distance, n.switch_x = 0.18, 0.36
    n.update_segment()
    n.last_pose_time = clock.now
    n.sub_scan_callback(curved_corridor_scan(n.pose, side))
    points, angles = n.scan_geometry()
    n.start_retreat(points, angles)


@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("limit", [3, 9])
def test_closed_loop_retreat_tracks_left_and_right_bends(motion_node, side, limit):
    n, clock = motion_node
    prepare_bent_trail(n, clock, side, limit)
    original_trail = np.asarray(n.trail).copy()
    saw_reverse_steering = False
    for _ in range(800):
        assert n.phase != "BLOCKED", n.frames[-1]["status"]
        if n.phase == "RETREAT_DONE":
            break
        speed, _, turn = n.commands[-1]
        assert speed <= 0
        saw_reverse_steering = saw_reverse_steering or (speed < 0 and side * turn < 0)
        heading = math.radians(n.pose[2])
        n.pose[:2] += speed * 0.05 * np.array([math.cos(heading), math.sin(heading)])
        n.pose[2] += math.degrees(turn * 0.05)
        radial_distance = np.linalg.norm(n.pose[:2] - [0, side * 0.3])
        assert abs(radial_distance - 0.3) < 0.015
        assert min(radial_distance - 0.12, 0.48 - radial_distance) > n.robot_radius
        clock.now += 0.05
        n.last_pose_time = clock.now
        n.sub_scan_callback(curved_corridor_scan(n.pose, side))
        n.timer_callback()
        assert np.asarray(n.trail) == pytest.approx(original_trail)
    assert saw_reverse_steering and n.phase == "RETREAT_DONE"
    assert n.trail_reached == list(range(8, 8 - limit, -1))
    assert side * n.pose[2] < 75 if limit == 3 else abs(n.pose[2]) < 5
    assert n.commands[-1] == (0, 0, 0)


def test_closed_loop_curved_retreat_aligns_before_tight_side_target(curved_route):
    n, clock = curved_route
    n.trail[3] = n.pose[:2] + [-0.005, 0.05]
    n.sub_scan_callback(box_scan(n.pose, right=2.0))
    n.last_pose_time = clock.now
    n.update_segment()
    n.start_retreat(*n.scan_geometry())
    saw_align = saw_curve = False
    for _ in range(240):
        assert n.phase != "BLOCKED", n.frames[-1]["status"]
        if n.phase == "RETREAT_DONE":
            break
        speed, _, turn = n.commands[-1]
        saw_align = saw_align or (speed == 0 and turn < 0)
        saw_curve = saw_curve or (speed < 0 and turn < 0)
        heading = math.radians(n.pose[2])
        n.pose[:2] += speed * 0.05 * np.array([math.cos(heading), math.sin(heading)])
        n.pose[2] += math.degrees(turn * 0.05)
        clock.now += 0.05
        n.last_pose_time = clock.now
        n.sub_scan_callback(box_scan(n.pose, right=2.0))
        n.timer_callback()
    assert saw_align and saw_curve and n.phase == "RETREAT_DONE"
    assert n.trail_reached == [3]


def test_viewer_shows_reverse_arc_and_steering_commands(motion_node, viewer_module):
    import matplotlib.pyplot as plt
    n, clock = motion_node
    prepare_bent_trail(n, clock)
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        assert "CURVED RETREAT TEST" in view.aim_text.get_text()
        assert "3-dot curved retreat" in view.path_scores.get_text()
        assert "reverse curve toward breadcrumb" in view.path_scores.get_text()
        curve = np.asarray(n.frames[-1]["motion_curve"])
        assert view.motion_curve.get_xdata() == pytest.approx(curve[:, 0])
        assert view.motion_curve.get_ydata() == pytest.approx(curve[:, 1])
        assert curve[-1][0] < 0 and curve[-1][1] > 0
        assert [view.carrot.get_xdata()[0], view.carrot.get_ydata()[0]] == pytest.approx(curve[-1])
    finally:
        plt.close(view.fig)


@pytest.fixture
def turn_route(blocked_route):
    n, clock = blocked_route
    n.backtrack_one, n.retreat_limit = False, 0
    n.recover = n.retreat_curves = True
    n.retreat_index, n.retreat_started, n.phase = 3, clock.now, "RETREAT"
    n.pose[:] = [0.15, 0, 0]
    n.red_last_position = np.array([0.2, 0.0])
    return n, clock


def open_turn_scene():
    return np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float), [-180, -90, 0, 90]


def test_red_segments_have_fixed_course_coordinates_and_point_toward_failure(turn_route):
    n, clock = turn_route
    n.record_red_position(n.pose[:2], finish=True)
    older, failed = n.red_segments[0]
    assert older == pytest.approx([0.15, 0])
    assert failed == pytest.approx([0.20, 0])
    n.pose[:] = [5, 6, 90]
    assert n.red_segments[0][0] == pytest.approx([0.15, 0])
    assert n.red_segments[0][1] == pytest.approx([0.20, 0])


@pytest.mark.parametrize("yaw,angle,blocked", [(0, 0, True), (0, 180, False),
                                               (0, 90, False), (90, -90, True),
                                               (90, 0, False)])
def test_red_memory_rejects_toward_failure_but_allows_outward_and_crossing(preview_node, yaw, angle, blocked):
    n = preview_node
    n.pose = np.array([0.2, 0, float(yaw)])
    n.red_segments = [(np.array([0.0, 0]), np.array([0.5, 0]))]
    assert n.path_returns_to_red(n.straight_centres(math.radians(angle), 0.4)) is blocked


def test_red_memory_checks_whole_path_even_when_endpoint_is_outside_red(preview_node):
    n = preview_node
    n.pose = np.array([-0.2, -math.sqrt(0.12), 0])
    n.red_segments = [(np.array([0.0, -0.4]), np.array([0.0, 0.4]))]
    path = n.straight_centres(math.pi / 3, 0.8)
    assert abs((path[-1] + n.pose[:2])[0]) > n.p["trail_radius_m"]
    assert n.path_returns_to_red(path)


def test_front_scorer_filters_red_returns_without_scoring_rear_candidates(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p["goal_a_deg"] = 0
    n.red_segments = [(np.zeros(2), np.array([0.4, 0]))]
    paths = n.score_directions(np.array([[3, 0], [0, 3], [0, -3]]), np.radians(np.arange(-180, 180)))
    assert min(paths["angles"]) == -120 and max(paths["angles"]) == 120
    forward = paths["angles"].index(0)
    assert paths["clear"][forward] and paths["red_blocked"][forward]
    assert forward not in paths["eligible"]
    assert not paths["red_blocked"][paths["selected_index"]]


def test_committed_forward_motion_cannot_override_red_memory(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.red_segments = [(np.zeros(2), np.array([0.4, 0]))]
    phase, speed, turn, reason = n.choose_motion(np.array([[2.0, 0]]), 0.0, 0.2)
    assert phase == "STOP" and speed == turn == 0
    assert "red failed trail" in reason


def test_recovery_waits_for_full_turn_envelope_not_small_driving_radius(turn_route):
    n, clock = turn_route
    points = np.array([[1, 0], [-1, 0], [0, 0.18], [0, -0.18]])
    motion_update(n, clock, points, [-180, -90, 0, 90])
    assert n.phase == "RETREAT" and n.retreat_index == 2
    assert len(n.red_segments) == 1 and n.trail_reached == [3]
    assert any("TURN SPACE:" in line and "blocked" in line for line in n.logs)


def test_turnaround_holds_one_heading_and_marks_no_new_green_dots(turn_route):
    n, clock = turn_route
    points, angles = open_turn_scene()
    original = np.array(n.trail)
    motion_update(n, clock, points, angles)
    target = n.turn_heading
    assert n.phase == "TURN_AROUND" and abs(target) == pytest.approx(180)
    assert n.commands[-1][0] == 0 and n.commands[-1][2] != 0
    n.p["goal_a_deg"] = 45  # Global goal changes cannot replace this turn target.
    n.pose[2] = -90
    clock.now += 0.05
    motion_update(n, clock, points, angles)
    assert n.turn_heading == target and n.commands[-1][0] == 0
    assert np.asarray(n.trail) == pytest.approx(original)
    assert len(n.red_segments) == 1


def test_turn_completion_trims_failed_tail_and_resumes_front_scoring(turn_route):
    n, clock = turn_route
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    n.pose[2] = n.turn_heading
    motion_update(n, clock, points, angles)
    assert n.phase == "SELECT" and n.retreat_index is None
    assert len(n.trail) == 4 and n.trail_reached == [] and len(n.red_segments) == 1
    assert n.commands[-1] == (0, 0, 0)
    motion_update(n, clock, [[2, 0]], [0])
    assert n.phase == "DRIVE" and n.commands[-1][0] > 0
    n.pose[0] -= 0.06
    clock.now += 0.05
    motion_update(n, clock, [[2, 0]], [0])
    assert n.trail[-1] == pytest.approx(n.pose[:2])
    assert len(n.trail) == 5 and len(n.red_segments) == 1


@pytest.mark.parametrize("fault", ["new_obstacle", "stale", "timeout"])
def test_turnaround_faults_hold_stop_without_automatic_resume(turn_route, fault):
    n, clock = turn_route
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    if fault == "new_obstacle":
        motion_update(n, clock, [[n.p["turn_clearance_m"] - 0.02, 0]], [0])
    elif fault == "stale":
        clock.now += n.p["sensor_timeout_s"] + 0.01
        n.timer_callback()
    else:
        clock.now += n.p["turn_timeout_s"] + 0.01
        motion_update(n, clock, points, angles)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    motion_update(n, clock, points, angles)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)


def test_second_recovery_retains_old_red_segments(turn_route):
    n, clock = turn_route
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    n.pose[2] = n.turn_heading
    motion_update(n, clock, points, angles)
    old = np.array(n.red_segments).copy()
    n.trail.append(np.array([-0.1, 0]))
    n.pose[:2] = [-0.1, 0]
    n.start_retreat(points, np.radians(angles))
    n.pose[:2] = n.trail[n.retreat_index]
    motion_update(n, clock, points, angles)
    assert len(n.red_segments) > len(old)
    assert np.array(n.red_segments[:len(old)]) == pytest.approx(old)


def turnaround_room_scan(n):
    """Wide room leading to a narrow dead-end corridor; exact ray/segment intersections."""
    vertices = np.array([[-2, -0.6], [0.35, -0.6], [0.35, -0.18], [1, -0.18],
                         [1, 0.18], [0.35, 0.18], [0.35, 0.6], [-2, 0.6]])
    raw = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    heading = math.radians(n.pose[2])
    angles = raw + math.radians(n.p["lidar_yaw_deg"]) + heading
    rays = np.column_stack((np.cos(angles), np.sin(angles)))
    sensor = n.pose[:2] + n.p["lidar_x_m"] * np.array([math.cos(heading), math.sin(heading)])
    ranges = np.full(360, np.inf)
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = cross(rays, edge)
            distance = cross(a - sensor, edge) / denominator
            fraction = cross(a - sensor, rays) / denominator
        ranges = np.minimum(ranges, np.where(
            (distance > 0) & (fraction >= 0) & (fraction <= 1), distance, np.inf))
    scan = LaserScan()
    scan.angle_min, scan.angle_increment = -math.pi, math.pi / 180
    scan.range_min, scan.range_max = 0.05, 10.0
    scan.ranges = ranges.tolist()
    return scan, vertices


def test_closed_loop_reverse_turn_and_resume_with_persistent_red_and_new_green(motion_node):
    n, clock = motion_node
    n.single_step = n.backtrack_one = False
    n.recover = n.retreat_curves = True
    n.retreat_limit = 0
    n.p.update(goal_a_deg=0, check_distance_m=0.4, curve_clearance_m=0.16,
               turn_clearance_m=0.21, turn_timeout_s=20.0)
    phases, turning_trail, resumed_at = set(), None, None
    for _ in range(900):
        clock.now += 0.05
        n.last_pose_time = clock.now
        scan, vertices = turnaround_room_scan(n)
        n.sub_scan_callback(scan)
        n.timer_callback()
        phases.add(n.phase)
        if n.phase == "TURN_AROUND":
            if turning_trail is None:
                turning_trail = np.array(n.trail)
            assert np.asarray(n.trail) == pytest.approx(turning_trail)
            assert n.commands[-1][0] == 0
        if turning_trail is not None and n.phase in ("DRIVE", "CURVE"):
            if resumed_at is None:
                resumed_at = n.pose[:2].copy()
            if np.linalg.norm(n.pose[:2] - resumed_at) >= 0.15:
                break
        assert n.phase not in ("BLOCKED", "RETREAT_DONE")
        speed, _, turn = n.commands[-1]
        heading = math.radians(n.pose[2])
        n.pose[:2] += speed * 0.05 * np.array([math.cos(heading), math.sin(heading)])
        n.pose[2] += math.degrees(turn * 0.05)
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = b - a
            closest = a + np.clip(np.dot(n.pose[:2] - a, edge) / np.dot(edge, edge), 0, 1) * edge
            required = n.p["turn_clearance_m"] if n.phase == "TURN_AROUND" else n.robot_radius
            assert np.linalg.norm(n.pose[:2] - closest) > required
    assert {"WAIT_BLOCKED", "RETREAT", "TURN_AROUND", "DRIVE"} <= phases
    assert resumed_at is not None and np.linalg.norm(n.pose[:2] - resumed_at) >= 0.15
    assert n.pose[0] < resumed_at[0]  # Moving out, away from the failed positive-x end.
    assert len(n.red_segments) > 1 and n.trail_reached == []
    assert n.trail[-1][0] < resumed_at[0]


def test_new_recovery_flag_cannot_be_combined_with_another_motion_mode(monkeypatch):
    from rb2301_fp.obstacle_course import main
    monkeypatch.setattr("rb2301_fp.obstacle_course.rclpy.init",
                        lambda **kwargs: pytest.fail("Conflicting flags must fail before ROS starts"))
    with pytest.raises(SystemExit) as error:
        main(["--recover", "--move"])
    assert error.value.code == 2


def test_viewer_keeps_red_history_fixed_after_turn_and_new_green_motion(turn_route, viewer_module):
    import matplotlib.pyplot as plt
    n, clock = turn_route
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    n.last_scan = box_scan(n.pose, right=2.0)
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        view.update(viewer_packet(n))
        view.fig.canvas.draw()
        assert "TURN_AROUND" in view.path_scores.get_text()
        assert view.turn_clearance.get_visible()
        assert view.turn_clearance.get_radius() == pytest.approx(n.p["turn_clearance_m"])
        assert not view.carrot.get_visible()
        assert "Turn target course" in view.aim_text.get_text()
        n.pose[2] = n.turn_heading
        motion_update(n, clock, points, angles)
        n.last_scan = box_scan(n.pose, right=2.0)
        view.update(viewer_packet(n))
        red_xy = np.column_stack(view.maze_reached.get_data())
        assert red_xy == pytest.approx(np.array([[0, 0.15], [0, 0.2]]))
        n.motion_enabled = False
        n.pose[:] = [-0.2, 0.1, 90]
        n.sub_scan_callback(box_scan(n.pose, right=2.0))
        n.timer_callback()
        view.update(viewer_packet(n))
        assert np.column_stack(view.maze_reached.get_data()) == pytest.approx(red_xy)
        assert not view.turn_clearance.get_visible()
        view.update(None)
        assert len(view.lidar_reached.get_xdata()) == 0
    finally:
        plt.close(view.fig)


def test_visited_overlap_distinguishes_travelled_right_from_new_left(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    distances = np.linspace(0.2, 0.4, 5)
    angle = -math.pi / 4
    n.visited_positions = [d * np.array([math.cos(angle), math.sin(angle)]) for d in distances]
    paths = n.score_directions(np.array([[2, 0], [0, -2], [0, 2]]), np.radians([-45, 45]))
    assert paths["visited_fractions"] == pytest.approx([1, 0])
    assert paths["visited_ends"] == pytest.approx([0.4, 0.4])


def test_visited_overlap_ignores_shared_start_and_counts_duplicate_dots_once(preview_node):
    n = preview_node
    obstacle = np.array([[2.0, 0]])
    origin_history = np.array([[0, 0], [0.05, 0], [0.08, 0]])
    fraction, _, _ = n.visited_overlap(obstacle, 0.0, origin_history)
    assert fraction == 0
    history = np.array([[0.3, 0]])
    once = n.visited_overlap(obstacle, 0.0, history)
    repeated = n.visited_overlap(obstacle, 0.0, np.repeat(history, 100, axis=0))
    assert once == repeated
    assert once[0] == 1


def test_visited_overlap_uses_adjustable_distance_independent_of_movement(preview_node):
    n = preview_node
    history = np.array([[0.7, 0], [0.8, 0]])
    points = np.array([[2.0, 0]])
    assert n.visited_overlap(points, 0, history)[0] == 0
    n.p["visited_check_distance_m"] = 0.8
    fraction, end, _ = n.visited_overlap(points, 0, history)
    assert fraction > 0 and end == 0.8
    assert n.p["check_distance_m"] == 0.3  # Fixture's movement setting did not change.


@pytest.mark.parametrize("angle", [0, math.pi / 4, -math.pi / 2])
def test_visited_overlap_stops_at_obstruction_and_ignores_history_beyond_it(preview_node, angle):
    n = preview_node
    direction = np.array([math.cos(angle), math.sin(angle)])
    wall = np.array([0.35 * direction])
    # This dot's 10 cm circle reaches into the window, but the dot is beyond
    # the obstruction-limited centre path and must not count through the wall.
    history = np.array([0.32 * direction, 0.4 * direction])
    fraction, end, mask = n.visited_overlap(wall, angle, history)
    assert end == pytest.approx(0.35 - n.robot_radius)
    assert fraction == 0 and not any(mask)


@pytest.mark.parametrize("points", [[[0.28, 0]], [[0.05, 0]], []])
def test_visited_overlap_reports_no_window_as_none_rather_than_unvisited(preview_node, points):
    n = preview_node
    fraction, end, mask = n.visited_overlap(np.asarray(points).reshape(-1, 2), 0, np.empty((0, 2)))
    assert fraction is None and end <= 0.2 and mask == []


@pytest.mark.parametrize("heading", [-90, 0, 45, 179])
def test_visited_overlap_uses_fixed_course_history_after_translation_and_rotation(preview_node, heading):
    n = preview_node
    n.pose = np.array([2.0, 3.0, float(heading)])
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    angle = math.radians(heading)
    direction = np.array([math.cos(angle), math.sin(angle)])
    n.visited_positions = [n.pose[:2] + d * direction for d in (0.2, 0.3, 0.4)]
    paths = n.score_directions(np.array([[2.0, 0]]), np.array([0.0]))
    assert paths["visited_fractions"] == [1.0]


def test_visited_history_survives_turnaround_trimming_and_records_reverse(turn_route):
    n, clock = turn_route
    n.visited_positions = [p.copy() for p in n.trail]
    old_history = np.array(n.visited_positions)
    n.pose[:2] = [0.12, 0]
    n.phase = "RETREAT"
    n.record_trail()
    assert n.visited_positions[-1] == pytest.approx([0.12, 0])
    assert np.array(n.visited_positions[:len(old_history)]) == pytest.approx(old_history)
    n.try_turnaround(np.array([[2.0, 0]]))
    n.pose[2] = n.turn_heading
    n.run_turnaround(np.array([[2.0, 0]]))
    assert len(n.trail) < len(old_history)
    assert np.array(n.visited_positions[:len(old_history)]) == pytest.approx(old_history)
    count = len(n.visited_positions)
    n.phase = "TURN"
    n.pose[:2] += [0.1, 0]
    n.record_trail()
    assert len(n.visited_positions) == count


def test_zero_visited_weight_cannot_change_candidate_scores_or_eligibility(preview_node):
    n = preview_node
    initialise(n, heading=0)
    points, angles = n.scan_geometry()
    before = n.score_directions(points, angles)
    n.visited_positions = [np.array([x, y]) for x in np.arange(-0.5, 0.5, 0.05)
                           for y in np.arange(-0.5, 0.5, 0.05)]
    n.p["visited_check_distance_m"] = 0.8
    after = n.score_directions(points, angles)
    for field in ("scores", "eligible", "selected_index", "clear", "soft_clear", "red_blocked"):
        assert before[field] == after[field]


def test_viewer_inspects_overlap_samples_without_sending_commands(preview_node, viewer_module):
    import matplotlib.pyplot as plt
    n = preview_node
    initialise(n, heading=0)
    # box_scan's exactly parallel horizontal ray produces -inf; supply the
    # known front-wall return so this click tests a supported 0-degree candidate.
    n.last_scan.ranges[0] = 2.0 - n.p["lidar_x_m"]
    n.p["visited_penalty_weight"] = 1.2
    n.visited_positions = [np.array([d, 0]) for d in (0.2, 0.3, 0.4)]
    points, angles = n.scan_geometry()
    n.publish_preview(n.score_directions(points, angles), "PREVIEW: motors stopped")
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    try:
        commands = list(n.commands)
        view.update(viewer_packet(n))
        view.on_click(SimpleNamespace(inaxes=view.ax, button=1, xdata=0, ydata=0.4))
        assert "Inspect +0°: 100% visited" in view.visited_text.get_text()
        assert "Visited penalty: -1.200" in view.visited_text.get_text()
        assert len(view.visited_old.get_xdata()) > 0
        assert len(view.visited_new.get_xdata()) == 0
        assert view.visited_end.get_xdata() == pytest.approx([0.4])
        assert n.commands == commands
        view.update(None)
        assert view.visited_text.get_text() == ""
        assert len(view.visited_old.get_xdata()) == len(view.maze_visited.get_xdata()) == 0
    finally:
        plt.close(view.fig)


def test_viewer_shows_obstruction_clipped_overlap_and_no_window(preview_node, viewer_module):
    import matplotlib.pyplot as plt
    n = preview_node
    initialise(n, heading=0)
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    view.inspected_visited_angle = 0
    try:
        for wall, expected in ((0.35, "0% visited"), (0.28, "N/A: blocked before window")):
            paths = n.score_directions(np.array([[wall, 0]]), np.array([0.0]))
            n.publish_preview(paths, "PREVIEW: motors stopped")
            view.update(viewer_packet(n))
            assert expected in view.visited_text.get_text()
            assert "Shortened at obstruction" in view.visited_text.get_text()
            assert view.visited_end.get_xdata() == pytest.approx([wall - n.robot_radius])
            assert np.all(np.asarray(view.visited_new.get_xdata()) <= wall - n.robot_radius + 1e-9)
    finally:
        plt.close(view.fig)


@pytest.mark.parametrize("phase,yaw", [("TURN", 50), ("CURVE", 10), ("DRIVE", 0), ("DONE", 0)])
def test_overlap_stays_live_without_reselecting_during_movement_or_step_stop(motion_node, phase, yaw):
    n, clock = motion_node
    n.p["visited_penalty_weight"] = 1.2
    motion_update(n, clock)
    held_heading = n.locked_heading
    n.phase, n.pose[2] = phase, yaw
    direction = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    n.visited_positions = [n.pose[:2] + d * direction for d in (0.2, 0.3, 0.4)]
    n.score_directions = lambda *args: pytest.fail("Display updates must not rescore/reselect a held heading")
    motion_update(n, clock)
    frame = n.frames[-1]
    assert n.phase == phase and n.locked_heading == held_heading
    assert frame["angles"] == [] and frame["scores"] == []  # No new navigation ranking.
    assert frame["visited_angles"] == [0.0] and frame["visited_fractions"] == [1.0]
    n.visited_positions = []
    motion_update(n, clock)
    assert n.frames[-1]["visited_fractions"] == [0.0]  # Fresh data, not the earlier percentage.
    assert n.locked_heading == held_heading
    if phase == "DONE":
        assert n.commands[-1] == (0, 0, 0)


def test_overlap_remains_available_during_reverse_and_turnaround_without_forward_selection(turn_route):
    n, clock = turn_route
    n.score_directions = lambda *args: pytest.fail("Recovery display must not select a forward departure")
    n.pose[0] = 0.20  # A few centimetres behind the target: start by reversing.
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    assert n.phase == "RETREAT" and n.frames[-1]["visited_angles"]
    n.pose[0] = 0.15
    motion_update(n, clock, points, angles)
    assert n.phase == "TURN_AROUND" and n.frames[-1]["visited_angles"]
    saved_turn = n.turn_heading
    n.pose[2] = 90
    motion_update(n, clock, points, angles)
    assert n.phase == "TURN_AROUND" and n.turn_heading == saved_turn
    assert n.frames[-1]["visited_angles"] and n.commands[-1][0] == 0


def test_stale_input_clears_live_overlap_overlay(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    motion_update(n, clock)
    assert n.frames[-1]["visited_fractions"]
    clock.now += n.p["sensor_timeout_s"] + 0.01
    n.timer_callback()
    assert not n.frames[-1].get("visited_fractions")
    assert n.commands[-1] == (0, 0, 0)


@pytest.mark.parametrize("goal_weight,weight,chosen", [(0.5, 0.0, -60), (0.5, 1.2, 120),
                                                     (1.0, 1.2, -60), (1.0, 2.2, 120)])
def test_visited_penalty_can_choose_unvisited_path_opposite_goal(preview_node, goal_weight, weight, chosen):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p.update(goal_a_deg=-60, virtual_goal_weight=goal_weight, visited_penalty_weight=weight)
    angles = np.radians([-60, 120])  # Opposite directions, both inside the 240-degree search.
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    points = (1.0 + n.robot_radius) * directions  # Equal depth and clearance.
    n.visited_positions = [d * directions[0] for d in np.linspace(0.2, 0.4, 5)]
    paths = n.score_directions(points, angles)
    assert paths["visited_fractions"] == pytest.approx([1.0, 0.0])
    assert paths["visited_penalties"] == pytest.approx([weight, 0.0])
    assert paths["angles"][paths["selected_index"]] == chosen


def test_visited_penalty_scales_with_partial_overlap(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.visited_positions = [np.array([0.2, 0])]
    points, angles = np.array([[2.0, 0]]), np.array([0.0])
    before = n.score_directions(points, angles)
    n.p["visited_penalty_weight"] = 1.2
    after = n.score_directions(points, angles)
    fraction = after["visited_fractions"][0]
    assert 0 < fraction < 1
    assert before["scores"][0] - after["scores"][0] == pytest.approx(1.2 * fraction)


def test_equally_visited_candidates_keep_relative_ranking(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.visited_positions = [np.array([x, y]) for x in np.arange(-0.5, 0.51, 0.05)
                           for y in np.arange(-0.5, 0.51, 0.05)]
    points, angles = np.array([[2, 0], [1.7, 1.7]]), np.radians([0, 45])
    before = n.score_directions(points, angles)
    n.p["visited_penalty_weight"] = 1.2
    after = n.score_directions(points, angles)
    assert after["visited_fractions"] == [1.0, 1.0]
    assert after["scores"] == pytest.approx(np.array(before["scores"]) - 1.2)
    assert after["selected_index"] == before["selected_index"]


def test_only_visited_route_remains_usable_even_with_negative_score(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p["visited_penalty_weight"] = 10.0
    n.visited_positions = [np.array([d, 0]) for d in (0.2, 0.3, 0.4)]
    paths = n.score_directions(np.array([[2.0, 0]]), np.array([0.0]))
    assert paths["scores"][0] < 0 and paths["eligible"] == [0] and paths["selected_index"] == 0


def test_no_overlap_window_adds_no_penalty_and_does_not_reject_clear_route(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p.update(visited_check_start_m=0.4, visited_check_distance_m=0.8, visited_penalty_weight=1.2)
    paths = n.score_directions(np.array([[0.45, 0]]), np.array([0.0]))
    assert paths["visited_fractions"] == [None] and paths["visited_penalties"] == [0.0]
    assert paths["eligible"] == [0]


def test_visited_penalty_cannot_override_physical_or_red_route_rejections(preview_node):
    n = preview_node
    n.pose = np.zeros(3)
    n.last_scan = SimpleNamespace(angle_increment=math.radians(1))
    n.p["visited_penalty_weight"] = 1000
    n.red_segments = [(np.zeros(2), np.array([0.2, math.sqrt(0.12)]))]
    paths = n.score_directions(np.array([[0.2, 0], [0.5, -math.sqrt(0.75)]]), np.radians([-60, 0, 60]))
    assert paths["eligible"] == [0] and paths["angles"][paths["selected_index"]] == -60
    assert not paths["clear"][1] and paths["red_blocked"][2]


def test_selection_log_records_visited_percentage_and_subtracted_penalty(motion_node):
    n, clock = motion_node
    n.p["visited_penalty_weight"] = 1.2
    n.visited_positions = [np.array([d, 0]) for d in (0.2, 0.3, 0.4)]
    motion_update(n, clock)
    line = next(l for l in n.logs if l.startswith("CANDIDATE selected:"))
    values = dict(field.split("=", 1) for field in line.split() if "=" in field)
    assert values["visited"] == "100.0%" and values["visited_penalty"] == "1.200"
    base = sum(float(values[k]) for k in ("goal", "depth_score", "soft", "side"))
    assert float(values["total"]) == pytest.approx(base - 1.2, abs=0.003)


def test_overlap_window_start_is_independent_of_end_and_movement_distance(preview_node):
    n = preview_node
    n.p.update(visited_check_start_m=0.2, visited_check_distance_m=0.8)
    points = np.array([[2.0, 0]])
    history = np.array([[0.2, 0]])
    fraction, end, mask = n.visited_overlap(points, 0.0, history)
    assert end == 0.8 and 0 < fraction < 0.5  # Include the earlier 20–30 cm section.
    assert len(mask) == math.ceil((0.8 - 0.2) / 0.01) + 1
    n.p["visited_check_start_m"] = 0.4
    assert n.visited_overlap(points, 0.0, history)[0] == 0.0
    assert n.p["visited_check_distance_m"] == 0.8 and n.p["check_distance_m"] == 0.3


def test_longer_overlap_window_uses_only_usable_part_before_obstruction(preview_node):
    n = preview_node
    n.p.update(visited_check_start_m=0.2, visited_check_distance_m=0.8)
    points = np.array([[0.61, 0]])  # 11 cm hard radius leaves 0.50 m centre travel.
    history = np.array([[d, 0] for d in (0.2, 0.3, 0.4, 0.5)])
    fraction, end, _ = n.visited_overlap(points, 0.0, history)
    assert end == pytest.approx(0.5) and fraction == 1.0
    # Old positions beyond the usable end cannot count through the obstruction.
    assert n.visited_overlap(points, 0.0, np.array([[0.6, 0], [0.7, 0]]))[0] == 0.0


def test_viewer_renders_independent_overlap_start_and_end(preview_node, viewer_module):
    import matplotlib.pyplot as plt
    n = preview_node
    initialise(n, heading=0)
    n.p.update(visited_check_start_m=0.2, visited_check_distance_m=0.8)
    n.last_scan.ranges[0] = 2.0 - n.p["lidar_x_m"]
    points, angles = n.scan_geometry()
    n.publish_preview(n.score_directions(points, angles), "PREVIEW: motors stopped")
    view = viewer_module.MinimalMap(plt, PARAMS["lidar_x_m"], PARAMS["lidar_y_m"], PARAMS["lidar_yaw_deg"])
    view.inspected_visited_angle = 0
    try:
        view.update(viewer_packet(n))
        assert "Check 0.20–0.80 m" in view.visited_text.get_text()
        samples = np.asarray(view.visited_new.get_xdata())
        assert samples.min() == pytest.approx(0.2) and samples.max() == pytest.approx(0.8)
        assert view.visited_end.get_xdata() == pytest.approx([0.8])
    finally:
        plt.close(view.fig)


def test_retreat_restores_segment_a_goal_when_crossing_back_into_first_region(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    n.p.update(goal_a_deg=-45, goal_b_deg=0)
    n.switch_x = 0.05
    n.recover = n.retreat_curves = True
    n.phase, n.segment = "RETREAT", "B"
    n.trail = [np.array([0.0, 0.0]), np.array([0.0, -0.1])]
    n.retreat_index, n.retreat_started = 0, clock.now
    n.pose[:] = [0.0, -0.08, -90.0]  # Rear points leftward, toward the earlier region.
    n.red_last_position = n.pose[:2].copy()
    n.score_directions = lambda *args: pytest.fail("Changing segment must not select a departure during retreat")
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    assert n.segment == "B" and n.virtual_heading_deg() == 0
    n.pose[1] = -0.04
    motion_update(n, clock, points, angles)
    assert n.segment == "A" and n.virtual_heading_deg() == -45
    assert n.phase == "RETREAT" and n.retreat_index == 0 and n.commands[-1][0] < 0
    assert n.frames[-1]["segment"] == "A" and n.frames[-1]["virtual_heading_deg"] == -45


def test_return_to_segment_a_does_not_interrupt_committed_heading(motion_node):
    n, clock = motion_node
    motion_update(n, clock)
    held = n.locked_heading
    n.p.update(goal_a_deg=-45, goal_b_deg=0)
    n.switch_x, n.segment = 0.05, "B"
    n.pose[1] = -0.04  # Return across the boundary during the committed advance.
    n.score_directions = lambda *args: pytest.fail("Keep the held heading until the advance completes")
    motion_update(n, clock)
    assert n.segment == "A" and n.virtual_heading_deg() == -45
    assert n.locked_heading == held and n.phase == "DRIVE" and n.commands[-1][0] > 0


def test_disabled_turn_recheck_keeps_initial_space_check_and_continues_retreat(turn_route):
    n, clock = turn_route
    n.p["recheck_turn_space_during_turn"] = False
    radius = n.p["turn_clearance_m"]
    motion_update(n, clock, [[radius - 0.01, 0]], [-180, 0])
    assert n.phase == "RETREAT" and n.retreat_index == 2
    assert n.turn_heading is None and n.commands[-1] == (0, 0, 0)
    assert any("TURN SPACE:" in line and "blocked" in line for line in n.logs)


def test_disabled_turn_recheck_completes_held_turn_despite_later_clearance_change(turn_route):
    n, clock = turn_route
    n.p["recheck_turn_space_during_turn"] = False
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    assert n.phase == "TURN_AROUND"
    held = n.turn_heading
    obstructed = np.array([[n.p["turn_clearance_m"] - 0.01, 0]])
    assert not n.path_is_clear(obstructed, 0.0, 0.0, n.p["turn_clearance_m"])
    motion_update(n, clock, obstructed, [0])
    assert n.phase == "TURN_AROUND" and n.turn_heading == held
    assert n.commands[-1][0] == 0 and abs(n.commands[-1][2]) > 0
    assert n.frames[-1]["turn_space_recheck"] is False
    assert "space_recheck=off" in n.frames[-1]["status"]
    n.pose[2] = held
    motion_update(n, clock, obstructed, [0])
    assert n.phase == "SELECT" and n.commands[-1] == (0, 0, 0)
    assert any("RECOVERY COMPLETE" in line for line in n.logs)


@pytest.mark.parametrize("fault", ["stale", "timeout", "missing_pose", "missing_scan"])
def test_disabled_turn_space_recheck_preserves_other_stop_conditions(turn_route, fault):
    n, clock = turn_route
    n.p["recheck_turn_space_during_turn"] = False
    points, angles = open_turn_scene()
    motion_update(n, clock, points, angles)
    assert n.phase == "TURN_AROUND"
    if fault == "stale":
        clock.now += n.p["sensor_timeout_s"] + 0.01
        n.timer_callback()
    elif fault == "timeout":
        clock.now += n.p["turn_timeout_s"] + 0.01
        motion_update(n, clock, points, angles)
    else:
        if fault == "missing_pose":
            n.pose = None
        else:
            n.last_scan = None
        n.timer_callback()
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
    motion_update(n, clock, points, angles)
    assert n.phase == "BLOCKED" and n.commands[-1] == (0, 0, 0)
