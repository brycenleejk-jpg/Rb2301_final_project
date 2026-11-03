"""Selection and command checks without starting ROS nodes or moving a robot."""
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rb2301_fp.obstacle_course import ObstacleCourseNode, PARAMS


@pytest.fixture
def node():
    # Use real controller methods, replacing only ROS communication and plotting.
    n = ObstacleCourseNode.__new__(ObstacleCourseNode)
    n.p = PARAMS.copy()
    n.p.update(goal_x_m=2.0, goal_y_m=0.0)
    n.robot_radius = n.p["robot_radius_m"]
    n.max_translate_velocity = 0.3
    n.pose = np.array([0.0, 0.0, 0.0])
    n.raw_pose = n.frame = n.course = None
    n.scan_msg = object()
    n.last_scan_time = n.last_pose_time = time.monotonic()
    n.calibrate = n.done = n.finishing = False
    n.approach_heading_deg = n.finish_heading_deg = None
    n.selected_course_heading_deg = None
    n.commands, n.logs, n.snapshots = [], [], []
    n.publisher_ = SimpleNamespace(publish=lambda msg: n.commands.append(
        (msg.linear.x, msg.linear.y, msg.angular.z)))
    n.get_logger = lambda: SimpleNamespace(info=lambda message, **kw: n.logs.append(message))
    n.plot_goal = lambda: None
    n.plot_direction = lambda *args, **kwargs: None
    n.plot_paths = lambda *args, **kwargs: n.snapshots.append((args, kwargs))
    return n


def geometry(degrees, ranges):
    angles = np.radians(np.asarray(degrees, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles))), angles


def frame(node, points, angles, heading=0.0):
    node.pose[2] = heading
    node.last_scan_time = node.last_pose_time = time.monotonic()
    node.scan_geometry = lambda: (points, angles)
    node.timer_callback()


def test_initial_choice_prefers_depth_within_score_band(node):
    points, angles = geometry([-10, 30], [1.11, 1.30])
    selected = node.select_direction(points, angles, 0.0, 0.3)
    assert math.degrees(selected) == pytest.approx(30.0)
    assert not node._selection_held
    assert "depth=1.19m" in node._selection_summary
    assert "choice=depth" in node._selection_summary
    # The alternative must not accidentally repeat the selected direction.
    assert "alternative=-10.0deg" in node._selection_summary
    assert "gap=0.024" in node._selection_summary


def test_depth_cannot_override_a_large_score_difference(node):
    points, angles = geometry([-10, 30], [1.11, 1.20])
    selected = node.select_direction(points, angles, 0.0, 0.3)
    assert math.degrees(selected) == pytest.approx(-10.0)


def test_near_tie_noise_does_not_reverse_turning(node):
    headings = []
    for i in range(100):
        ranges = [2.001, 1.999] if i % 2 == 0 else [1.999, 2.001]
        points, angles = geometry([-30, 30], ranges)
        frame(node, points, angles)
        headings.append(node.selected_course_heading_deg)
        assert node.commands[-1][2] < 0.0
        if i:
            assert node._selection_held
    assert headings == pytest.approx([-30.0] * 100)


@pytest.mark.parametrize("right_range,held", [(2.06, True), (2.06001, False), (2.30, False)])
def test_score_improvement_and_exact_tolerance_boundary(node, right_range, held):
    points, angles = geometry([-30, 30], [2.0, right_range])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(-30))
    assert node._selection_held is held
    assert math.degrees(selected) == pytest.approx(-30.0 if held else 30.0)


def test_blocked_previous_route_is_not_retained(node):
    points, angles = geometry([-30, 30, -30], [2.0, 2.0, 0.25])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(-30))
    assert math.degrees(selected) == pytest.approx(30.0)
    assert not node._selection_held
    assert node.path_is_clear(points, selected, 0.3, radius=node._selection_radius)


def test_course_heading_is_preserved_during_rotation(node):
    node.selected_course_heading_deg = -30.0
    for heading in (0.0, -10.0, -20.0, -30.0):
        points, angles = geometry([-30 - heading, 30 - heading], [2.0, 2.0])
        frame(node, points, angles, heading)
        assert node._selection_held
        assert node.selected_course_heading_deg == pytest.approx(-30.0)
        assert node.approach_heading_deg == pytest.approx(-30.0)


def test_course_heading_wraps_across_180_degrees(node):
    node.selected_course_heading_deg = -179.0
    node.p["goal_x_m"] = 2 * math.cos(math.radians(-179))
    node.p["goal_y_m"] = 2 * math.sin(math.radians(-179))
    points, angles = geometry([2, -28], [2.0, 2.0])
    frame(node, points, angles, heading=179.0)
    assert node._selection_held
    assert node.selected_course_heading_deg == pytest.approx(-179.0)
    assert "selected=+2.0deg" in node._selection_summary


def test_candidate_rounding_does_not_shift_the_remembered_heading(node):
    node.selected_course_heading_deg = -30.0
    for angle in (-29.0, -31.0, -28.5, -30.5):
        points, angles = geometry([angle, 30], [2.0, 2.0])
        frame(node, points, angles)
        assert node._selection_held
        assert node.selected_course_heading_deg == pytest.approx(-30.0)


def test_distant_candidate_is_not_mistaken_for_previous_direction(node):
    points, angles = geometry([-30, 30], [2.0, 2.0])
    node.select_direction(points, angles, 0.0, 0.3,
                          previous_direction=math.radians(40))
    assert not node._selection_held


def test_memory_survives_the_hard_clearance_retry(node):
    node.selected_course_heading_deg = -30.0
    points, angles = geometry([-30, 30, 90], [2.0, 2.0, 0.1244])
    frame(node, points, angles)
    assert node._selection_radius == pytest.approx(0.11)
    assert node._selection_held
    assert node.selected_course_heading_deg == pytest.approx(-30.0)
    assert 0.0 < node.commands[-1][0] <= 0.10
    assert "clearance=reduced" in node._selection_summary
    assert len(node.snapshots) == 1  # Only the final retry is plotted.


def test_preferred_clearance_still_takes_priority_over_memory(node):
    points = np.array([[0.25, 0.12], [2.0, 0.0],
                       [2 * math.cos(-math.pi / 6), 2 * math.sin(-math.pi / 6)]])
    angles = np.arctan2(points[:, 1], points[:, 0])
    selected = node.select_direction(points, angles, 0.0, 0.3, previous_direction=0.0)
    assert node._selection_radius == pytest.approx(0.13)
    assert selected < 0.0
    assert not node._selection_held


@pytest.mark.parametrize("kind", ["footprint", "no_forward", "empty", "stale_scan"])
def test_stop_clears_commitment_and_sends_zero_velocity(node, kind):
    points, angles = geometry([-30, 30], [2.001, 1.999])
    frame(node, points, angles)
    assert node.selected_course_heading_deg is not None
    if kind == "footprint":
        points, angles = geometry([0], [0.10])
    elif kind == "no_forward":
        points, angles = geometry([180], [2.0])
    elif kind == "empty":
        points, angles = np.empty((0, 2)), np.array([])
    node.scan_geometry = lambda: (points, angles)
    if kind == "stale_scan":
        node.last_scan_time -= 1.0
    node.timer_callback()
    assert node.commands[-1] == (0.0, 0.0, 0.0)
    assert node.selected_course_heading_deg is None


@pytest.mark.parametrize("tolerance", [0.0, -0.1])
def test_nonpositive_tolerance_only_accepts_top_scores(node, tolerance):
    node.p["similar_score_tolerance"] = tolerance
    points, angles = geometry([-30, 30], [2.0, 2.001])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(-30))
    assert math.degrees(selected) == pytest.approx(30.0)
    assert not node._selection_held
