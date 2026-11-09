"""Selection and command checks without starting ROS nodes or moving a robot."""
import math
import time

import numpy as np
import pytest



def geometry(degrees, ranges):
    angles = np.radians(np.asarray(degrees, dtype=float))
    ranges = np.asarray(ranges, dtype=float)
    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles))), angles


def frame(node, points, angles, heading=0.0):
    node.heading_deg = heading
    node.last_scan_time = node.last_heading_time = time.monotonic()
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
        headings.append(node.selected_heading_deg)
        assert node.commands[-1][2] < 0.0
        if i:
            assert node._selection_held
    assert headings == pytest.approx([-30.0] * 100)


@pytest.mark.parametrize("right_range,held", [(2.06, True), (2.06001, False), (2.30, False)])
@pytest.mark.parametrize("aligning", [False, True])
def test_score_improvement_and_exact_tolerance_boundary(node, right_range, held, aligning):
    points, angles = geometry([-30, 30], [2.0, right_range])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(-30),
                                     hold_previous=aligning)
    assert node._selection_held is held
    assert math.degrees(selected) == pytest.approx(-30.0 if held else 30.0)


def test_blocked_previous_route_is_not_retained(node):
    points, angles = geometry([-30, 30, -30], [2.0, 2.0, 0.25])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(-30))
    assert math.degrees(selected) == pytest.approx(30.0)
    assert not node._selection_held
    assert node.path_is_clear(points, selected, 0.3, radius=node._selection_radius)


def test_local_direction_is_preserved_during_rotation(node):
    node.selected_heading_deg = -30.0
    for heading in (0.0, -10.0, -20.0, -30.0):
        points, angles = geometry([-30 - heading, 30 - heading], [2.0, 2.0])
        frame(node, points, angles, heading)
        assert node._selection_held
        assert node.selected_heading_deg == pytest.approx(-30.0)


def test_direction_memory_wraps_across_180_degrees(node):
    node.p['virtual_goal_initial_deg'] = -179.0
    node.selected_heading_deg = -179.0
    points, angles = geometry([2, -28], [2.0, 2.0])
    frame(node, points, angles, heading=179.0)
    assert node._selection_held
    assert node.selected_heading_deg == pytest.approx(-179.0)
    assert "selected=+2.0deg" in node._selection_summary


def test_candidate_rounding_does_not_shift_the_remembered_heading(node):
    node.selected_heading_deg = -30.0
    for angle in (-29.0, -31.0, -28.5, -30.5):
        points, angles = geometry([angle, 30], [2.0, 2.0])
        frame(node, points, angles)
        assert node._selection_held
        assert node.selected_heading_deg == pytest.approx(-30.0)


def test_distant_candidate_is_not_mistaken_for_previous_direction(node):
    points, angles = geometry([-30, 30], [2.0, 2.0])
    node.select_direction(points, angles, 0.0, 0.3,
                          previous_direction=math.radians(40))
    assert not node._selection_held


def test_memory_survives_the_hard_clearance_retry(node):
    node.selected_heading_deg = -30.0
    points, angles = geometry([-30, 30, 90], [2.0, 2.0, 0.1244])
    frame(node, points, angles)
    assert node._selection_radius == pytest.approx(0.11)
    assert node._selection_held
    assert node.selected_heading_deg == pytest.approx(-30.0)
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
    assert node.selected_heading_deg is not None
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
    assert node.selected_heading_deg is None


@pytest.mark.parametrize("tolerance", [0.0, -0.1])
def test_nonpositive_tolerance_only_accepts_top_scores(node, tolerance):
    node.p["similar_score_tolerance"] = tolerance
    points, angles = geometry([-30, 30], [2.0, 2.001])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(-30))
    assert math.degrees(selected) == pytest.approx(30.0)
    assert not node._selection_held


@pytest.mark.parametrize('direction', [-135, 135])
def test_wide_sector_can_select_and_hold_an_escape_beyond_90(node, direction):
    node.p['view_half_angle_deg'] = 135.0
    points, angles = geometry([0, direction], [0.25, 2.0])
    selected = node.select_direction(points, angles, 0.0, 0.3,
                                     previous_direction=math.radians(direction), hold_previous=True)
    assert math.degrees(selected) == pytest.approx(direction)
    assert node._selection_held
    frame(node, points, angles)
    assert node.commands[-1][0] == 0
    assert node.commands[-1][2] * direction > 0


@pytest.mark.parametrize('half_angle', [90, 120, 135])
def test_configured_sector_excludes_readings_outside_its_bounds(node, half_angle):
    node.p['view_half_angle_deg'] = half_angle
    points, angles = geometry([half_angle + 1, -half_angle - 1, 180], [2, 2, 2])
    assert node.select_direction(points, angles, 0.0, 0.3) is None


def test_wide_sector_does_not_override_collision_checks(node):
    node.p['view_half_angle_deg'] = 135.0
    points, angles = geometry([-135, -90, -45, 0, 45, 90, 135], [0.20] * 7)
    frame(node, points, angles)
    assert node.commands[-1] == (0, 0, 0)
    assert node.selected_heading_deg is None
