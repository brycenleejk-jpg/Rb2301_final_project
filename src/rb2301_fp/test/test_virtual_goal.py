"""Virtual-bearing, measured-yaw and steering checks with real ROS messages."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry

from rb2301_fp.obstacle_course import ObstacleCourseNode, PARAMS, wrap_deg


def quaternion(degrees):
    half = math.radians(degrees) / 2
    return Quaternion(z=math.sin(half), w=math.cos(half))


def geometry(degrees, ranges):
    angles = np.radians(degrees)
    ranges = np.asarray(ranges)
    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles))), angles


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=100.0)
    monkeypatch.setattr('rb2301_fp.obstacle_course.time.monotonic', lambda: value.now)
    return value


@pytest.fixture
def virtual_node(node, clock):
    node.p = PARAMS.copy()
    node.heading_deg = None
    node.last_heading_time = 0.0
    return node


def sample(node, clock, yaw, dt=0.1):
    clock.now += dt
    ns = round(clock.now * 1e9)
    msg = Odometry()
    msg.header.stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
    msg.pose.pose.orientation = quaternion(yaw)
    # Deliberately invalid positions prove that navigation does not read them.
    msg.pose.pose.position.x = float('nan')
    msg.pose.pose.position.y = float('inf')
    node.odometer_callback(msg)
    return msg


def switch(node, clock, initial=0):
    sample(node, clock, initial)
    threshold = node.p["segment_right_turn_deg"]
    for right in np.linspace(threshold / 9, threshold, 9):
        sample(node, clock, initial - right)
    for _ in range(node.p['segment_confirm_samples'] - 1):
        sample(node, clock, initial - threshold)
    assert node.segment == 2


def drive_frame(node, clock, degrees, ranges):
    points, angles = geometry(degrees, ranges)
    node.scan_geometry = lambda: (points, angles)
    node.last_scan_time = clock.now
    node.timer_callback()
    return points, angles


@pytest.mark.parametrize('threshold', [45.0, 90.0])
@pytest.mark.parametrize('initial', [0, 50, 179, -179])
def test_switch_uses_relative_signed_yaw_from_any_start(virtual_node, clock, initial, threshold):
    n = virtual_node
    n.p['segment_right_turn_deg'] = threshold
    sample(n, clock, initial)
    assert n.heading_deg == 0
    for right in np.linspace(threshold / 10, threshold * 0.9, 9):
        sample(n, clock, initial - right)
        assert n.segment == 1
        assert n.virtual_bearing_deg() == pytest.approx(wrap_deg(-45 + right))
    for i in range(n.p['segment_confirm_samples']):
        sample(n, clock, initial - threshold)
        assert n.segment == (2 if i == n.p['segment_confirm_samples'] - 1 else 1)
    assert n.net_right_turn_deg == pytest.approx(threshold)
    assert n.virtual_bearing_deg() == pytest.approx(wrap_deg(threshold))


def test_left_turns_subtract_instead_of_accumulating_only_right(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 45.0
    sample(n, clock, 0)
    for yaw in [-10, -20, -30, -20, -10, -20, -30, -35]:
        sample(n, clock, yaw)
    assert n.net_right_turn_deg == pytest.approx(35)
    assert n.segment == 1
    for yaw in [-40, -45, -45, -45]:
        sample(n, clock, yaw)
    assert n.segment == 2


def test_switch_requires_consecutive_samples_and_never_reverts(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 45.0
    sample(n, clock, 0)
    for yaw in range(-5, -41, -5):
        sample(n, clock, yaw)
    for yaw in [-44, -45, -44.5, -45.1, -45.2]:
        sample(n, clock, yaw)
        assert n.segment == 1
    sample(n, clock, -45.3)
    assert n.segment == 2
    for yaw in range(-40, 1, 10):
        sample(n, clock, yaw)
        assert n.segment == 2


def test_after_switch_goal_returns_to_original_forward(virtual_node, clock):
    n = virtual_node
    switch(n, clock)
    total_left = n.p['segment_right_turn_deg']
    for left in np.arange(10, total_left + 1, 10):
        sample(n, clock, -n.p["segment_right_turn_deg"] + left)
        assert n.virtual_bearing_deg() == pytest.approx(wrap_deg(total_left - left))
    assert n.virtual_bearing_deg() == pytest.approx(0)
    # Once facing the virtual direction, a symmetric scan requests straight travel.
    drive_frame(n, clock, [-30, 0, 30], [2, 2, 2])
    assert n.commands[-1][0] > 0
    assert n.commands[-1][2] == pytest.approx(0)


def test_stage_one_prefers_right_and_switch_replaces_old_direction(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 90.0
    sample(n, clock, 0)
    drive_frame(n, clock, [-45, 45], [2, 2])
    assert n.commands[-1][2] < 0
    for yaw in list(range(-5, -91, -5)) + [-90, -90]:
        sample(n, clock, yaw)
    assert n.selected_heading_deg is None
    drive_frame(n, clock, [0, 90], [3, 1.11])
    assert 'selected=+90.0deg' in n._selection_summary
    assert n.commands[-1] == pytest.approx((0, 0, n.p['max_turn_rps']))


def test_left_door_can_beat_deeper_straight_corridor(virtual_node, clock):
    n = virtual_node
    switch(n, clock)
    drive_frame(n, clock, [0, 90], [3.11, 1.11])
    assert 'selected=+90.0deg' in n._selection_summary
    assert n.commands[-1][2] > 0
    assert n.commands[-1][0] == 0


def test_blocked_left_target_never_overrides_collision_check(virtual_node, clock):
    n = virtual_node
    switch(n, clock)
    points, _ = drive_frame(n, clock, [0, 90], [3.11, 0.25])
    assert 'selected=+0.0deg' in n._selection_summary
    assert n.commands[-1][0] > 0
    assert n.commands[-1][2] == pytest.approx(0)
    assert n.motion_is_clear(points, n.commands[-1][0], 0, n.p['check_distance_m'])


def test_left_90_reading_is_considered_at_sampling_boundary(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 90.0
    switch(n, clock)
    degrees = np.arange(-89.0, 91.0, 1.0)
    drive_frame(n, clock, degrees, np.full(len(degrees), 3.0))
    candidates = n.snapshots[-1][0][1]
    assert any(math.degrees(a) == pytest.approx(90) for a in candidates)
    # Depth tie-breaking may choose an adjacent ray within the existing score band.
    assert 87 <= n.selected_heading_deg - n.heading_deg <= 90 + 1e-9


def test_committed_turn_yields_to_a_clearly_better_option(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 90.0
    switch(n, clock)
    drive_frame(n, clock, [0, 90], [3.11, 1.11])
    assert n._alignment_active
    target = n.selected_heading_deg
    sample(n, clock, -n.p["segment_right_turn_deg"] + 10)
    drive_frame(n, clock, [80, 20], [1.11, 3.11])
    assert not n._selection_held
    assert n.selected_heading_deg != pytest.approx(target)
    assert 'selected=+20.0deg' in n._selection_summary
    assert not n._alignment_active
    assert n.commands[-1][0] > 0


def test_committed_turn_is_abandoned_immediately_if_blocked(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 90.0
    switch(n, clock)
    drive_frame(n, clock, [0, 90], [3.11, 1.11])
    sample(n, clock, -n.p["segment_right_turn_deg"] + 10)
    drive_frame(n, clock, [80, 20], [0.25, 3.11])
    assert not n._selection_held
    assert 'selected=+20.0deg' in n._selection_summary


def test_commitment_releases_when_aligned(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 90.0
    switch(n, clock)
    drive_frame(n, clock, [0, 90], [3.11, 1.11])
    for left in range(10, 81, 10):
        sample(n, clock, -n.p["segment_right_turn_deg"] + left)
    sample(n, clock, -n.p["segment_right_turn_deg"] + 85)
    drive_frame(n, clock, [5, 30], [3, 2])
    assert not n._alignment_active
    assert n.commands[-1][0] > 0


@pytest.mark.parametrize('value', ['scan', 'orientation'])
def test_stale_sensor_stops_even_with_an_active_virtual_goal(virtual_node, clock, value):
    n = virtual_node
    sample(n, clock, 0)
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1][0] > 0
    setattr(n, 'last_scan_time' if value == 'scan' else 'last_heading_time', clock.now - 1)
    n.timer_callback()
    assert n.commands[-1] == (0, 0, 0)
    assert n.selected_heading_deg is None
    assert not n._alignment_active


def test_orientation_resumes_after_a_gap_without_restart(virtual_node, clock):
    n = virtual_node
    sample(n, clock, 0)
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1][0] > 0
    clock.now += 0.6
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1] == (0, 0, 0)
    sample(n, clock, -10)
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1][0] > 0
    assert n.heading_deg == pytest.approx(-10)
    assert n.segment == 1


def test_large_measured_turn_is_accumulated_without_rate_filter(virtual_node, clock):
    n = virtual_node
    sample(n, clock, 0)
    sample(n, clock, -90)
    assert n.heading_deg == pytest.approx(-90)
    assert n.segment == 1
    for _ in range(n.p['segment_confirm_samples'] - 1):
        sample(n, clock, -90)
    assert n.segment == 2


def test_duplicate_messages_do_not_confirm_switch_or_refresh_sensor(virtual_node, clock):
    n = virtual_node
    sample(n, clock, 0)
    for yaw in np.linspace(-5, -n.p['segment_right_turn_deg'], 18):
        msg = sample(n, clock, yaw)
    last_fresh = n.last_heading_time
    for _ in range(10):
        clock.now += 0.1
        n.odometer_callback(msg)
    assert n.segment == 1
    assert n._switch_confirmations == 1
    assert n.last_heading_time == last_fresh
    drive_frame(n, clock, [0, 90], [3, 2])
    assert n.commands[-1] == (0, 0, 0)


@pytest.mark.parametrize('reset_to_zero', [False, True])
def test_orientation_continues_after_source_timestamp_reset(virtual_node, clock, reset_to_zero):
    n = virtual_node
    msg = sample(n, clock, 0)
    if reset_to_zero:
        msg.header.stamp = Time()
    else:
        msg.header.stamp.sec -= 1
    msg.pose.pose.orientation = quaternion(-10)
    n.odometer_callback(msg)
    assert n.heading_deg == pytest.approx(-10)
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1][0] > 0


def test_optitrack_supplies_signed_rotation_without_reading_position(virtual_node, clock):
    n = virtual_node
    threshold = n.p["segment_right_turn_deg"]
    for right in list(np.linspace(0, threshold, 10)) + [threshold, threshold]:
        clock.now += 0.1
        msg = PoseStamped()
        ns = round(clock.now * 1e9)
        msg.header.stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        msg.pose.orientation = quaternion(170 - right)
        msg.pose.position.x = float('nan')
        msg.pose.position.y = float('inf')
        n.optitrack_callback(msg)
    assert n.segment == 2
    assert n.net_right_turn_deg == pytest.approx(threshold)
    assert n.virtual_bearing_deg() == pytest.approx(wrap_deg(threshold))


def test_optitrack_resumes_after_dropout(virtual_node, clock):
    n = virtual_node
    msg = PoseStamped()
    msg.pose.orientation = quaternion(0)
    n.optitrack_callback(msg)
    clock.now += 0.6
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1] == (0, 0, 0)
    msg.pose.orientation = quaternion(-30)
    n.optitrack_callback(msg)
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1][0] > 0
    assert n.heading_deg == pytest.approx(-30)
    assert n.segment == 1


def test_commands_never_count_as_rotation(virtual_node, clock):
    n = virtual_node
    sample(n, clock, 0)
    for _ in range(100):
        n.move_2D(turn=-0.5)
    assert n.segment == 1
    assert n.net_right_turn_deg == 0


def test_monitor_mode_sends_only_zero_velocity(virtual_node, clock):
    n = virtual_node
    n.calibrate = True
    sample(n, clock, 0)
    drive_frame(n, clock, [-45, 0], [2, 2])
    assert n.commands[-1] == (0, 0, 0)


def test_virtual_plot_has_no_goal_coordinates_or_distance(virtual_node, clock):
    n = virtual_node
    messages = []
    n.publish_plot = messages.append
    sample(n, clock, 0)
    ObstacleCourseNode.plot_goal(n)
    assert messages[-1] == 'Goal: bearing=-45.0 deg, segment=1, net_right=-0.0 deg, radius=0.110 m'
    switch(n, clock)
    ObstacleCourseNode.plot_goal(n)
    bearing = wrap_deg(n.p['segment_right_turn_deg'])
    assert f'bearing={bearing:+.1f} deg, segment=2' in messages[-1]


@pytest.mark.parametrize('seed', [7, 24, 81])
@pytest.mark.parametrize('segment', [1, 2])
def test_closed_loop_open_space_settles_at_fixed_goal(virtual_node, clock, seed, segment):
    """Feed commanded turns back as measured yaw; no world position reaches the controller."""
    n = virtual_node
    rng = np.random.default_rng(seed)
    yaw = 0.0
    if segment == 2:
        switch(n, clock)
        yaw = -n.p['segment_right_turn_deg']
    degrees = np.arange(-180.0, 180.0, 5.0)
    history = []
    for _ in range(400):
        sample(n, clock, yaw + rng.normal(0, 0.15), dt=0.05)
        drive_frame(n, clock, degrees, 2.5 + rng.normal(0, 0.005, len(degrees)))
        speed, _, turn = n.commands[-1]
        yaw += math.degrees(turn * 0.05)
        history.append((yaw, speed, turn))
    assert n.segment == segment
    tail = np.array(history[-100:])
    # Each fixed goal becomes forward after alignment, so movement resumes.
    assert np.count_nonzero(tail[:, 1] > 0.1) >= 95
    assert np.ptp(tail[:, 0]) < 20
    assert abs(n.virtual_bearing_deg()) < 15
    target = n.p['virtual_goal_initial_deg' if segment == 1 else 'virtual_goal_after_turn_deg']
    assert abs(wrap_deg(yaw - target)) < 15


@pytest.mark.parametrize('initial', [0, 73, 179, -179])
@pytest.mark.parametrize('segment,target', [(1, -45), (2, 0)])
def test_goal_direction_stays_fixed_through_rotation_and_wraparound(virtual_node, clock, initial, segment, target):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 1000.0
    sample(n, clock, initial)
    n.segment = segment
    # Include the screenshot's near-half-turn and more than one complete turn.
    for heading in list(range(-10, -181, -10)) + list(range(-170, 381, 10)):
        sample(n, clock, initial + heading)
        assert n.heading_deg == pytest.approx(heading)
        assert wrap_deg(n.heading_deg + n.virtual_bearing_deg()) == pytest.approx(target)


def test_first_goal_is_straight_ahead_after_45_right_without_switching(virtual_node, clock):
    n = virtual_node
    n.p['segment_right_turn_deg'] = 90.0
    sample(n, clock, 0)
    for heading in range(-5, -46, -5):
        sample(n, clock, heading)
    assert n.segment == 1
    assert n.virtual_bearing_deg() == pytest.approx(0)
    drive_frame(n, clock, [-45, 0, 45], [2, 2, 2])
    assert n.commands[-1][0] > 0
    assert n.commands[-1][2] == pytest.approx(0)


def test_noise_does_not_accumulate_as_unsigned_rotation(virtual_node, clock):
    n = virtual_node
    rng = np.random.default_rng(2301)
    sample(n, clock, 170)
    for yaw in 170 + rng.normal(0, 0.2, 1000):
        sample(n, clock, yaw, dt=0.05)
        assert n.segment == 1
        assert abs(n.net_right_turn_deg) < 1
