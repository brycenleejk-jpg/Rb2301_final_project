"""Starting-width switching and motion checks without starting ROS nodes."""
import math
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

@pytest.fixture(params=['handwrite', 'obstacle_course'])
def controller(monkeypatch, request):
    module = importlib.import_module(f'rb2301_fp.{request.param}')
    monkeypatch.setattr(module.time, 'monotonic', lambda: 100.0)
    n = module.ObstacleCourseNode.__new__(module.ObstacleCourseNode)
    n.p = module.PARAMS.copy()
    n.robot_radius = n.p['robot_radius_m']
    n.max_translate_velocity = 0.3
    n.pose = np.array([0., 0., 0.])
    n.last_pose_time = n.last_scan_time = 100.0
    n.last_scan = object()
    n.calibrate = False
    n.raw_pose = None
    n._ticks = 0
    n.segment = 'A'
    n.width_checked = False
    n.start_pose = n.starting_width = n.switch_right_m = None
    n.start_left_m = n.start_right_m = None
    n.commands, n.logs = [], []
    n.publisher_ = SimpleNamespace(publish=lambda m: n.commands.append(
        (m.linear.x, m.linear.y, m.angular.z)))
    n.get_logger = lambda: SimpleNamespace(info=lambda msg: n.logs.append(msg))
    return n


def measure(n):
    points = np.array([[0., .6], [0., -.4]])
    angles = np.array([math.pi / 2, -math.pi / 2])
    assert n.measurement_starting_width(points, angles)


def test_width_uses_both_sides_and_threshold_uses_right_side(controller):
    n = controller
    measure(n)
    assert n.starting_width == pytest.approx(1.0)
    assert n.start_right_m == pytest.approx(.4)
    assert n.switch_right_m == pytest.approx(.4 + .11 + .03)
    n.pose[:] = [1., -1., 90.]
    assert n.measurement_starting_width(np.empty((0, 2)), np.array([]))
    assert n.starting_width == pytest.approx(1.0)
    assert n.start_pose == pytest.approx([0., 0., 0.])


@pytest.mark.parametrize('points,angles', [
    ([[0., .6]], [math.pi / 2]),
    ([[0., .6], [0., math.nan]], [math.pi / 2, -math.pi / 2]),
    ([[0., .6], [0., .4]], [math.pi / 2, -math.pi / 2]),
])
def test_failed_width_measurement_does_not_capture_start(controller, points, angles):
    n = controller
    assert not n.measurement_starting_width(np.array(points), np.array(angles))
    assert not n.width_checked and n.start_pose is None


@pytest.mark.parametrize('heading', [0., 45., 90., -90., 179.])
def test_switch_uses_fixed_start_frame_and_returns_on_reentry(controller, heading):
    n = controller
    n.pose = np.array([2., 3., heading])
    measure(n)
    r = math.radians(heading)
    forward = np.array([math.cos(r), math.sin(r)])
    right = np.array([math.sin(r), -math.cos(r)])
    for progress, segment in [(0., 'A'), (n.switch_right_m - .001, 'A'),
                              (n.switch_right_m + .001, 'B'), (0., 'A')]:
        n.pose[:2] = n.start_pose[:2] + 2 * forward + progress * right
        n.pose[2] = heading - 70 # Current rotation must not rotate the reference frame.
        assert n.rightward_displacement() == pytest.approx(progress)
        assert n.x_position == pytest.approx(2.)
        assert n.y_position == pytest.approx(-progress)
        n.update_segment()
        assert n.segment == segment


def test_switch_at_threshold_and_goals_are_relative_to_start(controller):
    n = controller
    n.pose[2] = 170.
    measure(n)
    assert n.virtual_bearing_deg() == pytest.approx(-45.)
    n.pose[2] = -170.
    assert n.virtual_bearing_deg() == pytest.approx(-65.)
    n.segment = 'B'
    assert n.virtual_bearing_deg() == pytest.approx(-20.)
    n.segment = 'A'
    n.start_pose[:] = [0., 0., 0.]
    n.pose[:] = [0., -n.switch_right_m, 0.]
    n.update_segment()
    assert n.segment == 'B'


@pytest.mark.parametrize('fault', ['pose_missing', 'pose_invalid', 'pose_stale',
                                  'scan_missing', 'scan_stale', 'calibration'])
def test_timer_stops_before_planning_on_missing_or_stale_input(controller, fault):
    n = controller
    if fault == 'pose_missing':
        n.pose = None
    elif fault == 'pose_invalid':
        n.pose[0] = math.nan
    elif fault == 'pose_stale':
        n.last_pose_time = 99.
    elif fault == 'scan_missing':
        n.last_scan = None
    elif fault == 'scan_stale':
        n.last_scan_time = 99.
    else:
        n.calibrate = True
    n.timer_callback()
    assert n.commands[-1] == (0., 0., 0.)
    assert not n.width_checked


def test_timer_waits_for_both_sides_and_stops_when_no_readings(controller):
    n = controller
    for points, angles in [(np.array([[0., .6]]), np.array([math.pi / 2])),
                           (np.empty((0, 2)), np.array([]))]:
        n.scan_geometry = lambda: (points, angles)
        n.timer_callback()
        assert n.commands[-1] == (0., 0., 0.)
        assert not n.width_checked


def test_full_scan_measures_once_drives_and_switches(controller):
    n = controller
    scan = LaserScan(angle_min=-math.pi, angle_increment=math.pi / 360,
                     range_min=.05, range_max=10., ranges=[2.] * 720)
    n.sub_scan_callback(scan)
    n.timer_callback()
    assert n.width_checked
    assert n.starting_width == pytest.approx(4., abs=.02)
    assert n.commands[-1][0] > 0
    assert n.commands[-1][2] < 0
    old_start = n.start_pose.copy()
    n.pose[1] = -n.switch_right_m - .001
    n.timer_callback()
    assert n.segment == 'B'
    assert n.virtual_bearing_deg() == pytest.approx(0.)
    assert abs(n.commands[-1][2]) < math.radians(8.)
    assert n.start_pose == pytest.approx(old_start)


def test_blocked_robot_stops(controller):
    n = controller
    measure(n)
    n.scan_geometry = lambda: (np.array([[0., .05]]), np.array([math.pi / 2]))
    n.timer_callback()
    assert n.commands[-1] == (0., 0., 0.)


def test_reduced_clearance_also_reduces_speed(controller):
    n = controller
    n.p['preferred_clearance_m'] = .17
    points, angles = np.array([[0., .14]]), np.array([0.])
    selected = n.select_direction(points, angles, 0., .3)
    assert selected == 0.
    assert n._selection_radius == n.robot_radius
    n.drive_toward(points, selected, .3)
    assert n.commands[-1] == pytest.approx((n.p['reduced_clearance_speed_mps'], 0., 0.))


def test_large_angle_only_turns_and_curve_collision_is_rejected(controller):
    n = controller
    n._selection_radius = .17
    n.drive_toward(np.array([[2., 2.]]), math.pi / 2, .3)
    assert n.commands[-1][0] == 0.
    assert n.commands[-1][2] > 0.
    # Obstacle at the predicted centre after 0.2 m along a left-turning arc.
    point = np.array([[.3 * math.sin(.2 / .3), .3 * (1 - math.cos(.2 / .3))]])
    assert not n.motion_is_clear(point, .3, 1., .3)
    assert not n.motion_is_clear(np.array([[0., .14]]), 0., 1., .3, radius=.17)


def test_odometry_normalizes_quaternion_and_rejects_invalid_pose(controller):
    n = controller
    msg = Odometry()
    msg.pose.pose.orientation = Quaternion(z=2 * math.sin(math.pi / 4),
                                           w=2 * math.cos(math.pi / 4))
    n.odometer_callback(msg)
    assert n.pose[2] == pytest.approx(90.)
    assert n.last_pose_time == 100.
    msg.pose.pose.position.x = math.nan
    n.odometer_callback(msg)
    assert n.pose is None
    msg.pose.pose.position.x = 0.
    msg.pose.pose.orientation = Quaternion(x=0., y=0., z=0., w=0.)
    n.odometer_callback(msg)
    assert n.pose is None


def geometry(degrees, ranges):
    angles = np.radians(degrees)
    ranges = np.asarray(ranges)
    return np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles))), angles


def test_depth_weight_selects_open_space_instead_of_shallow_forward_path(controller):
    n = controller
    points, angles = geometry([0., 135.], [.61, 3.11])
    n.p['depth_weight'] = 1.5
    assert n.select_direction(points, angles, 0., .3) == 0.
    n.p['depth_weight'] = 3.5
    selected = n.select_direction(points, angles, 0., .3)
    assert selected == pytest.approx(math.radians(135.))
    n.drive_toward(points, selected, .3)
    assert n.commands[-1][0] == 0.
    assert n.commands[-1][2] > 0.


def test_near_equal_scores_prefer_depth(controller):
    n = controller
    n.p.update(depth_weight=3.5, virtual_goal_weight=1.5) # Keep this score-gap example independent of tuning.
    points, angles = geometry([-10., 30.], [1.11, 1.25])
    assert n.select_direction(points, angles, 0., .3) == pytest.approx(math.radians(30.))


def test_each_scan_selects_independently(controller):
    n = controller
    for i in range(10):
        ranges = [2.001, 1.999] if i % 2 == 0 else [1.999, 2.001]
        points, angles = geometry([-30., 30.], ranges)
        selected = n.select_direction(points, angles, 0., .3)
        expected = -30. if i % 2 == 0 else 30.
        assert selected == pytest.approx(math.radians(expected))
    assert not hasattr(n, 'selected_heading_deg')


def test_turn_command_depends_only_on_the_current_direction(controller):
    n = controller
    n._selection_radius = n.robot_radius
    points = np.array([[2., 2.]])
    n.drive_toward(points, math.radians(70.), .3)
    assert n.commands[-1][0] == 0.
    n.drive_toward(points, math.radians(30.), .3)
    assert n.commands[-1][0] > 0.
    assert not hasattr(n, '_alignment_active')


def test_reentry_buffer_prevents_boundary_chatter(controller):
    n = controller
    measure(n)
    reentry = n.start_right_m - n.robot_radius - n.p['switch_tolerance_m']
    for progress, expected in [(n.switch_right_m - .001, 'A'),
                               (n.switch_right_m + .001, 'B'),
                               (n.switch_right_m - .001, 'B'),
                               (reentry + .001, 'B'), (reentry - .001, 'A')]:
        n.pose[1] = -progress
        n.update_segment()
        assert n.segment == expected
    assert any(message.startswith('SEGMENT |') for message in n.logs)


def test_blocked_curve_aligns_instead_of_using_straight_fallback(controller):
    n = controller
    n._selection_radius = .17
    # This point is outside the straight footprint but inside the curve's padding.
    points = np.array([[-.175, 0.]])
    selected = math.radians(20.)
    speed = .3 * math.cos(selected)
    assert n.path_is_clear(points, 0., .3, radius=.17)
    assert not n.motion_is_clear(points, speed, selected, .3, radius=.17)
    n.drive_toward(points, selected, .3)
    assert n.commands[-1][0] == 0.
    assert n.commands[-1][2] > 0.


def test_no_supported_candidates_stops(controller):
    n = controller
    measure(n)
    points, angles = geometry([180.], [2.])
    n.scan_geometry = lambda: (points, angles)
    n.timer_callback()
    assert n.commands[-1] == (0., 0., 0.)
    assert any('no-rays-in-view' in line for line in n.logs)


def test_status_is_throttled_and_reports_current_direction_and_actual_commands(controller, monkeypatch):
    n = controller
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('time.monotonic', lambda: clock.now)
    n._selection_radius = .17
    n.log_data()['direction'] = 20. # A display value, not a navigation input.
    n.move_2D(x=1., turn=1.)
    for _ in range(10):
        n.report_status('DRIVE', 'curve-clear')
    statuses = [line for line in n.logs if line.startswith('DRIVE ')]
    assert len(statuses) == 1
    assert 'direction=+20.0deg' in statuses[0]
    assert 'v=0.30 w=+0.60' in statuses[0]
    assert not any(word in statuses[0] for word in ('target=', 'memory=', 'kept-score', 'held-turn'))
    clock.now += .51
    n.report_status('DRIVE', 'curve-clear')
    assert len([line for line in n.logs if line.startswith('DRIVE ')]) == 2


def test_fast_turn_reversals_are_counted_even_when_event_logs_are_throttled(controller, monkeypatch):
    n = controller
    clock = SimpleNamespace(now=100.)
    monkeypatch.setattr('time.monotonic', lambda: clock.now)
    n._selection_radius = .17
    n.log_data()['direction'] = 0.
    n.move_2D(turn=.2)
    n.report_status('TURN', 'test')
    for turn in [-.2, .2, -.2]:
        clock.now += .05
        n.move_2D(turn=turn)
        n.report_status('TURN', 'test')
    clock.now = 100.6
    n.report_status('TURN', 'test')
    statuses = [line for line in n.logs if line.startswith(('TURN A |', 'TURN B |'))]
    assert len(statuses) == 2
    assert 'turn-direction-changes=3' in statuses[-1]
    reversals = [line for line in n.logs if line.startswith('TURN CHANGE |')]
    assert len(reversals) == 2 # First event and a later summary, rather than every flip.
    assert '2 events suppressed' in reversals[-1]


@pytest.mark.parametrize('degrees', [.1, -.1, 2.1, -2.1, 5., -5.])
def test_near_forward_uses_checked_straight_path_instead_of_minimum_turn(controller, degrees):
    n = controller
    n._selection_radius = .17
    points = np.array([[-.175, 0.]])
    angle = math.radians(degrees)
    assert not n.motion_is_clear(points, .3 * math.cos(angle), angle, .3, radius=.17)
    assert n.path_is_clear(points, 0., .3, radius=.17)
    n.drive_toward(points, angle, .3)
    assert n.commands[-1] == pytest.approx((.3 * math.cos(angle), 0., 0.))
    assert any('DRIVE ' in line and 'straight-clear' in line for line in n.logs)


@pytest.mark.parametrize('stale,expected', [('scan', 'scan-stale:0.75s'),
                                          ('pose', 'pose-stale:0.75s')])
def test_stop_log_identifies_which_sensor_is_stale(controller, stale, expected):
    n = controller
    if stale == 'scan':
        n.last_scan_time = 99.25
    else:
        n.last_pose_time = 99.25
    n.timer_callback()
    assert any(line.startswith('STOP ') and expected in line for line in n.logs)


def test_near_forward_does_not_drive_through_a_blocked_straight_path(controller):
    n = controller
    n._selection_radius = .17
    n.drive_toward(np.array([[.2, 0.]]), math.radians(.1), .3)
    assert n.commands[-1] == (0., 0., 0.)
    assert any('near-forward-path-blocked' in line for line in n.logs)


def test_straight_fallback_keeps_the_selected_clearance(controller):
    n = controller
    n._selection_radius = .17
    points = np.array([[0., .15]])
    assert n.path_is_clear(points, 0., .3, radius=n.robot_radius)
    assert not n.path_is_clear(points, 0., .3, radius=.17)
    n.drive_toward(points, math.radians(.1), .3)
    assert n.commands[-1] == (0., 0., 0.)


def test_straight_fallback_does_not_override_a_larger_selected_turn(controller):
    n = controller
    n._selection_radius = .17
    n.drive_toward(np.array([[-.175, 0.]]), math.radians(5.1), .3)
    assert n.commands[-1][0] == 0.
    assert n.commands[-1][2] > 0.


def test_small_ray_angle_changes_do_not_cause_spinning_in_the_timer(controller):
    n = controller
    n.p['preferred_clearance_m'] = .17
    measure(n)
    for degrees in [.1, -.1, 2.1, -2.1] * 3:
        points, angles = geometry([180., degrees], [.175, 2.])
        n.scan_geometry = lambda: (points, angles)
        n.timer_callback()
        assert n.commands[-1][0] > 0.
        assert n.commands[-1][2] == 0.
