"""Raycast scans and controller commands without starting ROS nodes or motors."""
from collections import deque
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from rb2301_fp.moving_obstacle_test import (
    PARAMS, MotionDetector, MovingObstacleTest, ScanFrame, SidePositioner,
    in_start_frame, measured_motion, motion_clusters, pose_at, scan_points,
)


WALLS = [([-2., .8], [4., .85]), ([-2., -.85], [4., -.8])]


def scan_of_boxes(pose, boxes, stamp=0.):
    """Raycast the mounted 360-ray LiDAR into world-frame boxes."""
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    sensor = pose[:2] + np.array([c * PARAMS["lidar_x_m"], s * PARAMS["lidar_x_m"]])
    world_angles = angles + math.radians(PARAMS["lidar_yaw_deg"]) + pose[2]
    rays = np.column_stack((np.cos(world_angles), np.sin(world_angles)))
    ranges = np.full(360, np.inf)
    for lower, upper in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (np.asarray(lower) - sensor) / rays, (np.asarray(upper) - sensor) / rays
        near = np.minimum(a, b).max(axis=1)
        far = np.maximum(a, b).min(axis=1)
        ranges = np.minimum(ranges, np.where((far >= near) & (near > 0), near, np.inf))
    return SimpleNamespace(
        ranges=ranges, angle_min=-math.pi, angle_increment=2*math.pi/360,
        range_min=.05, range_max=10.,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=int(stamp), nanosec=round((stamp % 1)*1e9))))


def scene(pose, bar_y=0., stamp=0., bar_x=1.1, width=.3):
    return scan_of_boxes(pose, [([bar_x, bar_y-width/2], [bar_x+.05, bar_y+width/2])] + WALLS, stamp)


def update(detector, scan, pose, stamp):
    angles, points, covered = scan_points(scan, detector.p)
    return detector.update(angles, points, covered, pose, stamp, scan.range_max)


def test_scan_points_preserve_rays_and_mount_correction():
    angles, points, covered = scan_points(scene(np.zeros(3)), PARAMS)
    assert len(points) == 360
    assert np.all(np.diff(angles) > 0)
    assert np.all(covered)
    front = np.abs(angles) < .01
    assert points[front, 0] == pytest.approx([1.1])


def test_changed_points_form_clusters_without_a_bar_shape_requirement():
    irregular = np.array([[1., 0.], [1.06, .075], [1., .15], [1.06, .225], [1., .3]])
    groups = motion_clusters(irregular, .10, 3)
    assert len(groups) == 1
    assert len(groups[0]) == 5
    assert not motion_clusters(irregular, .03, 3)


def test_fixed_frame_removes_robot_rotation_and_translation():
    pose = np.array([.3, -.2, .4])
    fixed = np.array([[1., .2], [2., -.3]])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    robot = (fixed - pose[:2]) @ np.array([[c, -s], [s, c]])
    assert in_start_frame(robot, pose, np.zeros(3)) == pytest.approx(fixed)


@pytest.mark.parametrize("moving,width", [(False, .3), (True, .3), (True, .1), (True, .55)])
def test_ego_compensated_lateral_motion_without_bar_width_filter(moving, width):
    detector = MotionDetector(PARAMS)
    detection = None
    for i in range(14):
        stamp = i*.1
        pose = np.array([.1*stamp, .02*stamp, math.radians(3)*stamp])
        scan = scene(pose, -.3 + (.6*stamp if moving else 0.), stamp, width=width)
        detection, status = update(detector, scan, pose, stamp)
        if detection:
            assert moving
            assert 1.1-pose[0] > .5
            break
    assert bool(detection) is moving, status


def test_longitudinal_object_motion_is_detected_too():
    detector = MotionDetector(PARAMS)
    detection = None
    for i in range(14):
        stamp = i*.1
        pose = np.array([.1*stamp, 0., 0.])
        detection, status = update(detector, scene(pose, 0., stamp, bar_x=1.3-.3*stamp), pose, stamp)
        if detection:
            break
    assert detection, status


@pytest.mark.parametrize("period", [2.83, 3.92])
@pytest.mark.parametrize("phase", [0., math.pi/2, math.pi, 3*math.pi/2])
def test_oscillating_gate_at_different_starting_phases(period, phase):
    detector = MotionDetector(PARAMS)
    detection = None
    for i in range(31):
        stamp = i*.1
        pose = np.array([.1*stamp, 0., 0.])
        y = .6 * math.sin(2*math.pi*stamp/period + phase)
        detection, status = update(detector, scene(pose, y, stamp, bar_x=1.3), pose, stamp)
        if detection:
            break
    assert detection, status


def test_moving_object_seen_at_an_angle_is_not_rejected_for_bar_shape():
    detector = MotionDetector(PARAMS)
    detection = None
    for i in range(15):
        stamp = i*.1
        heading = math.radians(25)
        pose = np.array([.1*stamp*math.cos(heading), .1*stamp*math.sin(heading), heading])
        detection, status = update(detector, scene(pose, -.2+.6*stamp, stamp), pose, stamp)
        if detection:
            break
    assert detection, status


def test_missing_returns_and_partial_visibility_do_not_make_static_object_move():
    detector = MotionDetector(PARAMS)
    rng = np.random.default_rng(13)
    for i in range(20):
        stamp = i*.1
        pose = np.array([.1*stamp, 0., math.radians(2)*stamp])
        scan = scene(pose, .15, stamp)
        scan.ranges[rng.random(360) < .15] = np.nan
        detection, _ = update(detector, scan, pose, stamp)
        assert detection is None


def test_static_scene_with_range_noise_and_robot_turning_is_not_dynamic():
    detector = MotionDetector(PARAMS)
    rng = np.random.default_rng(7)
    for i in range(25):
        stamp = i*.1
        pose = np.array([.1*stamp, .01*stamp, math.radians(4)*stamp])
        scan = scene(pose, .1, stamp)
        finite = np.isfinite(scan.ranges)
        scan.ranges[finite] += rng.normal(0., .01, np.count_nonzero(finite))
        detection, _ = update(detector, scan, pose, stamp)
        assert detection is None


def test_newly_appearing_but_stationary_object_does_not_confirm_motion():
    detector = MotionDetector(PARAMS)
    for i in range(15):
        stamp = i*.1
        pose = np.array([.05*stamp, 0., 0.])
        boxes = WALLS + ([([1., -.15], [1.05, .15])] if i >= 3 else [])
        detection, _ = update(detector, scan_of_boxes(pose, boxes, stamp), pose, stamp)
        assert detection is None


def test_occluded_or_missing_old_rays_do_not_count_as_previously_free():
    pose = np.zeros(3)
    scan = scan_of_boxes(pose, [([.6, -.2], [.65, .2])] + WALLS)
    angles, points, covered = scan_points(scan, PARAMS)
    frame = ScanFrame(0., pose, angles, points, covered)
    query = np.array([[1.1, 0.], [.4, 0.]])
    assert frame.free_before(query, PARAMS).tolist() == [False, True]
    frame.covered[:] = False
    assert not np.any(frame.free_before(query, PARAMS))


def test_no_return_does_not_claim_free_space_beyond_sensor_range():
    scan = scan_of_boxes(np.zeros(3), [])
    angles, points, covered = scan_points(scan, PARAMS)
    frame = ScanFrame(0., np.zeros(3), angles, points, covered, range_max=.5)
    assert frame.free_before(np.array([[.4, 0.], [1., 0.]]), PARAMS).tolist() == [True, False]


def test_large_pose_mismatch_fails_background_consistency_check():
    detector = MotionDetector(PARAMS)
    scan = scene(np.zeros(3))
    update(detector, scan, np.zeros(3), 0.)
    detection, status = update(detector, scan, np.array([0., .25, 0.]), .2)
    assert detection is None
    assert "insufficient stationary background" in status


def test_no_stationary_background_is_reported_as_uncertain():
    detector = MotionDetector(PARAMS)
    for i in range(8):
        stamp = i*.1
        pose = np.zeros(3)
        scan = scan_of_boxes(pose, [([1., -.3+.06*i], [1.05, .06*i])], stamp)
        detection, status = update(detector, scan, pose, stamp)
        assert detection is None
    assert "uncertain" in status


def test_repeated_scan_cannot_build_motion_confirmation():
    detector = MotionDetector(PARAMS)
    for i in range(10):
        detection, _ = update(detector, scene(np.zeros(3), .05*i), np.zeros(3), 0.)
        assert detection is None
    assert len(detector.frames) == 1


def test_visibility_gap_forgets_old_motion_track():
    detector = MotionDetector(PARAMS)
    detector.update_tracks(np.array([[1., 0.]]), 0.)
    detector.update_tracks(np.array([[1., .2]]), .5)
    assert detector.tracks[0].number == 2
    assert len(detector.tracks[0].samples) == 1


def test_reordering_two_static_changed_regions_does_not_create_motion():
    detector = MotionDetector(PARAMS)
    for i in range(10):
        points = np.array([[1., -.2], [1.05, .2]])
        if i % 2:
            points = points[::-1]
        detection, _ = detector.update_tracks(points, i*.1)
        assert detection is None
    assert len(detector.tracks) == 2


def test_pose_interpolation_wraps_heading_and_rejects_old_pose():
    poses = deque([(0., np.array([0., 0., math.radians(179)])),
                   (.1, np.array([.1, 0., math.radians(-179)]))])
    middle = pose_at(poses, .05, .05)
    assert middle[:2] == pytest.approx([.05, 0.])
    assert abs(middle[2]) == pytest.approx(math.pi)
    assert pose_at(poses, 1., .05) is None


def test_measured_stop_uses_actual_pose_motion():
    poses = deque([(0., np.zeros(3)), (.2, np.array([.02, 0., .04]))])
    assert measured_motion(poses, .2, .05) == pytest.approx((.1, .2))
    assert measured_motion(poses, 1., .05) is None


def test_pose_is_not_interpolated_across_a_localization_dropout():
    poses = deque([(0., np.zeros(3)), (1., np.array([.1, 0., 0.]))])
    assert pose_at(poses, .5, .05) is None


def test_coverage_and_body_bounds_block_unsafe_forward_commands():
    node = SimpleNamespace(p=PARAMS)
    angles = np.radians(np.arange(-24, 25))
    covered = np.ones(len(angles), dtype=bool)
    points = np.full((len(angles), 2), np.nan)
    assert MovingObstacleTest.clear(node, angles, points, covered, 0., .2)
    points[24] = [.3, 0.]
    assert not MovingObstacleTest.clear(node, angles, points, covered, 0., .2)
    points[24] = np.nan
    covered[:20] = False
    assert not MovingObstacleTest.clear(node, angles, points, covered, 0., .2)


@pytest.mark.parametrize("y,expected_direction", [(-.4, -1), (.4, 1)])
def test_side_sequence_retains_nearer_wall_and_strafe(y, expected_direction):
    angles, points, _ = scan_points(scene(np.array([0., y, 0.])), PARAMS)
    positioner = SidePositioner(PARAMS)
    command, _ = positioner.command(angles, points, lambda *args, **kw: True)
    assert positioner.side == expected_direction
    assert command[0] == command[2] == 0.
    assert command[1] * expected_direction > 0.


def test_side_positioning_completes_at_target_clearance():
    angles, points, _ = scan_points(scene(np.array([0., .55, 0.])), PARAMS)
    positioner = SidePositioner(PARAMS)
    command, status = positioner.command(angles, points, lambda *args, **kw: True)
    assert positioner.done
    assert command == (0., 0., 0.)
    assert "POSITIONED" in status


def make_node():
    node = MovingObstacleTest.__new__(MovingObstacleTest)
    node.p = PARAMS.copy()
    node.scan = None
    node.scan_received = 0.
    node.poses = deque(maxlen=100)
    node.origin = node.processed_stamp = None
    node.state = "APPROACH"
    node.stopped_since = node.aligned_since = None
    node.last_log = -math.inf
    node.last_forward_command = 0.
    node.motion_ready = True
    node.detector = MotionDetector(node.p)
    node.side = SidePositioner(node.p)
    node.commands, node.logs = [], []
    node.pub = SimpleNamespace(publish=lambda msg: node.commands.append((msg.linear.x, msg.linear.y, msg.angular.z)))
    node.get_logger = lambda: SimpleNamespace(info=node.logs.append, warning=node.logs.append, error=node.logs.append)
    return node


def feed(node, stamp, pose, bar_y=0.):
    node.poses.append((stamp, pose))
    node.scan_callback(scene(pose, bar_y, stamp))
    node.tick()


def test_controller_approaches_detects_stops_aligns_then_strafes():
    node = make_node()
    for i in range(14):
        stamp = i*.1
        pose = np.array([.1*stamp, 0., 0.])
        feed(node, stamp, pose, -.3 + .6*stamp)
        if node.state == "STOP":
            break
    assert any(cmd[0] > 0 for cmd in node.commands)
    assert node.state == "STOP"
    assert node.commands[-1] == (0., 0., 0.)
    assert any("MOVING OBJECT DETECTED" in log and "forward_command_active=True" in log for log in node.logs)
    states = []
    for _ in range(15):
        stamp += .1
        feed(node, stamp, pose)
        states.append(node.state)
        if node.state == "SIDE":
            break
        assert node.commands[-1][1] == 0.
    assert "ALIGN" in states and node.state == "SIDE"
    feed(node, stamp+.1, pose)
    assert node.commands[-1][0] == 0.
    assert abs(node.commands[-1][1]) > 0.


def test_stop_does_not_complete_while_robot_still_moves():
    node = make_node()
    node.state = "STOP"
    for i in range(10):
        feed(node, i*.1, np.array([i*.01, 0., 0.]))
        assert node.state == "STOP"
        assert node.commands[-1] == (0., 0., 0.)


def test_alignment_turns_toward_absolute_zero_without_translation():
    node = make_node()
    node.state = "ALIGN"
    for i in range(5):
        feed(node, i*.1, np.array([0., 0., math.radians(8)]))
        assert node.state == "ALIGN"
        assert node.commands[-1][0:2] == (0., 0.)
        assert node.commands[-1][2] < 0.


def test_side_pauses_and_realigns_if_heading_drifts():
    node = make_node()
    node.state = "SIDE"
    feed(node, 0., np.array([0., 0., math.radians(4)]))
    assert node.state == "ALIGN"
    assert node.commands[-1] == (0., 0., 0.)


def test_unmatched_pose_or_stale_scan_cannot_command_forward():
    node = make_node()
    node.scan_callback(scene(np.zeros(3)))
    node.tick()
    assert node.commands[-1] == (0., 0., 0.)
    node.poses.append((0., np.zeros(3)))
    node.scan_received = time.monotonic()-1.
    node.tick()
    assert node.commands[-1] == (0., 0., 0.)


def test_duplicate_scan_timestamp_cannot_keep_scan_fresh():
    node = make_node()
    node.scan_callback(scene(np.zeros(3), stamp=0.))
    node.scan_received = time.monotonic()-1.
    node.scan_callback(scene(np.zeros(3), stamp=0.))
    node.tick()
    assert node.commands[-1] == (0., 0., 0.)
    assert time.monotonic()-node.scan_received > .5


def test_approach_stops_when_static_background_cannot_validate_motion():
    node = make_node()
    for i in range(4):
        stamp = i*.1
        pose = np.array([.01*i, 0., 0.])
        node.poses.append((stamp, pose))
        node.scan_callback(scan_of_boxes(pose, [([1., -.15], [1.05, .15])], stamp))
        node.tick()
    assert node.state == "APPROACH"
    assert node.commands[-1] == (0., 0., 0.)
    node.tick()  # No fresh scan yet: uncertainty must continue to inhibit motion.
    assert node.commands[-1] == (0., 0., 0.)
