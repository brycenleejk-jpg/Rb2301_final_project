# Moving-object test: approach, stop, align to zero, strafe

The separate controller is `rb2301_fp/moving_obstacle_test.py`. It starts before
the gate, facing approximately along the approach direction. It drives slowly,
looks for actual scene motion while compensating for robot movement, requests a
stop, verifies low measured speed, aligns to the target heading, then positions
beside the nearer wall. It holds there; gate crossing is outside this test.

The main obstacle-course controller is not modified. Use just this controller
as the source of `cmd_vel` for the test.

## Run

With ROS and the workspace sourced, from the repository root:

```bash
python3 src/rb2301_fp/rb2301_fp/moving_obstacle_test.py --ros-args -p use_sim_time:=true
```

The existing test entry point also works after rebuilding and sourcing the package:

```bash
ros2 run rb2301_fp moving_obstacle_test --ros-args -p use_sim_time:=true
```

Default topics are `scan`, `odom`, and `cmd_vel`. For OptiTrack instead:

```bash
python3 src/rb2301_fp/rb2301_fp/moving_obstacle_test.py --ros-args \
  -p pose_type:=pose_stamped \
  -p pose_topic:=/vrpn_mocap/bingda_003/pose
```

Select the actual robot's pose topic and verify heading and LiDAR calibration.
Scan and pose header timestamps must share a clock. Poses are matched/interpolated
at the scan time, with at most 0.05 s endpoint mismatch and no interpolation
across a gap longer than twice that tolerance. A continuously published pose
stream is required throughout approach, stop verification, heading alignment,
and strafing. If no initial stationary pose is published, the robot waits.

**Heading zero means 0 degrees in the selected pose source's coordinate frame.**
It does not reset the measured yaw or redefine the initial heading as zero.
Set `target_heading_deg` if the gate requires a different absolute heading.

## New motion detector

There is no 30 cm bar-width test, forward-depth shape test, or requirement that
points lie on one line. The detector looks for sustained movement of surfaces.

1. Convert each valid LiDAR ray into a point using the scan angles and sensor
   mounting offsets. Keep missing/no-return information separately.
2. Use measured robot pose to put current and older scan points in the same fixed
   coordinate frame. Robot translation and rotation are accounted for here.
3. Compare against a scan about 0.20 s old. Check stationary background outside
   the forward detection area. If insufficient background agrees, report motion
   as uncertain and stop approaching, rather than trusting the comparison.
4. Within the forward detection area, find current hits more than 4 cm from old
   surface hits. This alone does NOT prove motion: a stationary surface can
   become visible when an occluder moves or the robot changes viewpoint.
5. Require these current hits to lie in space the older scan actually observed
   as free. A location behind an old foreground return, outside the old field
   of view, or on missing rays is unknown and does not qualify. Three neighbouring
   old rays must support the free-space test.
6. Group nearby changed points spatially. Require at least three supporting
   points. This groups motion evidence, not entire objects of a particular shape.
7. Track those regions across fresh scans with one-to-one, speed-limited
   association. Require at least 0.30 s of observations, over 6 cm displacement,
   plausible speed, a coherent 2D trajectory, and two qualifying evaluations.
   Either forward/backward or sideways object movement can qualify.

A single range change, a newly appearing stationary object, or a shifting
visible centroid by itself is insufficient. Temporal observation is essential.
The overlapping middle of a sideways-moving flat bar can look unchanged; its
moving edges supply the evidence. Partial visibility can therefore delay or
prevent detection rather than immediately producing a motion label.

## Controller sequence

```text
APPROACH -> STOP -> ALIGN -> SIDE -> HOLD
```

- **APPROACH:** command 0.10 m/s forward while checking fresh scans. Stop if the
  path is occupied/unknown, scan/pose is unavailable, motion comparison is
  uncertain, or approach distance/heading limits are exceeded.
- **STOP:** command zero velocity. Measured speed must stay below 0.02 m/s and
  measured turn rate below 0.05 rad/s for 0.30 s before proceeding.
- **ALIGN:** translate neither forward nor sideways. Turn toward the absolute
  heading target. Stay within 2 degrees, with low measured speed/turn rate,
  for 0.20 s before proceeding. Check room for body rotation.
- **SIDE:** retain the friend's nearer-wall choice, two-sample wall estimate,
  and sideways clearance control. Heading alignment is handled explicitly by
  ALIGN. If heading drifts outside tolerance, stop strafing and return to ALIGN.
- **HOLD:** stop when the target wall clearance is reached. No gate crossing.

Speed estimates use pose change over 0.20 s, rather than requested motor commands.
The obstacle checks use the repository's collision-mesh bounds (0.182 m forward,
0.034 m rear, 0.096 m half-width) plus 0.04 m margin. Verify these against hardware.

## Parameters to tune

All settings live in the file's `PARAMS` dictionary and are exposed as ROS
parameters. For example:

```bash
python3 src/rb2301_fp/rb2301_fp/moving_obstacle_test.py --ros-args \
  -p use_sim_time:=true -p detect_far_m:=1.8 -p approach_speed_mps:=0.08
```

Restart after changing parameters; runtime updates are not applied to cached settings.

| Parameter | Default | Meaning |
|---|---:|---|
| `detect_near_m`, `detect_far_m` | 0.20, 1.50 m | Forward detection area relative to the robot origin. |
| `detect_half_width_m` | 1.0 m | Sideways detection-area extent. |
| `reference_delay_s` | 0.20 s | Separation between compared scans. Longer allows more movement but more visibility changes. |
| `point_change_m` | 0.04 m | Minimum separation from old surface points. Set above stationary scan/pose error. |
| `free_space_margin_m` | 0.04 m | Clearance required between the new point and the old ray's hit. |
| `min_static_points`, `min_static_fraction` | 12, 0.50 | Required background agreement outside the detection area. |
| `motion_cluster_radius_m` | 0.10 m | Connect nearby motion-evidence points. Not an object width limit. |
| `min_motion_points` | 3 | Minimum supporting changed points per region. |
| `observation_window_s`, `min_observation_s` | 0.40, 0.30 s | History retained and minimum observation duration. |
| `motion_distance_m` | 0.06 m | Minimum fixed-frame displacement over that history. |
| `min_object_speed_mps`, `max_object_speed_mps` | 0.15, 1.50 m/s | Accepted fitted speed range. |
| `motion_fit_error_m` | 0.025 m | Maximum RMS position error about the fitted 2D trajectory. |
| `confirm_scans` | 2 | Successive qualifying evaluations on distinct scans. |
| `track_gap_s` | 0.25 s | Forget motion tracks after longer gaps. |
| `association_slack_m` | 0.06 m | Association distance = slack + maximum speed times elapsed time. |
| `approach_speed_mps` | 0.10 m/s | Initial forward command. |
| `stop_ahead_m` | 0.20 m | Checked distance beyond the inflated robot nose. |
| `target_heading_deg` | 0 degrees | Absolute pose-frame heading to reach after stopping. |
| `heading_tolerance_deg` | 2 degrees | Required heading accuracy before and during strafing. |
| `wall_clearance_m`, `side_speed_mps` | 0.25 m, 0.10 m/s | Robot-origin wall clearance and sideways speed. |

First test a stationary object while the robot approaches. `background` should
show agreement and no moving-object event should occur. If background comparison
is uncertain, investigate pose timestamps, sensor mounting, localization quality,
and available background before relaxing thresholds. Then test the moving gate
at several phases and starting distances. Compare first-detection distance,
missed detections, and false detections. These defaults are starting values,
not settings validated on the physical robot.

## Logs and offline checks

The confirmation log contains:

```text
MOVING OBJECT DETECTED | forward_command_active=True measured_robot_motion=(...) | track=... dt=... dx=... dy=... speed=... fit=... confirm=2/2
STOP confirmed from measured pose; aligning heading
ALIGN confirmed; beginning side positioning
POSITIONED: wall distance=...; test complete
```

`dx`/`dy` are in the fixed pose frame. `forward_command_active` reports the
command before the stop; `measured_robot_motion` separately reports measured
translation speed and absolute turn rate. Approach diagnostics include agreeing
background points, changed points, points previously in free space, and groups.

Run the synthetic tests without starting ROS nodes or moving a robot:

```bash
PYTHONPATH="src/rb2301_fp:$PYTHONPATH" python3 -m pytest -q \
  src/rb2301_fp/test/test_moving_obstacle_detection.py
```

They include noisy static scenes, robot translation/rotation, different object
widths, longitudinal motion, oscillation phases/reversals, missing returns,
occlusion, pose mismatch/dropout, and the stop-align-strafe sequence.

The detector is more selective than shape/centroid comparisons, but cannot
prove motion under arbitrary sensor or localization errors. Sparse background,
heavy occlusion, slow movement below thresholds, close interacting objects,
or turns/reversals within the observation window can prevent confirmation.
Accurate mounting calibration and synchronized measured poses remain necessary.
Synthetic checks do not establish a live Gazebo or hardware success rate.
