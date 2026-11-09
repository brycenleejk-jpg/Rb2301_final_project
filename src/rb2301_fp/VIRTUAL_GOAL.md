# Virtual goal navigation

**Historical controller notes.** The active controller has been restarted from
the course template. See [Stages 1 and 2](STAGES_1_2.md) for its current stationary
preview, fixed maze goals, position-based switch and manual verification steps.
The controller described below is the earlier implementation.

The active `obstacle_course` controller uses LiDAR depth and an angular route
preference. It reads OptiTrack orientation on hardware, or odometry orientation
in simulation. It reads no robot position, goal position, course calibration,
or global transforms. The previous controller remains in
`rb2301_fp/backup.py`.

## Segment sequence

1. Define the first valid measured orientation as **0°**. The segment 1 goal
   is fixed at **-45° (right) relative to that original starting heading**.
   It is a scoring preference, not an unconditional turn command. Convert it
   to a robot-relative bearing by subtracting measured heading: after turning
   right 30° the bearing is -15°; after right 45° it is 0°.
2. Accumulate signed, measured yaw changes relative to the first orientation
   sample. Right increases the total; left subtracts. Commands never count.
3. At the configured `segment_right_turn_deg` (**90° net right** currently),
   require three consecutive fresh orientation samples at or beyond the
   threshold, then switch once to segment 2.
4. The segment 2 goal is fixed at **0° (the original forward direction)**,
   not the heading at the switch. Its robot-relative
   bearing is also `wrap(goal_from_start - measured_heading_from_start)`.
   For example, after a 90° right turn (measured heading -90°), the new goal
   is 90° left of the robot. Once it faces the original forward direction
   again (measured heading 0°), the goal is straight ahead (bearing 0°).
   The segment never reverts, even as the robot turns left.

The initial absolute heading is irrelevant. Signed yaw differences wrap at
+/-180°. Turning to avoid an obstacle contributes to the total, so reaching
the turn threshold does not establish that the robot has reached the intended
course corner. On an unobstructed floor the fixed -45° goal can be reached
without reaching the 90° switch threshold, so segment 1 may remain active.
This sequence assumes the known course order; it is not a general maze solver.
Restarting starts segment 1 again. There is no automatic goal arrival test or
final heading alignment. Stop externally with Ctrl+C at the finish.

## Scoring and consistency

Reject blocked candidate paths using the existing geometry checks, then score:

```
score = 1.5 * (path_depth / 3.0) + 1.5 * cos(candidate_bearing - virtual_bearing)
```

Angles in the cosine are radians; path depth is capped at 3 m as before. Both
weights are in `PARAMS`. Setting `virtual_goal_weight` to 0 disables the angular
preference. This score measures depth and direction, not distance progress.
The side preference is soft: an obstructed left target never permits collision.
The search honours `view_half_angle_deg`, limited to 0–180°. Its current value
of 135° searches a total of 270° and excludes the rear 90°. There is no hidden
90° cap. The search includes the measured ray nearest the virtual bearing,
even if regular candidate sampling would skip it. Directions behind either
side require turning in place first, with the existing footprint check.
This sector excludes directions relative to the current robot heading; it
does not identify or exclude previously visited paths.

Selection retains the previous direction only while its score is within 0.03
of the highest score, including during an in-place alignment. It must remain
supported by nearby scan readings and its exact path must still be clear.
The remembered bearing compensates for robot rotation. A blocked, unsupported,
or clearly lower-scoring path is reconsidered immediately; memory also clears
on any stop or segment switch. This reduces turn reversals from small score changes without disabling
collision checks. Straight/curved collision geometry, the minimum robot radius,
preferred clearance, and reduced-clearance speed are unchanged.

## Orientation sources

Hardware defaults to **OptiTrack orientation** from
`/vrpn_mocap/bingda_NNN/pose`. NNN comes from `ROS_DOMAIN_ID` (014 for robot 14;
fallback 003 if the domain is not a robot number). Start the lab's VRPN client
and ensure the tracked body's orientation arrives continuously.

Only `pose.orientation` is used. Position fields are ignored, and there is no
need to load `optitrack_variables.config`: a constant heading-frame offset
cancels when calculating signed yaw differences. The goal remains an angular
preference without global goal coordinates. IMU input is no longer supported.

```bash
./fp.sh --hardware --calibrate
./fp.sh --hardware
```

Stop the monitor before driving. If the tracked body uses a different topic,
remap the default topic, for example on robot 14:

```bash
./fp.sh --hardware --ros-args -r /vrpn_mocap/bingda_014/pose:=/vrpn_mocap/my_robot/pose
```

Simulation continues to read orientation from `/odom`, which is supplied by the
Gazebo odometry publisher. To explicitly choose OptiTrack in either mode, add
`--heading-source optitrack`. `--heading-source odom` remains available.

A missing/stale scan or orientation stops motion after 0.5 s. Nonzero duplicate
orientation timestamps do not refresh freshness or confirm a switch. Unstamped
messages use arrival time. Fresh orientation readings resume turn tracking after
a dropout without requiring a restart. The controller no longer checks backwards
timestamps, quaternion validity, orientation gaps, or implausible yaw jumps.

## Build, monitor, and drive

For the installed simulation environment:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select rb2301_fp --symlink-install
./fp.sh --simulation --calibrate
```

Run Gazebo separately. Monitor mode publishes zero velocity and logs the segment,
virtual bearing and net right turn. Stop the monitor before running:

```bash
./fp.sh --simulation
```

On the physical robot, source its installed ROS environment and workspace, start
base control, LiDAR and the chosen local orientation source, then use
`./fp.sh --hardware --calibrate` followed by `./fp.sh --hardware` after stopping
the monitor. Hardware velocity limits remain unchanged. Omitting both mode flags
retains the old scan-length detector; explicit flags avoid that heuristic.

The viewer displays a purple virtual-goal arrow on the LiDAR plot and a
forward-up aiming compass beside it. Purple is the virtual goal; dashed orange
is the selected path. A readout shows both bearings, the segment and measured
net right turn. Compass arrow lengths are display cues, not goal distances.
Missing/stale updates clear the corresponding arrow. The existing carrot and
scored-path overlays remain available. Restart the viewer after updating it;
restart the controller when its source changes.

## Verification and practical limits

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="$PWD/src/rb2301_fp:$PYTHONPATH" /usr/bin/python3 -m pytest \
  src/rb2301_fp/test/test_direction_selection.py \
  src/rb2301_fp/test/test_virtual_goal.py -q
```

The tests exercise real controller methods and ROS messages without creating
ROS nodes or publishing robot commands. They cover fixed goals in the starting
frame through rotation and wraparound, signed turns, switch confirmation,
wide-sector selection and turn commitment, blocked paths, stale sensors and recovery, and
noisy closed-loop open-space steering in both segments. Synthetic scans are not a
Gazebo or hardware course trial. Door acquisition and complete-course success
must still be measured across the course layouts and actual sensor conditions.
