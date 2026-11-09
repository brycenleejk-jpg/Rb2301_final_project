# Live plot and case logger

The [visited-overlap preview](../src/rb2301_fp/VISITED_OVERLAP_TRIAL.md) adds an
independent `visited_check_start_m` (0.20 m) and `visited_check_distance_m` (0.80 m),
visited percentages and click-to-inspect samples. It clips at obstructions and keeps travelled
history after recovery trims the green trail. The current trial subtracts
`visited_penalty_weight * visited_fraction` at new heading selections, with
weight 2.2 initially (0 disables the penalty). Compare the controlled synthetic
cases with `python3 visualization/live_map.py --demo --demo-visited penalty-off`
and `--demo-visited penalty-on`; neither command connects to ROS or moves a robot.
Percentages stay live during committed movement and recovery, and remain visible
after `--step` stops. Their calculation does not reselect the held heading.

## Current stage 1 and 2 preview

By default the controller commands zero velocity and sends its calculations to
this viewer. Follow [Stages 1 and 2](../src/rb2301_fp/STAGES_1_2.md) for the preview,
or [Stage 3](../src/rb2301_fp/STAGE_3.md) for `--move`/`--step` movement tests.
[Stage 4](../src/rb2301_fp/STAGE_4.md) adds an ordered trail: green dots after
at least 5 cm displacement, with dotted circles of 10 cm radius in the fixed
course inset. The circles are visual guides in normal navigation and the bounded
retreat tests. The optional [turn-around recovery](../src/rb2301_fp/STAGE_7.md)
uses their radius as the width of directional red-route memory.
Turning adds no dots, and the inset keeps the whole recorded trail visible.
The main LiDAR view also shows nearby green trail dots, transformed into the
robot's current frame; dotted circles stay in the fixed-course inset.
[Stages 5 and 6a](../src/rb2301_fp/STAGES_5_6A.md) show WAIT_BLOCKED/BLOCKED and
the `--backtrack-one` test: an orange target, remaining retreat distance and
red reached dots/circles. The test reverses to one dot, then holds a stop.
[Stage 6b](../src/rb2301_fp/STAGE_6B.md) adds `--backtrack-dots 3`: the viewer
shows `MULTI-DOT TEST`, the current orange target and reached progress such as
`1 / 3`. Each reached dot/circle turns red; completion holds zero velocity.
[Stage 6c](../src/rb2301_fp/STAGE_6C.md) adds `--backtrack-curves 3`. Its readout
shows curved retreat or rear alignment, signed steering commands and the checked
reverse arc in teal. The straight test modes remain available.
`--recover` reverses until the blue turning envelope fits, turns toward the
older route and resumes the current front-240-degree scoring. Red route history persists
while new movement records green dots again. Grey rays are physical-clear paths
excluded for returning toward a remembered failed end. During `TURN_AROUND`,
the orange target is a heading rather than a translation carrot.
The turn panel reports whether the turning-space recheck is ON or OFF. The
initial check always runs; the current default skips space rechecks after the
turn starts, while missing/stale sensors and the timeout still stop movement.
During movement the viewer shows TURN/CURVE/DRIVE/DONE, a fixed orange committed
heading, measured half-lookahead progress and the completed-advance count. The
orange line length is the remaining advance distance; scores are paused during
the attempt. A teal line shows the commanded robot-centre path over the short
motion prediction. The movement readout includes the decision reason and linear/
angular speed. The no-flag preview retains the candidate score table.

The large view keeps the robot's nose pointing up. Green candidates fit soft
clearance, amber candidates fit only hard clearance, and red candidates are
blocked. All green and amber candidates compete; green receives the configured
`soft_clearance_bonus`. The [sideways-clearance trial](../src/rb2301_fp/SIDE_CLEARANCE_TRIAL.md)
adds a gradual reward up to `side_clearance_weight` between the hard and soft
radii. Orange marks the selected preview path; purple marks the virtual goal.
The score table shows the selected candidate (`*`), score, depth in metres and
capped perpendicular side clearance in centimetres (`Side(cm)`). The two
circles show hard (red) and soft (gold) clearance.

The **fixed course inset** labels x as rightward and y as forward/up. It keeps
course-forward (0 degrees) pointing up and course-right (-90 degrees) pointing
right. Display x is `-relative_course_y`; display y is `relative_course_x`.
This is a fixed display rotation;
it does not rotate with the robot or change navigation coordinates. It shows relative position,
measured robot heading, the fixed virtual heading, initial right-wall reference
and a **vertical lateral switch line**. The switch uses rightward displacement,
not forward movement. Returning across it from B to A restores A's virtual goal,
including during retreat. A committed motion or recovery turn retains its saved
target until completion. The readout shows segment A/B, course heading, display x/y,
initial right distance, switch x and the selected path's course heading. The
controller calls the template's `to_course()` before saving or subtracting poses.
Current goals are fixed course -45 degrees for A and 0 degrees for B.

```bash
python3 visualization/live_map.py --simulation
# Or render on the laptop from a robot, for example:
python3 visualization/live_map.py --robot 14 --connect bingda@192.168.1.214
# Synthetic controller snapshots, without ROS or a GUI:
python3 visualization/live_map.py --demo --demo-segment A
python3 visualization/live_map.py --demo --demo-segment B
```

Restart the viewer after updating it. The remaining notes describe earlier
controllers and viewer features; the stage guide above is the current reference.

## Earlier aiming display

The current viewer includes a **robot-relative aiming compass** on the right,
with the robot facing upward. Purple shows the angular virtual goal; dashed
orange shows the controller's selected path. Their lengths are display cues,
not distances. The readout gives both bearings as left/right angles, the active
segment, and measured net right turn. If an update is missing or stale, its
arrow disappears and the readout says `none / stale`.

The main LiDAR plot also keeps the robot's front pointing up, with left and
right matching the screen. Scans, the virtual goal, the selected carrot, and
scored paths share the same display rotation. The LiDAR mounting correction
is separate from this display rotation. The compass does not change navigation.
Restart the viewer to load this display. Run it with
`python3 visualization/live_map.py --simulation`, or from the laptop with
`python3 visualization/live_map.py --robot 14 --connect bingda@192.168.1.214`.
Use `--demo` to render synthetic example data without ROS.

See [the virtual-goal guide](../src/rb2301_fp/VIRTUAL_GOAL.md) for current
controller behavior. The older setup and case-logger notes below describe an
earlier viewer; this script does not support all of those options.


The viewer shows a LiDAR map, the robot's chosen carrot, and the openings and
depths reported by the controller. The plot has no side panels or score charts.
It never publishes driving commands. Both programs use Python.

## Real robot: use Foxy for both programs

Run the controller and viewer on robot 14 under ROS 2 Foxy and
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. Display the plot on your laptop using
SSH X11 forwarding. The laptop does not run a ROS viewer in this arrangement.
The viewer checks the ROS version, middleware and robot domain before starting.
Full copy, build and run instructions are in
[real_robot_commands.txt](../real_robot_commands.txt).

After copying the project and building on the robot, open a laptop terminal:

```bash
ssh -X bingda@192.168.1.214
```

Inside that SSH session:

```bash
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=14
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd ~/Downloads/rb2301_fp_10
source install/setup.bash
python3 visualization/live_map.py --robot 14 --lidar-yaw-deg 180 --plot-hz 2
```

Keep base control, LiDAR and the controller in their own Foxy robot terminals.
First monitor with `./fp.sh --calibrate`, then use `./fp.sh` to navigate.
The local viewer does not require Optitrack or `vrpn`.

The default hardware view is **relative to the robot**. It shows current scans
and the chosen carrot without a pose. The robot stays at the display origin;
the viewer does not infer motion from `/cmd_vel` or invent a travelled route.
Cases are recorded with timestamps and no position.

For an accumulated map in fixed coordinates, keep the `vrpn` client running and use:

```bash
python3 visualization/live_map.py --robot 14 --pose-source optitrack --lidar-yaw-deg 180 --plot-hz 2
```

This subscribes to `/vrpn_mocap/bingda_014/pose`. The orange star is the logged
carrot, and the dashed orange line points
from the logged robot position towards that carrot. It requires arriving
Optitrack poses and controller logs; the viewer does not start those services.

If the robot publishes valid wheel odometry, use it for map coordinates with:

```bash
python3 visualization/live_map.py --robot 14 --pose-source odom --lidar-yaw-deg 180
```

This reads `/odom` pose and twist. The map requires arriving odometry in this
mode. Odometry is optional for the viewer and is not required by navigation.

SSH X11 forwarding and a graphical Matplotlib backend must be available. Check
`echo "$DISPLAY"` inside the SSH session if a window does not appear. Logs are
saved on the robot; the command file includes rsync commands to copy them back.
These changes have not been tested on the physical robot.

The viewer's hardware LiDAR yaw defaults to 0 degrees. The commands above use
180 degrees to match the current robot 14 controller. The x offset defaults to
0.11423 m. These are not measured hardware dimensions. Use
`--lidar-x`, `--lidar-y` and `--lidar-yaw-deg` after checking the mount, and make
navigation use the same transform.

## Simulation on the laptop

Gazebo uses the existing Jazzy installation. Keep it on a separate ROS domain
from the real robot. Run Gazebo and `./fp.sh --simulation`, then open another
laptop terminal in the workspace:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 visualization/live_map.py --simulation
```

Simulation uses `/odom` for position and velocity and the simulated LiDAR's
180-degree mounting rotation. The viewer needs numpy, matplotlib and the ROS
Python packages used by the project.

## What the plot shows

- Dark dots: latest LiDAR hits. Invalid readings and positive infinity are omitted.
- Light grey dots: accumulated LiDAR map when position data is available.
- Orange star and dashed line: the controller's current logged carrot and aim.
  They disappear after 2.5 seconds without an update, or on `Carrot: none`.
- Blue point and arrow: the robot's position and heading.
- Green region: the chosen opening; blue regions: other clear candidates;
  red regions: blocked candidates. These are logged angular regions.
- `R1  24.0°  D=1.20 m`: region ID, angular opening and the controller's
  representative depth. Up to six labels are shown, prioritising the chosen
  region and higher scores. Regions are clipped to the nearby view while depth
  labels retain the reported value. `(max)` marks the scan range limit, which
  may come from a no-return reading rather than a measured obstacle.
- Purple edge line and `W≈0.45 m`: estimated physical opening width, shown only
  when the controller reports an entry and the scan provides two usable edges.
  This estimate is separate from the controller's logged angular opening.

The same artists are updated in place to keep drawing light over X11. There are
no legends, score charts or past-target trails. Opening overlays disappear when
their controller logs are more than 2.5 seconds old.

In the local hardware view, the origin is a display reference. No route, past
scan map or past target trail is accumulated in that moving frame.

## Recorded data

| Source | Values |
| --- | --- |
| `/scan` | Ranges, angles and valid range limits |
| `/cmd_vel` | Commanded forward speed and yaw rate |
| `/rosout`, logger `obstaclecourse` | Carrot, region bounds, depth, target distance, score and clearance reason |
| `/odom` when enabled | Position, quaternion converted to yaw, and measured twist |

Commands, scores and observed states remain in the run logs for later analysis.
Source settings are read without executing the saved controller snapshot.

ROS messages are received continuously on a separate thread. The GUI consumes
the latest sensor and command samples rather than replaying an old backlog.
Message ages use their arrival times, including when drawing is slow.

The plot and trace default to 2 Hz on hardware and 5 Hz in simulation. Adjust
`--plot-hz` to change the redraw limit; try `--plot-hz 1` for a slow X11
connection. The terminal prints received-message totals every five seconds for
scans, commands, poses and controller logs. Increasing totals confirm reception
even when drawing is slow. A zero pose count is expected in local mode.

Carrots and regions arrive in controller logs at about 1 Hz with rounded values.
Messages arrive asynchronously, so the plot is an approximate diagnostic view.
The accumulated scan trace with position data is not SLAM; pose errors and
moving obstacles affect it.

## Mark and save cases

Click the plot to give it keyboard focus:

| Key | Action |
| --- | --- |
| `1` | Mark a stop in a gap |
| `2` | Mark an unwanted U-turn |
| `m` | Mark another case |
| `f` | Switch between whole-map and following views |
| `q` | Save and close the viewer |

Case marks are saved in the event log. Closing the viewer does not stop the controller.

Each run creates a dated folder under `visualization/runs/`:

| File | Contents |
| --- | --- |
| `map.png` | Final plot, saved on normal exit |
| `trace.csv` | Commands, states, targets and available pose/motion measurements |
| `events.csv` | Case marks and state changes |
| `frames.jsonl` | Display snapshots, regions, scores and sensor ages |
| `controller.log` | Received controller logs |
| `controller_snapshot.py` | Saved source copied at viewer startup |
| `viewer_settings.json` | ROS version, middleware, pose source and mounting settings |

Missing measurements are blank in CSV and `null` in JSON. Local scan points
use `scan_points_robot`; odometry-transformed points use `scan_points_map`.
The CSV's `carrot_body_*` columns are robot-relative; `carrot_x_m` and
`carrot_y_m` require odometry and remain blank in local mode. Restart the
controller and viewer after changing saved code so the snapshot matches the run.

## Preview without ROS

```bash
python3 visualization/live_map.py --demo
```

This saves an illustrative map and carrot with example logs in a separate
`demo-...` folder. It is not evidence from a robot run.
