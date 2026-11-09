# Stages 1 and 2: stationary preview

With no movement flag, `obstacle_course.py` runs this stationary preview and
publishes **zero velocity**. The orange selected path is a preview, not a movement
command. See [Stage 3](STAGE_3.md) for `--move` and `--step`, which enable movement.

## What is implemented

The template still converts OptiTrack to the maze frame on hardware and reads
Gazebo odometry in simulation. That pose conversion has not been rewritten.
Navigation then calls the provided `to_course()` before saving the starting
pose or calculating relative position. The same course heading is used for wall
measurement, goal bearings and the viewer. The original template pose stays intact.
In simulation, `to_course()` keeps the existing odometry pose.

1. Wait for a pose and scan. Convert the scan using its angle metadata and LiDAR
   mounting correction. Keep the full scan; do not discard every other return.
2. Find returns within 5 degrees of **fixed course-right (-course y)**, using
   the course heading. Their median rightward projection is the initial wall distance.
   This assumes those returns are the intended wall; verify it in the viewer.
3. Save that distance and the starting pose together, once.
4. Compute relative x/y by subtracting the starting course position from the
   current course position. The axes do not rotate with the robot.
5. Start in A with fixed course goal **-45 degrees**. Use B, goal **0 degrees**,
   when `rightward >= switch_distance_multiplier * initial_right_distance`.
   Returning below that same boundary restores A and its -45-degree goal.
   Here `rightward = -relative_course_y`; forward travel does not trigger the switch.
6. Generate front-facing candidates every **3 degrees**. Omit directions without
   a valid nearby scan return. Check the entire hard footprint along each path.
7. All hard-clear candidates compete. A soft-clear candidate gets a small bonus;
   it does not exclude narrow hard-only openings:

   ```text
   score = depth_weight * depth / depth_cap
           + virtual_goal_weight * cos(candidate - local_goal)
           + (soft_clearance_bonus if soft-clear, otherwise zero)
   local_goal = wrap(course_goal - current_course_heading)
   ```

8. Among scores within `similar_score_tolerance` of the highest eligible score,
   select greater depth. Publish the decision for display; keep the motors stopped.

The virtual heading is a scoring preference. The selected candidate can point
elsewhere because of obstacles, clearance preference or a greater depth score.

All settings are together in `PARAMS` near the top of the controller. The
current tuning is lookahead 0.40 m, hard radius 0.11 m, soft radius 0.20 m,
depth cap 3 m, depth weight 1.5, goal weight 1.0, and score tolerance 0.03.

`path_is_clear()` and `path_depth()` retain the earlier straight-path geometry.
The first uses the requested clearance; depth is measured using the hard radius.
Movement is not enabled in this preview mode. There is no visited-point memory,
width score or dead-end recovery; stage 3 adds the turn-and-advance cycle.

## Run

From the workspace root, with your usual ROS environment and sensor publishers
running, stop the previous controller and run this source file directly:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py
```

The template chooses simulation or hardware from the scan length. Use the normal
invocation above for the stage preview; `--calibrate` retains the template's
separate raw-pose calibration display. To use `./fp.sh` instead, first rebuild
and source the package so that its installed entry point uses this code.

In another terminal, for Gazebo:

```bash
python3 visualization/live_map.py --simulation
```

For laptop rendering from robot 14, for example:

```bash
python3 visualization/live_map.py --robot 14 --connect bingda@192.168.1.214
```

The controller runs on the robot alongside OptiTrack/VRPN and LiDAR; the SSH viewer
only receives data. The viewer uses the controller's course-frame snapshot rather
than performing a second pose conversion.

## What to verify, in order

| Test | Expected result |
| --- | --- |
| Start stationary with the intended wall on the fixed course-right side. | Relative position becomes (0, 0). The measured right-wall distance looks correct. The vertical switch line is twice that distance to the right. |
| Rotate the robot without translating. | Relative x/y stay approximately zero. The purple arrow in the **fixed course** inset stays at -45 degrees. The robot-relative purple arrow changes bearing. |
| Move forward (+course x). | Plot y increases; forward movement alone does not trigger the switch. |
| Move rightward (-course y). | Plot x increases, regardless of which way the robot faces. Right-wall distance and the vertical switch line stay fixed. |
| Move leftward. | Plot x decreases; the positive rightward threshold does not trigger. |
| Cross the switch line. | A becomes B and the fixed course goal becomes 0 degrees, pointing up in the inset. |
| Move back across the line. | B becomes A and the virtual goal returns to -45 degrees. |
| Inspect candidate lines and the score table. | Candidates span -90 to +90 degrees locally at the configured step. The table marks the selected candidate with `*`. |
| Put an obstacle in a proposed path. | Its candidate turns red and cannot be selected. Green means soft-clear; amber means hard-only. |
| Compare close scores. | Soft-clear paths receive a 0.15 bonus. Hard-only paths remain eligible. Among final scores within 0.03, greater depth wins. |
| Observe commands throughout. | All commanded linear and angular velocities remain zero. Move the robot manually or reposition it in Gazebo to test position changes. |

The fixed-course inset labels horizontal **x as rightward** and vertical **y as
forward/up**. These display components are `(-relative_course_y, relative_course_x)`.
The template's course coordinates stay unchanged. The inset displays 0 degrees
upward and -90 degrees rightward, so a -45-degree goal points upper-right.

The fixed-course inset's vertical wall line is the **initial right-wall reference**,
not a map of the whole wall. Its vertical green line is the lateral switching threshold.
For example, initial right distance 0.30 m gives a switch at rightward x=0.60 m;
forward y can take any value. `switch_x` names this display/lateral x threshold,
not the template's forward course x component. Purple is the fixed
virtual heading; blue is the measured robot heading. On the large LiDAR view,
the robot always faces up; the orange candidate ends at the lookahead distance.

The two clearance circles surround the robot: red is hard, gold is soft. Soft
clearance is preferred space, not an additional footprint-stop rule.

## Offline checks and example plots

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="src/rb2301_fp:$PYTHONPATH" /usr/bin/python3 -m pytest -q \
  src/rb2301_fp/test/test_stage_preview.py
python3 visualization/live_map.py --demo --demo-segment A
python3 visualization/live_map.py --demo --demo-segment B
```

The examples are saved controller outputs from synthetic room scans, not a
Gazebo or hardware run. Tests call controller methods without starting ROS nodes
or publishing commands. Earlier tests for the superseded controllers do not
describe this new stage-by-stage design.

## Movement and later recovery

Stage 3 advances half the lookahead before selecting again. Lookahead is how far
a candidate is checked; the advance target is calculated as half that distance.

To compare a candidate with a failed direction, compare their **course headings**:
`abs(wrap_deg(candidate_heading - failed_heading))`. For example, 50 and 56
degrees differ by 6 degrees; 179 and -179 differ by 2 degrees. Combine that with
proximity to the failure position. Neither failure memory nor recovery is
implemented yet.
