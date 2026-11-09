# Stage 3: hold a heading, steer while moving, choose again after half the lookahead

Default startup still runs the stationary stage 1/2 preview. Movement is enabled
with `--move` for repeating cycles, or `--step` for a single cycle.

The existing course conversion, segment switch, clearance checks and scoring are
reused. [Stage 4](STAGE_4.md) adds trail recording and dotted circles on the fixed
map. [Stages 5 and 6a](STAGES_5_6A.md) now add persistent front-blockage reporting
and an optional one-dot straight retreat test.

## The movement cycle

```text
SELECT -> TURN / CURVE / DRIVE -> SELECT ...
```

1. **SELECT:** score the front-facing candidates using the full lookahead.
   All hard-clear candidates compete; soft-clear paths receive the configured
   `soft_clearance_bonus`. The [sideways-clearance trial](SIDE_CLEARANCE_TRIAL.md)
   also rewards perpendicular side room between the hard and soft radii.
   Convert the winning local angle to a fixed
   course heading and save it. A hard-only doorway can beat a wider alternative.
2. On every update, `choose_motion()` picks a command for that saved heading:
   - **DRIVE:** error within 3 degrees and the remaining straight path is hard-clear.
   - **CURVE:** error above 3 and below 45 degrees, and the predicted moving turn
     satisfies `curve_clearance_m`. Translation slows as error approaches 45
     degrees. Angular speed follows the signed error, capped by `max_turn_rps`.
   - **TURN:** error at least 45 degrees, or the moving curve is blocked. Rotate
     toward the same heading if its remaining straight path is hard-clear.
     Otherwise stop, discard the attempt and retry selection next update.
     In Gazebo only, `sim_min_turn_rps` applies to these in-place turns.
3. Save the current course position on the **first translation**, including a
   moving turn. Keep it through later CURVE, DRIVE and TURN updates. Complete the
   advance when progress along the held heading reaches **half the lookahead**.
4. **SELECT again:** record `ADVANCE COMPLETE`, increment the displayed count,
   and select/check the next command in the **same update**. A clear continuation
   needs no intervening zero command. `--step` instead enters **DONE**, publishing
   zero until the controller restarts.

Progress is measured from `to_course()` positions, projected along the committed
heading. Time, yaw changes and sideways displacement do not count as progress.
Turning in place does not start the progress measurement.

New scans check the actual commanded motion throughout the attempt. A better
score alone cannot change its heading. A blocked curve can switch to TURN without
discarding that heading. Once aligned, hard-clear straight motion is allowed even
if the preferred soft margin is unavailable. This lets the robot leave a tight
space. Interrupted attempts return to SELECT; failed directions are not remembered.

There is no separate rule to drive straight just because it is deep. Straight
ahead competes in the existing direction scores and can win despite the virtual
goal. It still depends on depth, alignment weights and clearance eligibility.

## How the curved clearance check works

`motion_centres()` predicts robot-centre positions for the current constant
command. Local x is forward and y is left. For speed `v`, turn rate `w` and
travel distance `s`, the turning radius is `R = v / w` and the swept angle is
`w * s / v`. Centre positions are `(R * sin(angle), R * (1 - cos(angle)))`.
The turning-circle centre is **not** the point checked against obstacles.

`motion_is_clear()` checks each obstacle's distance from every predicted
**robot-centre position**. Each distance must exceed the chosen footprint radius.
The sample interval is set in `motion_centres()`; adding half the actual sample
spacing to the clearance radius covers the spaces between samples. Straight motion uses the
existing exact `path_is_clear()` instead. In-place rotation checks the current
circular footprint.

`curve_clearance_m` controls only the moving-curve check. It defaults to 0.16 m;
lowering it to 0.15 m reduces the required curve clearance without changing
`soft_clearance_m` or the candidate scores. Both values are radii measured from
the robot centre, not additional margins outside its body. The curve radius is
clamped to at least `robot_radius_m`, and sampling padding is still added.
The 0.01 m sampling spacing adds at most 0.005 m of padding. Curve status messages
show the effective curve radius before that padding.

For CURVE, predict the next `motion_prediction_s` (initially 0.5 seconds), capped
at the remaining advance distance. Recompute every 0.05-second controller update.
This is a short prediction assuming the current command remains constant; it is
not a guarantee that the complete route around an obstacle is open.

The initial lookahead is not demanded again from every new position. For example,
with lookahead 0.60 m the advance target is 0.30 m. After advancing 0.10 m, check
the remaining 0.20 m to finish this attempt. The next SELECT uses a fresh full
lookahead from the new position.

The A/B segment switches on rightward displacement, not forward progress:
`-relative_course_y >= 2 * initial_right_distance`. It keeps updating during movement.
Its new virtual goal affects the
next selection; it does not interrupt an otherwise clear committed attempt.

## Current settings

All settings remain together in `PARAMS` in `rb2301_fp/obstacle_course.py`:

| Setting | Current value |
| --- | --- |
| `check_distance_m` | 0.40 m |
| Advance target, calculated automatically | 0.20 m |
| `forward_speed_mps` | 0.30 m/s |
| `soft_clearance_m` | 0.16 m, heading-selection preference |
| `curve_clearance_m` | 0.16 m, moving-curve check only |
| `soft_clearance_bonus` | 0.15 |
| `max_turn_rps` | 0.40 rad/s |
| `sim_min_turn_rps` | 0.15 rad/s, simulation only |
| `heading_tolerance_deg` | 3 degrees |
| `moving_turn_limit_deg` | 45 degrees |
| `motion_prediction_s` | 0.50 s |
| `sensor_timeout_s` | 0.50 s |

Setting lookahead to 0.60 m automatically changes the advance target to 0.30 m.
Movement stops if pose/scan data is missing, invalid or no longer updating.
Ctrl+C sends a zero command before destroying the node. There is no sensor-fault
latch or requirement to restart after a temporary data interruption.

## Test in Gazebo, one observation at a time

Start Gazebo as usual, source ROS in both terminals, and stop the old controller.

Controller, repeating cycles:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --move
```

Viewer, in another terminal:

```bash
python3 visualization/live_map.py --simulation
```

For one turn-and-advance followed by a held stop, use `--step` instead of `--move`.
Omit both flags to inspect the original stationary preview. `--calibrate` remains
a separate non-driving mode. Restart both scripts after updating their source.

| Observation | What to check |
| --- | --- |
| TURN | Forward velocity is zero. Orange in the fixed-course inset holds one heading while the blue measured heading turns toward it. |
| CURVE | Forward and angular velocities are nonzero. Error reduces while progress increases. A teal line shows predicted robot-centre positions. |
| DRIVE | The displayed progress increases from zero toward 0.200 m. The orange heading stays fixed, even when other space becomes visible. |
| Completion | Progress reaches or slightly exceeds 0.200 m. The terminal prints `ADVANCE COMPLETE: .../0.200m`, and the completed-advance count increases. Only now can a new heading be selected. |
| Repeat | A new clear forward/curved command follows immediately; a new large turn still requires zero forward speed. |
| New obstacle in the remaining segment | The robot stops and reports `committed path blocked`; it is not forced to complete the minimum distance. |
| No clear candidate | The robot stops. Turning around or backing out is a later stage. |

The main orange line now shows **remaining advance distance**, not the initial
lookahead. The aiming compass and fixed-course orange arrows show the committed
direction. Teal shows the actual commanded centre path over the short prediction.
The readout shows phase, movement reason, linear/angular commands, locked heading,
progress and completed count. Logs show SELECT, state changes, blocked-path stops
and ADVANCE COMPLETE, with numerical status updates once a second.

## Automated checks

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="src/rb2301_fp:$PYTHONPATH" /usr/bin/python3 -m pytest -q \
  src/rb2301_fp/test/test_stage_preview.py
```

Tests run without ROS nodes or physical commands. They cover straight-path
selection despite the virtual goal, arc geometry and collisions, blocked-curve
alignment, progress through curved motion, heading commitment, transformed
positions, hard-block interrupts, stale sensors, continuous reselection, the
viewer, two closed-loop advances with a synthetic bin, and crossing a 30 cm
doorway at both 0.10 and 0.30 m/s. Synthetic checks are
not a live Gazebo or hardware course trial.
