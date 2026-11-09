# Visited overlap and score penalty trial

The current trial subtracts a visited penalty **when selecting a new heading**.
Clearance and red-return rejection still apply first. Held movement and recovery
do not reselect because a live overlap percentage changes.

Settings in `rb2301_fp/obstacle_course.py`:

| Parameter | Initial value | Meaning |
| --- | --- | --- |
| `visited_check_start_m` | `0.20` | Distance from the robot where the comparison starts. |
| `visited_check_distance_m` | `0.80` | Distance from the robot where the comparison ends, unless obstructed. |
| `visited_radius_m` | `0.10` | Distance from a travelled position that counts a sample as visited. |
| `visited_penalty_weight` | `2.20` | Maximum score deduction, reached at 100% visited. Set 0 for the earlier display-only behaviour. |

The start and end are independent. The current window is **0.20 to 0.80 m**;
positions closer than 0.20 m are ignored. Movement lookahead is a separate setting.
Samples are spaced at most 1 cm apart. Each sample counts once if any remembered
position is within the visited radius. Repeated dots at the same location do
not multiply the percentage.

```text
visited percentage = 100 × visited samples / checked samples
final score = depth + goal + soft bonus + side reward − weight × visited fraction
```

The end is shortened to the first measured hard-circle contact. Positions past
that end are excluded from the comparison. If the obstruction is before the
0.20 m start, show **N/A**, not 0%: there is no comparison window. N/A adds no
visited penalty because there is no measured overlap evidence. If it ends
partway through, the percentage describes only the remaining checked portion.
The readout shows the shortened end, so a short window is visible.

The percentage is the fraction of this usable candidate window covered by
travelled-position circles, not the number of old dots or the percentage of a
whole room explored. Half the checked samples near history means 50% visited.
When an obstruction shortens the window, only samples before that end form the
denominator; it can still be 100% even when less than 0.60 m was checked.

This uses the current scan, not an occupancy map. It does not identify complete
dead ends, infer unseen space, or establish whether two nearby routes connect.
All-visited directions remain usable; visits change scores, not eligibility.
When all fractions are equal, the same deduction leaves their ranking unchanged.

With goal weight 1.0, the goal contribution ranges from -1 to +1: a full
opposite-direction difference of 2.0. Penalty weight 2.2 can overcome that
difference when one path is 100% visited and the other 0%, with equal remaining
rewards. Different depth and clearance can still outweigh it; this is a soft
preference rather than a guarantee that every unvisited direction wins.

`visited_positions` is a separate history of fixed course-frame positions from
forward and reverse travel, recorded at the existing trail spacing. Pure turns
add no positions. Recovery can trim `trail` for backtracking without erasing
this history. Restarting the controller clears it. Display coordinates subtract
the starting position, as with the existing green and red histories.

## First tests: saved synthetic scenes, no ROS needed

Compare `penalty-off` and `penalty-on` to see the same controlled two-path scene
with weight 0 and 2.2. Both paths have equal depth and clearance. With weight 0,
the visited -60° goal direction wins. With weight 2.2, the unvisited +120° path
opposite the goal wins. The overlap distance in these two scenes is 0.40 m.

```bash
python3 visualization/live_map.py --demo --demo-visited penalty-off
python3 visualization/live_map.py --demo --demo-visited penalty-on
```

The five earlier overlap-only scenes below retain their saved 0.40 m distance
and disabled penalty, for verifying the measurement separately from scoring.

From the project root:

```bash
python3 visualization/live_map.py --demo --demo-visited right-travelled
```

Open `visualization/runs/stage-preview-right-travelled/map.png`.
Change `right-travelled` to another case below. These snapshots were generated
by the controller's calculation with synthetic scans and mocked command output;
they are not recordings of robot runs.

| Case | Expected |
| --- | --- |
| `right-travelled` | -45° (right) is 100% visited; +45° (left) is 0%. Existing selection is unchanged. |
| `nearby-dots` | Dots only within 8 cm of the robot leave the second-half forward comparison at 0%. |
| `wall` | Wall at 0.35 m limits centre travel to 0.24 m with hard radius 0.11 m. History beyond that end is ignored. |
| `short-window` | Wall at 0.28 m limits travel to 0.17 m: N/A, since the comparison starts at 0.20 m. |
| `all-visited` | Clear candidates can show 100% visited and still remain eligible. |

## Live visualisation

Restart the controller and viewer. Use the controller's default stationary
preview first, and the usual viewer:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py
python3 visualization/live_map.py --simulation
```

A fresh stationary run has only the initial history position, so most readings
will be 0%. The synthetic scenes above test meaningful stored routes without
driving. Normal `--move`/`--recover` runs collect travelled history and apply
the configured penalty at their next heading selection.

The overlap box compares fixed local probes: +90°, +45°, 0°, -45°, -90° (when
supported by the scan). Click a candidate in the main plot to inspect any other
sampled direction. Disable toolbar pan/zoom first. The click affects the viewer
only and sends no robot command.

- Cyan samples: not near a remembered position.
- Violet samples: visited.
- Black x: end of this direction's comparison, including obstruction clipping.
- Pale grey dots: persistent travelled history; existing green/red trails keep
  their original meanings.

Check that moving or rotating the robot preserves the history's fixed location.
Compare percentages, checked distances, candidate scores and the logged penalty.
Use weight 0 to compare behaviour with the previous stage.

Percentages now update during TURN/CURVE/DRIVE, recovery and the stopped DONE
state. This uses a separate display calculation, so a held heading is never
reselected. Without a click, the overlay inspects the held heading's nearest
supported candidate; clicking a ray changes the inspected direction only.

For a short live test, use `--step` instead of `--recover`:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --step
```

With the current 0.40 m movement lookahead it advances 0.20 m, then stays stopped
with the overlap display still updating. This is one advance per controller run,
not an interactive stepping mode. Restarting clears the histories, so use the
normal continuous movement/recovery mode to inspect a longer accumulated route.

## Occasional reverse-curve stop

The hard-clearance rule remains unchanged. On a recurrence the stop now includes
the target dot, target distance, nearest sampled obstacle distance, hard radius,
sampling padding and command. For a curve the required sampled separation is
`hard_radius + padding`. This helps distinguish a tight arc from the unrelated
turn-around envelope. A stopped recovery still holds until restart.
