# Turn-around recovery with directional red history

The controller reverses out of a blockage, finds space for a turn, faces the
older route and resumes the current front-240-degree scoring. The old full-circle
departure preview and comparison against a backtracking score have been removed.

Stop the previous controller. Start before the obstacle so this run records its
own trail:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --recover
```

Restart the updated viewer:

```bash
python3 visualization/live_map.py --simulation
```

`--step`, `--move` and the bounded `--backtrack-*` tests remain available.
`--move` still holds after confirmed blockage; `--recover` enables automatic
reverse/turn/resume cycles. `--recovery-preview` is no longer a mode.

## Flow

```text
SELECT -> TURN/CURVE/DRIVE
  no candidate for blockage_confirm_s
    -> RETREAT through older dots, recording red movement
    -> TURN_AROUND when its enclosing circle fits
    -> SELECT using front 240 degrees, recording green movement again
```

At each reached reverse dot, `try_turnaround()` checks the turning envelope
against all valid LiDAR obstacle points. If blocked, it continues to the next
older dot. If clear, it saves one course heading toward the next older dot and
`run_turnaround()` rotates in place toward it. At the trail start, it uses the
last reverse segment to determine the outward direction.

The heading remains fixed throughout the turn. The initial turning-space check
always runs. `recheck_turn_space_during_turn=False` disables subsequent space
checks after the turn starts; later reduced clearance will not stop rotation.
Set it to `True` to restore the `turning space became blocked` stop. Fresh
pose/scan checks and the turn timeout remain active in either mode.
If no turning place is found by the trail start, it holds `RETREAT_DONE`.

After alignment, it removes the failed tail from the active breadcrumb list.
The remaining older prefix is retained, and subsequent movement adds green
positions again. Red history is stored separately and survives later recovery
cycles. It is cleared only by restarting the controller.

## How red history becomes a direction restriction

`red_segments` stores pairs of **fixed course-frame positions**:

```text
(older/outward endpoint, endpoint toward the failed end)
```

`record_red_position()` records actual reverse displacement at breadcrumb
spacing and finishes the remaining short segment when a dot is reached. Turning
adds no green or red displacement samples. These coordinates do not move when
the robot moves or rotates.

`path_returns_to_red()` transforms a proposed robot-local path into the course
frame. It rejects travel when BOTH are true:

- The path passes within `trail_radius_m` of a stored red segment.
- Its local movement direction points within `red_return_angle_deg` of that
  segment's direction toward failure.

This checks the sampled path, not only its endpoint. Half the sample spacing
pads the memory radius to cover between samples. A red position is not a physical
obstacle: outward travel and crossings outside the angle window remain allowed.
A coordinate may appear in both red history and a newer green route.

Normal selection scores front-240-degree candidates using the existing
weights. Red-return candidates retain their diagnostic scores but are excluded
from `eligible`. Actual commanded forward motion is checked too, so a moving
curve cannot bypass the memory filter. The forced breadcrumb retreat is checked
for physical clearance rather than forbidden by its own red history.

The viewer receives `red_segments_relative`, subtracting only the fixed startup
position for plotting. Its LiDAR view additionally rotates those fixed positions
into the current robot frame. Navigation stores the original course coordinates.

## Important parameters

| Parameter | Meaning |
| --- | --- |
| `trail_radius_m` | Remembered route width from its centreline, initially 0.10 m |
| `red_return_angle_deg` | Direction window toward the failed end, initially 45 degrees |
| `turn_clearance_m` | Rotation envelope about the current pose origin, initially 0.21 m |
| `recheck_turn_space_during_turn` | False: check space before starting only. True: also stop if space becomes blocked during rotation. |
| `turn_timeout_s` | Maximum duration of one turn, initially 20 s |

The 0.21 m turn radius encloses the Gazebo model around its rearward reference
origin. It is separate from the existing driving radius. A narrow corridor may
allow reversing but offer no place for this conservative turn circle. This
change does not fix or validate the smaller configured forward-driving footprint.
Your existing scoring and driving settings are preserved.

## Viewer and expected logs

- Green dots: active trail, including new movement after turning.
- Red dots/circles: persistent abandoned-route history.
- Grey candidate rays: clear paths excluded for returning toward a red failed end.
- Blue dotted circle: turning envelope, visible during automatic retreat/turn.
- Orange direction during `TURN_AROUND`: saved outward target; no translation carrot.

Typical sequence, with illustrative dot indices:

```text
WAIT_BLOCKED: no clear candidate; confirming front blockage; cause=physical clearance
RETREAT: front blockage confirmed; target trail dot #14; limit=until turning space
TURN SPACE: radius=0.210m blocked; continue to older dot
TURN_AROUND: space clear; target course=+180.0deg radius=0.210m red_segments=13
TURN_AROUND: target course=+180.0deg | error=-90.0deg w=-0.400rad/s; red history retained
RECOVERY COMPLETE: outward heading reached; resume front 240; new trail green
SELECT: ... red_returns_ignored=...
```

If physical paths exist but all are excluded by memory, the blockage log reports
`cause=red-return memory`. Recovery can run again; earlier red segments remain.

## First tests

1. Start with a straight dead end leading back into an open area. Verify red
   recording, continued reverse when the blue circle is blocked, a stationary
   turn and forward movement with new green dots.
2. Verify the old red history remains fixed on the map after the robot rotates.
3. Try a return toward that failed route: candidates should become grey; moving
   outward or across the history should remain possible.
4. Test a second blockage and check that both abandoned sections remain visible.
5. Then test the V-shaped pocket. Red history remembers travelled routes, not
   the entire pocket; an untravelled part can still be explored.

Automated tests use mocked publishers and geometry-derived scans. They cover
reverse/turn/resume, independent wall-clearance checks, repeated recovery memory,
direction filtering, whole-path overlap, faults and map coordinate conversion.
They are not live Gazebo or hardware trials.
