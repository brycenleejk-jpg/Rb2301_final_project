# Stages 5 and 6a: confirm front blockage, retreat to one dot, stop

This first recovery test targets a U-shaped dead end with a clear footprint
and an earlier breadcrumb approximately straight behind the robot.
[Stage 6b](STAGE_6B.md) extends the checked movement to several straight dots;
the `--backtrack-one` test described here still stops after one.
The existing front-180-degree scorer is reused. `selected_index is None` is
the blockage condition; there is no separate dead-end geometry detector.

## State flow

```text
Normal movement -> no candidate -> WAIT_BLOCKED (stop immediately)
                                   |
                       repeated fresh scans for 0.5 s
                                   |
                  --move: BLOCKED, hold stop
         --backtrack-one: RETREAT -> RETREAT_DONE, hold stop
```

A candidate appearing before confirmation resets the timer and resumes normal
selection. Once a blockage is confirmed in detection-only mode, the test holds
its stop until the controller restarts. A single blocked committed path first
returns to normal selection; it does not itself request retreat.

Missing/stale data resets pending confirmation. No supported front rays is a
scan-coverage problem, not confirmed blockage. A point already inside the hard
footprint produces a separate held stop, without automatic reverse. These
distinctions prevent sensor loss or existing contact from triggering a retreat.

## One-dot retreat

`--backtrack-one` first drives normally and records its trail. After confirmed
front blockage:

1. Search the trail backwards in chronological order. Skip points within the
   1 cm arrival tolerance and choose the latest earlier point outside it.
   Do not choose the globally nearest historical point.
2. Require that target to be behind the robot, within 1 cm of its straight
   reverse centre line. This stage does not steer through bends or turn around.
3. Require a valid rear-facing scan ray. Check the actual reverse segment to
   the target's projection with `path_is_clear(points, pi, distance)` and the
   hard radius. The larger soft radius is not required for this retreat.
4. Reverse at 0.05 m/s with zero angular velocity. Recheck the target and rear
   sweep on each controller update. Do not append new green trail dots.
5. Within 1 cm of the target, mark its dot/circle red and enter RETREAT_DONE.
   Hold zero velocity; do not choose another dot or a new forward direction.

A rear obstacle, lost data, an off-line target, overshoot or a four-second
timeout stops and holds the test with a reason. Completed tests also remain
stopped after sensor pauses. Restarting is required to run another test.

| Parameter | Initial value | Meaning |
| --- | --- | --- |
| `blockage_confirm_s` | 0.5 s | Persistent no-candidate confirmation time |
| `retreat_speed_mps` | 0.05 m/s | Straight reverse speed |
| `retreat_reach_m` | 0.01 m | Target arrival tolerance and permitted lateral offset |
| `retreat_timeout_s` | 4 s | Maximum duration of the one-dot attempt |

The 10 cm dotted trail circles remain display guides. They are not the 1 cm
arrival tolerance. Position/direction memory, visited-departure rejection,
360-degree recovery scoring and multi-dot retreat are later stages. Red dots
currently show reached retreat targets; they do not yet reject future paths.

## Test in Gazebo

Stop the old controller before starting a new mode. Begin each run before the
dead end so it can record a trail. Restarting at the dead end creates only a
start dot and leaves no earlier target to reverse toward.

First, test detection only:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --move
```

Expected: no front candidate -> WAIT_BLOCKED with zero velocity -> after about
0.5 s, BLOCKED with `front blockage confirmed`. There must be no reverse command.

Then reset to the corridor approach and test one-dot retreat:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --backtrack-one
```

Restart the viewer in another terminal:

```bash
python3 visualization/live_map.py --simulation
```

Expected: WAIT_BLOCKED -> RETREAT, negative linear velocity and zero angular
velocity -> RETREAT_DONE. The retreat is only a few centimetres. Orange shows
the target dot in the LiDAR view and fixed inset; a reached target turns red.
The readout shows target index, remaining distance and command speeds.

If a footprint violation is reported, this test deliberately stays stopped.
Reposition the robot before testing the ordinary U-dead-end case. Escaping
an already-violated hard footprint needs a separate recovery rule.

## Automated verification

```bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH="src/rb2301_fp:$PYTHONPATH" /usr/bin/python3 -m pytest -q \
  src/rb2301_fp/test/test_stage_preview.py
```

Checks include a closed-loop synthetic U corridor, transient blockage, target
order, rear scan coverage, reverse clearance, new obstacles, sensor failures,
timeout, stopping and the red/orange viewer overlays. These are synthetic tests,
not a live Gazebo or hardware trial.

Controller-generated previews: [one-dot retreat](../../visualization/previews/retreat_one_dot.png)
and [reached target](../../visualization/previews/retreat_done.png).
