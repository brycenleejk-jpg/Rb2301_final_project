# Stage 6c: steer backwards along a bent trail

[Turn-around recovery](STAGE_7.md) adds the separate `--recover` mode to
reverse until there is turning space and then resume front scoring. The bounded
tests described here remain available unchanged.

This opt-in test adds steering to the bounded breadcrumb retreat. The existing
`--backtrack-one` and `--backtrack-dots N` modes keep their straight-only behaviour.
The front-blockage trigger, target order, red reached dots and final held stop
are unchanged.

## Test command

Stop the old controller and begin before the dead end so the new run records
a trail that includes a gentle bend:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --backtrack-curves 3
```

Restart the viewer in another terminal:

```bash
python3 visualization/live_map.py --simulation
```

The controller drives normally until no front candidate persists for 0.5 s.
It then follows up to three earlier dots and holds a stop in RETREAT_DONE.
The viewer shows CURVED RETREAT TEST, the current action and signed command
speeds. Orange marks the target; teal shows the checked reverse arc.

If the last three dots are straight, zero angular velocity is expected.
To test a bend farther back, choose enough dots to reach it, for example
`--backtrack-curves 9`. The controller still stops at the requested count or
the start of the trail. Restarting clears the trail, so do not begin a new
controller while already trapped and expect it to remember the previous run.

## Command choice for the current dot

The saved target is converted into robot-local `local_x, local_y`, just as in
the straight test. Arrival still means within `retreat_reach_m` (1 cm).

1. Require valid rear scan coverage and a supported scan ray toward the target.
2. If the target is more than 1 cm ahead, stop this reverse test with a reason.
3. Calculate the target's angle relative to the rear axis. If its magnitude
   exceeds `retreat_turn_limit_deg` (45 degrees), turn in place to align the
   rear before translating. The current circular footprint must remain clear.
4. Otherwise aim a circular reverse arc through the current centre and the dot.
   This is the pure-pursuit construction:

   ```text
   curvature = 2 * local_y / distance_squared
   angular_speed = signed_linear_speed * curvature
   ```

   A left-side rear target needs negative angular velocity: the nose turns
   right while the rear moves left. Reverse speed is reduced when necessary
   to respect `max_turn_rps`; it never exceeds `retreat_speed_mps`.
5. Check the entire proposed arc to this dot with hard clearance and the
   curve-sampling padding. If blocked, hold a stop. Do not substitute a clear
   straight ray for a blocked curve.
6. Publish the checked command and repeat with the latest pose/scan. On arrival,
   mark the dot red and advance to the preceding index, or finish the test.

The signed-speed updates to `motion_centres()` and `motion_is_clear()` now
support real reverse lines and arcs. Zero speed still means a stationary
footprint check. Forward curve behaviour and its separate clearance setting
are preserved.

| Setting | Initial value | Purpose |
| --- | --- | --- |
| `retreat_speed_mps` | 0.05 m/s | Maximum reverse speed |
| `max_turn_rps` | 0.40 rad/s | Maximum yaw rate |
| `retreat_turn_limit_deg` | 45 degrees | Larger rear errors align in place first |
| `retreat_curve_timeout_s` | 8 s per dot | Allows aligning and reversing through a bend |
| `retreat_timeout_s` | 4 s per dot | Still used by the straight-only tests |

## What to verify

- A rear-left target produces reverse speed and negative yaw; a rear-right
  target produces reverse speed and positive yaw.
- Tight bends slow translation instead of exceeding the yaw-rate limit.
- Large heading errors produce zero translation while aligning first.
- The orange target remains fixed until reached. Red count and target indices
  progress in order; reverse movement adds no green trail points.
- After the requested count, linear and angular velocities remain zero.
- Rear/target scan loss, stale sensors, hard-footprint violation, a blocked
  curve or timeout holds a stop with a reason.

This controller checks an arc directly to the next recorded dot. If that arc
cuts a corner and fails clearance, it stops; it does not search for another
reverse route. Escaping an already-violated footprint, new-departure scoring
and visited-direction memory remain later stages.

Automated checks cover reverse geometry and steering signs, arc collisions,
large-error alignment and closed-loop left/right bends, including a complete
90-degree trail in a synthetic curved corridor. These checks are not a live
Gazebo or hardware trial.
