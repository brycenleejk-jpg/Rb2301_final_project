# Stage 6b: retreat through several straight trail dots, then stop

Stage 6a reached one dot and held a stop. This stage repeats that same checked
reverse movement through a requested number of consecutive dots. Start with
three dots on a straight approach to the U-shaped dead end.
[Stage 6c](STAGE_6C.md) adds a separate `--backtrack-curves N` steering test;
the commands described here remain straight-only.

## What changed

`--backtrack-dots N` enables normal forward movement and trail recording, then
allows a bounded N-dot retreat after confirmed front blockage. The original
`--backtrack-one`, `--move`, `--step` and stationary preview modes remain available.

`retreat_limit` stores the requested count. Zero means detection only; one is
the original one-dot test. The initial target is still the latest earlier dot
outside the 1 cm arrival tolerance.

On reaching each dot:

1. Publish zero velocity and mark its dot/circle red.
2. If the requested count is reached, enter RETREAT_DONE and hold a stop.
3. If index zero was reached first, stop at the trail start and report the
   smaller achieved count. Do not wrap around to the end of the list.
4. Otherwise decrement `retreat_index` by one and restart the per-dot timer.
5. On the following valid timer update, check the new target and actual rear
   sweep before publishing another reverse command.

For example, a three-dot test might visit indices `7 -> 6 -> 5`, then stop.
It never chooses an older point because that point happens to be spatially
closer. The trail is preserved and no new green dots are recorded in reverse.

## What stays the same

- Reverse speed: `retreat_speed_mps`, initially 0.05 m/s.
- Arrival tolerance: `retreat_reach_m`, initially 0.01 m.
- Timeout: `retreat_timeout_s`, initially 4 s **per dot**.
- Require usable rear scan coverage and hard clearance on every segment.
- A rear obstacle, footprint violation, sensor failure, timeout or target that
  needs a bend stops the attempt and holds the stop.
- No turning or curved reverse movement is added in this stage.
- Reached red dots are display evidence, not a visited-direction filter.
- No 360-degree departure scoring or automatic forward restart is added.

## Test

Stop the previous controller. Reset the robot to before the dead end so this
new run records a trail. Starting a new controller at the blocked position
leaves no earlier dots to follow.

Controller:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --backtrack-dots 3
```

Restart the viewer in a separate terminal:

```bash
python3 visualization/live_map.py --simulation
```

Expected states:

```text
Normal movement -> WAIT_BLOCKED -> RETREAT
                                     dot 7 reached -> target 6
                                     dot 6 reached -> target 5
                                     dot 5 reached -> RETREAT_DONE, hold stop
```

Actual indices depend on the recorded trail. The viewer should show
`MULTI-DOT TEST`, a moving orange target, reached counts `1 / 3`, `2 / 3`,
`3 / 3`, and three red dots/circles. During reverse, linear x is negative
and angular z is zero. There is a brief stopped update between targets while
the next segment is prepared. On completion, both velocities stay zero.

If a later dot is off the straight reverse line, expect BLOCKED with a reason
and the already-reached dots left red. That is the intended boundary of this
test; bend-following will be tested separately.

Controller restarts clear the trail and reached count. Viewer restarts retain
the controller's current trail. The original one-dot command still stops after
one reached point:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --backtrack-one
```

The automated checks cover ordering, limits, trail exhaustion, per-dot timeout,
new rear obstacles, later bends, converted course positions, the display and
a closed-loop synthetic U corridor. They do not replace a live Gazebo trial.
