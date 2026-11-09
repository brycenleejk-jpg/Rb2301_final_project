# Stage 4: record and display the travelled trail

The controller records ordered course-frame positions in `self.trail`.
`record_trail()` runs after valid pose/scan data and the start position are
available, before the next movement command is chosen.

1. Save the initial position once.
2. During DRIVE or CURVE, append the current position when its distance from
   the last saved dot reaches `trail_spacing_m` (initially 0.05 m).
3. TURN, SELECT and DONE do not append points. Changing heading alone adds none.
4. Keep revisits as separate entries in chronological order.

The phase at this point describes the previous command, whose displacement is
now being measured. Thus arrival can be recorded before an advance completes
or the next command switches to turning. The distance is ordinary Euclidean
displacement, rather than progress projected along a goal heading. Dots are
measured positions; their spacing can slightly exceed the threshold because
the controller samples at 20 Hz.

## Green dots and dotted circles

The fixed-course inset shows a green dot at each saved position and a green
dotted circle of radius `trail_radius_m` (initially 0.10 m) around it. The title
shows the dot count and circle radius. The map includes the whole trail, even
after the robot returns to its start.

The main LiDAR view also shows green trail dots. They are translated and rotated
relative to the current robot pose, so the robot stays at the centre facing up
while its history moves around it. Only dots inside the current view are visible.
The dotted circles remain in the fixed-course inset to keep the scan readable.

Internal positions use the template's `to_course()` conversion. Displayed dots
use the same relative coordinates as the robot marker:

```text
display rightward x = -relative course y
display forward/up y = relative course x
```

The circle is a visual indication of a nearby area around a visited centre.
It does not establish that the whole enclosed area was travelled or is clear.
At this stage it does not affect scores, clearance, movement or revisits.
The trail's chronological order is retained for later backtracking.

| Parameter | Initial value | Meaning |
| --- | --- | --- |
| `trail_spacing_m` | 0.05 m | Minimum displacement before appending another dot |
| `trail_radius_m` | 0.10 m | Dotted-circle radius in the fixed-course inset |

The trail stays in controller memory. Restarting the controller clears it;
restarting only the viewer displays the current full trail again.

## Test one behaviour at a time

Restart the controller and viewer after updating their source:

```bash
python3 src/rb2301_fp/rb2301_fp/obstacle_course.py --move
```

In a separate terminal:

```bash
python3 visualization/live_map.py --simulation
```

Check that translation adds dots approximately every 5 cm; turning in place
does not. Verify right/up movement matches the axes, previous dots remain
visible, and every circle has a 10 cm radius. Crossing from segment A to B
must retain the trail. A return along the route adds another chronological
visit; it does not erase the earlier one.

Default stationary preview records only its initial dot. Use `--step` or
`--move` to test trail recording during controller-driven translation.

[Stages 5 and 6a](STAGES_5_6A.md) add persistent front-blockage reporting and an
optional one-dot straight retreat. Trail recording itself does not reject
visited areas or choose retreat directions.
[Stage 6b](STAGE_6B.md) follows several straight trail dots in reverse order.

[Synthetic fixed-map preview](../../visualization/previews/trail_fixed_map.png)
shows the green dots and dotted circles after three simulated advances. This
is a controller-generated illustration, not a live Gazebo or hardware trial.
