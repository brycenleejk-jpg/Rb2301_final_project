# Sideways-clearance scoring trial

This trial adds one term to the existing score. Hard rejection, depth and goal
weights, the existing soft bonus, and the depth-first tie rule stay unchanged.
Set `side_clearance_weight` to `0` to compare with the previous selector.

For each hard-clear direction, use the existing path projection:

```text
along = point_x * cos(angle) + point_y * sin(angle)
side  = -point_x * sin(angle) + point_y * cos(angle)
```

Take the smallest `abs(side)` among obstacles with `0 <= along <= lookahead`.
This is distance from the proposed robot-centre line, not extra space outside
the robot body. Cap it at the soft radius. With no points beside that segment,
use the cap: there is no measured side restriction there. The hard check still
uses all points and both ends of the finite segment.

```text
fraction = clip((side_clearance - hard_radius) / (soft_radius - hard_radius), 0, 1)
side_score = side_clearance_weight * fraction
total = depth_score + goal_score + existing_soft_bonus + side_score
```

Equal hard and soft radii give zero gradual side reward. Hard-blocked candidates
never receive a score. Extra room beyond the soft radius adds no further reward.

At goal weight 1, a 3-degree change gives these goal-score differences:

| Error relative to virtual goal | Goal-score difference |
| --- | ---: |
| 0 to 3 degrees | 0.00137 |
| 6 to 9 degrees | 0.00683 |
| 45 to 48 degrees | 0.03798 |
| 87 to 90 degrees | 0.05234 |

The trial weight is **0.30 maximum**, with hard 13 cm and soft 16 cm.
Across this 3 cm interval, 1 cm more side clearance adds 0.10 score. This exceeds
a 0.05234 heading change plus the 0.03 tie band, when the other score
contributions are equal. Changing the clearance interval changes this
sensitivity: a 5 cm interval would need weight 0.50 for the same 0.10 per cm.
Larger depth differences and the existing soft bonus can still change the winner.

Older ROS logs contain only the selected total and soft/hard category. They
cannot reconstruct the competing candidates' clearances. New `CANDIDATE` lines
record the selected direction and up to two alternatives **from the same scan**:
`total`, `goal`, `depth_score`, `soft`, and `side` are score contributions;
`side_clearance` is the capped distance in metres. Logging happens at selection,
not on every turn or drive update. Scores within the existing 0.03 band can
still lose to a deeper candidate.

Restart the controller and viewer for the trial. In the stationary preview,
the table adds `Side(cm)` for the selected path and alternatives. Test a wall
alongside the robot, then a narrow doorway, using the same parameters for both.
Compare `side_clearance_weight=0` and `0.30` without changing other weights.
Check whether side clearance improves, whether doorway passage still works,
and whether repeated selections oscillate.

This preference does not guarantee a centred route. A nearby obstacle beside
the start of several candidate paths may give them similar side distances.
It also does not change the curved-motion collision check or add retreat logic.
