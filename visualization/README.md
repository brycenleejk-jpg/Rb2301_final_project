# Live plot and case logger

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
