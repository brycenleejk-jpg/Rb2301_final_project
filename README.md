# RB2301 Final Project - Obstacle Course

The robot has to get through three sections of a "real-world" obstacle course: a zig-zag corridor with static obstacles, a small door, and dynamic swinging obstacles. You get the robot's pose and LiDAR scan. There is no occupancy grid, and the walls are rectilinear. Any approach is allowed.

Everything you write goes in one file, inside `ObstacleCourseNode.timer_callback()` between the two `###### INSERT CODE HERE ######` lines:

```
src/rb2301_fp/rb2301_fp/obstacle_course.py
```

What you have: `self.pose` (`[x, y, heading]`, metres and degrees, from Gazebo odometry or Optitrack), `self.last_scan` (LiDAR ranges, set once the first scan arrives), `self.move_2D(x, y, turn)` to drive, and `self.is_simulation`, which is set automatically from the LiDAR scan size.

## Set up (simulation)

Requirements: Ubuntu with ROS 2 Jazzy and Gazebo Harmonic, `ros-jazzy-ros-gz`, numpy, and your existing RB2301 workspace (the folder containing `src/`, e.g. `~/Documents/rb2301`).

1. Get this repo: `git clone https://github.com/rudra-8000/rb2301_final_project.git` (or Code > Download ZIP).
2. Copy `fp.sh` and `gz_fp.sh` into the workspace folder, next to `src/`.
3. Copy `src/rb2301_fp` and `src/rb2301_gz` into your workspace's `src/`. The new `rb2301_gz` replaces your old one and has the worlds and launch files for CA1, CA2 and the final project.
4. Build:

```bash
cd ~/Documents/rb2301
chmod +x fp.sh gz_fp.sh
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

`--symlink-install` means edits to `obstacle_course.py` take effect without rebuilding.

## Run

Two terminals, both in the workspace folder:

```bash
./gz_fp.sh     # terminal 1: Gazebo. Wait until the course and robot appear
./fp.sh --simulation  # terminal 2: your node
```

The goal is to get past the moving obstacles. To start the robot somewhere else (for example past the gate), change `x:=` and `y:=` in `gz_fp.sh`; a commented example is in the file. Stop with `Ctrl+C`.

The unchanged starter code just drives forward at 0.1 m/s.

## The simulation map

- Each time you launch, the map is randomised from `obstacle_modifier.py`: gate position, zig-zag size, obstacle speed, and the position of the static obstacles. Switch each one on or off with the `randomise_...` variables at the top of that file.
- The untouched map is `obstacle_course_world_fp_original.sdf`. Do not edit it; the launched map is generated from it.
- The moving obstacles have no collision in the simulation.

## Problems

- `Permission denied` on a `.sh` file: `chmod +x fp.sh gz_fp.sh`.
- Gazebo hangs on "Requesting list of world names": an old Gazebo is still running. `pkill -9 ruby`, then run `./gz_fp.sh` again.
- Nothing happens when you run `./fp.sh`: Gazebo has to be fully loaded first (check terminal 1 is still running). The node waits for the LiDAR before it starts.
- `ros2: command not found`: `source /opt/ros/jazzy/setup.bash`. Run the scripts from the workspace folder (the one with `install/`).
- Gazebo window black, slow or crashing (virtual machine, no GPU): in `src/rb2301_gz/launch/fp_gazebo.launch.py` set `LIBGL_ALWAYS_SOFTWARE` to `"1"`.

## Real robot: Python on ROS 2 Foxy

The controller and viewer use Python. Hardware mode checks the shared settings
in `src/rb2301_fp/rb2301_fp/runtime.py`: ROS 2 Foxy and
`RMW_IMPLEMENTATION=rmw_fastrtps_cpp`. Run both ROS processes on the robot. Use
SSH X11 forwarding to display the plot on your computer; do not start a Jazzy
hardware viewer in the same ROS domain.

The current robot is **14**, at `bingda@192.168.1.214`, with workspace
`~/Downloads/rb2301_fp_10`. Complete copy/build/run commands are in
[real_robot_commands.txt](real_robot_commands.txt).

Hardware navigation uses `/scan` and publishes `/cmd_vel`. It does not subscribe
to Optitrack and does not need a pose. Missing or stale LiDAR still stops the
robot. The original frame-conversion helper functions remain in the starter file
but are not used by hardware navigation.

Save your code, then copy it FROM YOUR COMPUTER:

```bash
cd /home/rmi-015/rb2301_final_project
ssh bingda@192.168.1.214 'mkdir -p ~/Downloads/rb2301_fp_10/src'
rsync -av src/rb2301_fp bingda@192.168.1.214:~/Downloads/rb2301_fp_10/src/
rsync -av fp.sh bingda@192.168.1.214:~/Downloads/rb2301_fp_10/
rsync -av --exclude=runs --exclude=__pycache__ visualization bingda@192.168.1.214:~/Downloads/rb2301_fp_10/
```

In every robot terminal, use:

```bash
source /opt/ros/foxy/setup.bash
export ROS_DOMAIN_ID=14
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

Build on the robot:

```bash
cd ~/Downloads/rb2301_fp_10
colcon build --symlink-install
chmod +x fp.sh
```

Run `basecontrol` and `rplidar` in separate SSH terminals. Optitrack/`vrpn` is not
required. In the controller terminal, first inspect scans/targets while stopped:

```bash
cd ~/Downloads/rb2301_fp_10
./fp.sh --hardware --calibrate
```

After verifying the physical footprint, LiDAR mount and sensor directions,
stop monitoring with Ctrl+C and run:

```bash
./fp.sh --hardware
```

For the viewer, use `ssh -X bingda@192.168.1.214` from your computer, set the Foxy
runtime in that SSH session, then run:

```bash
cd ~/Downloads/rb2301_fp_10
source install/setup.bash
python3 visualization/live_map.py --robot 14
```

The default hardware plot is robot-centred: current scan, candidate regions,
reported scores and carrot. It records command/target snapshots and timestamped
case marks, without inventing a position or travelled route. If the base actually
publishes `/odom`, opt into route plotting with `--pose-source odom`.

Use `--simulation` for Gazebo and `--hardware` for the robot; mode selection no
longer depends on LiDAR ray count. The hardware velocity cap remains 0.3 m/s.
Changing ROS environments is not a substitute for a hardware trial: the original
DDS deserialization cause remains unconfirmed until tested on the robot.
