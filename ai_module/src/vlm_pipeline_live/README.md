# Pipeline B — Live Scene Graph

Separate ROS 2 package from Track A (`vlm_pipeline`). Builds a live scene representation from onboard sensors for held-out test scenes and the real robot.

## Current status

| Module | Status |
|--------|--------|
| `explorer.py` | Rotate-in-place exploration (4 views) |
| `scan_stability.py` | Registered scan stabilisation gate |
| `live_scene_graph.py` | Not started |
| `live_detector.py` | GroundingDINO + LiDAR fusion → 3D boxes |
| `equirect_to_perspective.py` | 360° → 4×90° perspective crops + ray maps |

## Explorer

Publishes four viewing waypoints with headings **0° / 90° / 180° / 270°**. Each waypoint is offset **`rotation_standoff_m`** (default 0.6 m) from the anchor pose along the target heading — the autonomy stack treats same-`(x,y)` goals as already reached and sends zero speed, so a small forward offset is required for the robot to rotate and drive. After each view it waits **`scan_settle_sec`** (default 6 s) before the next heading; optional **`require_scan_stable`** uses point-count spread on `/registered_scan` (usually off for accumulated maps).

### Topics

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/state_estimation` | `nav_msgs/Odometry` |
| Subscribe | `/registered_scan` | `sensor_msgs/PointCloud2` |
| Publish | `/way_point_with_heading` | `geometry_msgs/Pose2D` |
| Publish | `/vlm_live/exploration_complete` | `std_msgs/Bool` |

Downstream Pipeline B nodes (detector, scene graph) should subscribe to `/vlm_live/exploration_complete`.

### Build

```bash
cd ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline_live
source install/setup.bash
```

### Run (simulator must be in waypoint mode)

```bash
ros2 launch vlm_pipeline_live explorer.launch.py
```

Or:

```bash
ros2 run vlm_pipeline_live explorer_node
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `headings_deg` | `[0, 90, 180, 270]` | View headings (degrees) |
| `rotation_standoff_m` | `0.8` | Forward offset from current pose for each waypoint (match Track A standoff; must exceed `position_tolerance_m`) |
| `position_tolerance_m` | `0.2` | Max distance to waypoint to accept a view (stack `waypointXYRadius` is ~0.3 m) |
| `min_settle_before_reached_sec` | `1.0` | Ignore reach checks briefly after publishing (avoids false positives) |
| `heading_tolerance_rad` | `0.35` | Yaw error to accept a view |
| `heading_wait_timeout_sec` | `120.0` | Max wait per view for robot to reach waypoint |
| `waypoint_republish_sec` | `2.0` | Republish interval while waiting |
| `require_scan_stable` | `false` | If true, gate on scan point-count stability (poor fit for growing maps) |
| `scan_settle_sec` | `6.0` | Fixed pause after each view when `require_scan_stable` is false |
| `scan_stable_window_sec` | `5.0` | Window for optional scan stability check |
| `scan_change_threshold` | `0.05` | Max relative point-count spread when `require_scan_stable` is true |
| `scan_stable_timeout_sec` | `30.0` | Max wait per view for stable scan |

## Equirect → perspective

`equirect_to_perspective.py` converts `/camera/image` (1920×640, 360°×120° FOV) into four **640×640** perspective crops aligned with explorer headings (**0° / 90° / 180° / 270°**, ~90° HFOV each).

Each crop includes a precomputed **pixel → unit ray** map in the camera optical frame (`x` right, `y` down, `z` forward). `live_detector.py` will use this for LiDAR–camera fusion.

```python
from vlm_pipeline_live.equirect_to_perspective import (
  EquirectPerspectiveProjector,
  ros_image_to_numpy,
)

projector = EquirectPerspectiveProjector()
crops = projector.crop_all(ros_image_to_numpy(camera_msg))

for crop in crops:
  ray = crop.ray_at(px=320, py=320)  # unit ray at crop center
```

Run the built-in sanity check:

```bash
python3 -m vlm_pipeline_live.equirect_to_perspective
```

## Live detector

`live_detector.py` runs **GroundingDINO** on the four perspective crops, then fuses each 2D box with **`/registered_scan`** using robot pose from **`/state_estimation`**.

After **`/vlm_live/exploration_complete`**, the node publishes:

| Topic | Type | Purpose |
|-------|------|---------|
| `/vlm_live/detections_json` | `std_msgs/String` | Fused 3D detections (JSON) |
| `/vlm_live/detection_markers` | `visualization_msgs/MarkerArray` | Debug boxes in `map` |
| `/vlm_live/detection_complete` | `std_msgs/Bool` | Signals downstream scene-graph build |

### GroundingDINO setup (inside ai_module container)

```bash
pip install torch torchvision
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

mkdir -p /home/docker/models
# Download config + checkpoint into /home/docker/models/
```

Set paths via launch args or environment:

```bash
export GROUNDINGDINO_CONFIG=/home/docker/models/GroundingDINO_SwinT_OGC.py
export GROUNDINGDINO_CHECKPOINT=/home/docker/models/groundingdino_swint_ogc.pth
```

### Run

```bash
# Terminal 1 — exploration
ros2 launch vlm_pipeline_live explorer.launch.py

# Terminal 2 — detector (auto-runs when exploration completes)
ros2 launch vlm_pipeline_live live_detector.launch.py \
  box_threshold:=0.3 text_threshold:=0.25
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_config_path` | `/home/docker/models/GroundingDINO_SwinT_OGC.py` | GroundingDINO config |
| `model_checkpoint_path` | `/home/docker/models/groundingdino_swint_ogc.pth` | Weights |
| `box_threshold` | `0.3` | Detection confidence |
| `text_threshold` | `0.25` | Text-token confidence |
| `detection_prompt` | indoor object list | Caption for exploration mapping |
| `min_lidar_points` | `8` | Min LiDAR points inside a 2D box |
| `auto_run_on_exploration_complete` | `true` | Wait for explorer; set `false` for manual teleop mode |
| `allow_repeat_detection` | `true` | Allow multiple `/vlm_live/run_detection` triggers |
| `accumulate_detections` | `true` | Merge objects across manual snapshots (3D NMS) |
| `save_snapshots` | `false` | Save equirect PNG + pose JSON per trigger |
| `snapshot_dir` | `/tmp/vlm_live_snapshots` | Snapshot output directory |
| `nms_distance_m` | `0.5` | 3D dedup distance across crops |

Question-specific prompts can be built in Python:

```python
from vlm_pipeline_live.grounding_dino_backend import prompt_from_question
prompt_from_question("Find the pillow closest to the book on the stool.")
# → "pillow . book . stool"
```

## Package layout

```
vlm_pipeline_live/
├── launch/
│   ├── explorer.launch.py
│   ├── live_detector.launch.py
│   ├── pipeline_b_cpu.launch.py
│   └── pipeline_b_manual_cpu.launch.py
└── vlm_pipeline_live/
    ├── explorer.py
    ├── scan_stability.py
    ├── equirect_to_perspective.py
    ├── grounding_dino_backend.py
    ├── lidar_camera_fusion.py
    └── live_detector.py
```

Integration with Pipeline A (`GraphSearchMatcher`, `main_node`) is planned after `live_scene_graph.py` exists.

## Manual teleop mapping (recommended for now)

Auto exploration is disabled while waypoint navigation is being tuned. Use **teleop** to drive the robot, then trigger detection at each stop.

### Build

```bash
cd /home/docker/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline_live
source install/setup.bash
```

### Run

Terminal 1 — sim + autonomy (teleop / smart joystick in RViz as usual).

Terminal 2 — live detector only (CPU):

```bash
ros2 launch vlm_pipeline_live pipeline_b_manual_cpu.launch.py
```

### Workflow

1. Drive the robot with RViz teleop or joystick to a viewpoint.
2. Let `/registered_scan` and `/camera/image` update for a second or two.
3. Trigger detection (CPU: expect ~2–5 min for 4 crops):

```bash
ros2 topic pub --once /vlm_live/run_detection std_msgs/msg/Bool "{data: true}"
```

4. Repeat from new viewpoints — detections **accumulate** in the map (3D NMS merge).
5. View fused boxes in RViz: **MarkerArray** on `/vlm_live/detection_markers`, frame **map**.
6. Read JSON map: `ros2 topic echo /vlm_live/detections_json --once`

Snapshots (equirect PNG + pose + detections JSON) save to `/tmp/vlm_live_snapshots/` by default.

Clear the accumulated map:

```bash
ros2 topic pub --once /vlm_live/clear_detections std_msgs/msg/Bool "{data: true}"
```

### Manual-mode topics

| Topic | Type | Role |
|-------|------|------|
| `/vlm_live/run_detection` | `std_msgs/Bool` | Run GroundingDINO + fusion on latest sensors |
| `/vlm_live/clear_detections` | `std_msgs/Bool` | Reset accumulated map |
| `/vlm_live/detections_json` | `std_msgs/String` | Merged 3D object list |
| `/vlm_live/detection_markers` | `visualization_msgs/MarkerArray` | RViz debug |

## CPU test (auto exploration — optional)

Use this while Docker GPU passthrough is broken. GroundingDINO runs on CPU (slow: ~2–5 min for 4 crops on first run).

### Prerequisites (inside `iros2026_ai_module`)

```bash
python3 -c "import torch, groundingdino; print('ok', torch.cuda.is_available())"
ls /home/docker/models/GroundingDINO_SwinT_OGC.py /home/docker/models/groundingdino_swint_ogc.pth
```

**Transformers version:** GroundingDINO expects the transformers 4.x BERT API. If you see  
`AttributeError: 'BertModel' object has no attribute 'get_head_mask'`, either rebuild after pulling the latest `vlm_pipeline_live` (includes a runtime patch), or pin:

```bash
pip install --break-system-packages 'transformers>=4.37,<5'
```

### Build

```bash
cd /home/docker/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline_live
source install/setup.bash
```

### Run (sim must be up, RViz in waypoint mode)

One launch starts **explorer + detector** (`force_cpu:=true`, shorter prompt):

```bash
ros2 launch vlm_pipeline_live pipeline_b_cpu.launch.py
```

Expected sequence:

1. ~60–120 s exploration (4 standoff waypoints + 6 s settle each)
2. `Exploration complete` on `/vlm_live/exploration_complete`
3. `Starting live detection...` then per-crop timing logs
4. `Published N fused 3D detections` on `/vlm_live/detections_json`

### Verify

```bash
ros2 topic echo /vlm_live/exploration_complete --once
ros2 topic echo /vlm_live/detection_complete --once
ros2 topic echo /vlm_live/detections_json --once
```

In RViz: add **MarkerArray** on `/vlm_live/detection_markers`, fixed frame **map**.

### Separate terminals (optional)

```bash
ros2 launch vlm_pipeline_live live_detector.launch.py force_cpu:=true
ros2 launch vlm_pipeline_live explorer.launch.py
```
