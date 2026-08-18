# Pipeline B — Live Scene Graph

Separate ROS 2 package from Track A (`vlm_pipeline`). Builds a live scene representation from onboard sensors for held-out test scenes and the real robot.

## Current status

| Module | Status |
|--------|--------|
| `explorer.py` | Rotate-in-place exploration (4 views) |
| `scan_stability.py` | Registered scan stabilisation gate |
| `live_scene_graph.py` | VLA-3D-style relations → `SceneData` / scene_graph.json |
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
│   ├── pipeline_b.launch.py
│   ├── pipeline_b_cpu.launch.py
│   ├── pipeline_b_manual.launch.py
│   ├── pipeline_b_manual_cpu.launch.py
│   └── live_scene_graph.launch.py
└── vlm_pipeline_live/
    ├── explorer.py
    ├── scan_stability.py
    ├── equirect_to_perspective.py
    ├── grounding_dino_backend.py
    ├── lidar_camera_fusion.py
    ├── live_detector.py
    ├── live_scene_graph.py
    └── live_scene_graph_node.py
```

## Live scene graph

`live_scene_graph.py` converts fused 3D detections into a **VLA-3D / IRef-VLA** scene graph (same JSON schema as `*_scene_graph.json`) and a Pipeline A `SceneData` object that `GraphSearchMatcher` / `CountPipeline` can consume.

Heuristics follow [VLA-3D `scene_graph/generate_scene_info.py`](https://github.com/HaochenZ11/VLA-3D/tree/main/scene_graph) (above/below/on/in/near/closest/farthest/between/hanging_on), with **absolute** near distance (default 1.5 m) matching Pipeline A's geometric fallback.

### Relations

| Relation | Rule (AABB) |
|----------|-------------|
| `above` / `below` | XY IOM ≥ 0.5 and vertical separation |
| `on` | Contact gap ≤ 15 cm + XY IOM ≥ 0.5 + larger support footprint |
| `in` | Target center inside container + smaller size (container-like labels) |
| `near` | XY center distance < 1.5 m |
| `beside` | Near and not stacked |
| `closest` / `farthest` | Per-class distance ranking (VLA-3D schema) |
| `between` | Target projects onto segment between two anchors (perp ≤ 0.5 m) |
| `hanging_on` | Elevated near anchor, not already on/in something |

### Offline / library usage

```python
from vlm_pipeline_live.live_scene_graph import LiveSceneGraphBuilder
from vlm_pipeline.graph_search import GraphSearchMatcher

result = LiveSceneGraphBuilder(scene_name="live_scene").build_from_detections(detections)
scene = result.scene  # vlm_pipeline.scene_loader.SceneData
matcher = GraphSearchMatcher()
# matcher.find(scene, parsed_query)
```

### ROS

After `/vlm_live/detection_complete`, the scene-graph node publishes:

| Topic | Type |
|-------|------|
| `/vlm_live/scene_graph_json` | `std_msgs/String` (VLA-3D JSON) |
| `/vlm_live/scene_graph_complete` | `std_msgs/Bool` |

Also writes `/tmp/vlm_live_scene_graph.json` by default.

A checked-in sample from a sim run lives at
[`data/live_scene/`](data/live_scene/) (`live_scene_scene_graph.json` + CSV object list).

```bash
ros2 launch vlm_pipeline_live live_scene_graph.launch.py
# or included in pipeline_b_manual_cpu.launch.py
```

### Unit test

```bash
cd ai_module/src/vlm_pipeline_live
PYTHONPATH=../vlm_pipeline:$PYTHONPATH python3 -m unittest tests.test_live_scene_graph
```

Integration with Pipeline A (`main_node` `scene_mode:=live`) is the next step after you have live graphs from teleop mapping.

## Manual teleop mapping (recommended — GPU)

Use this on a machine with NVIDIA GPU + CUDA torch. Teleop the robot, then trigger detection.

### Build

```bash
cd /home/docker/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline vlm_pipeline_live
source install/setup.bash
```

### Verify GPU inside the AI container

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

### Run

```bash
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py
# optional: device:=cuda:0
# CPU fallback: pipeline_b_manual_cpu.launch.py
```

### Workflow

1. Drive the robot with RViz teleop / smart joystick to a viewpoint.
2. Let `/registered_scan` and `/camera/image` update briefly.
3. Trigger detection (GPU: typically seconds–tens of seconds for 4 crops):

```bash
ros2 topic pub --once /vlm_live/run_detection std_msgs/msg/Bool "{data: true}"
```

4. Scene-graph node builds relations after `/vlm_live/detection_complete` → `/vlm_live/scene_graph_json` (+ `/tmp/vlm_live_scene_graph.json`).
5. Move to new viewpoints and re-trigger (detections accumulate; graph rebuilds).
6. RViz: MarkerArray on `/vlm_live/detection_markers`, frame `map`.

Expect log: `GroundingDINO backend available | device=cuda`

### Manual-mode topics

| Topic | Type | Role |
|-------|------|------|
| `/vlm_live/run_detection` | `std_msgs/Bool` | Run GroundingDINO + fusion |
| `/vlm_live/clear_detections` | `std_msgs/Bool` | Reset accumulated map |
| `/vlm_live/detections_json` | `std_msgs/String` | Merged 3D objects |
| `/vlm_live/detection_markers` | `visualization_msgs/MarkerArray` | RViz boxes |
| `/vlm_live/scene_graph_json` | `std_msgs/String` | VLA-3D scene graph |
| `/vlm_live/scene_graph_complete` | `std_msgs/Bool` | Graph ready |

## CPU fallback

```bash
ros2 launch vlm_pipeline_live pipeline_b_manual_cpu.launch.py
# or explorer+detector:
ros2 launch vlm_pipeline_live pipeline_b_cpu.launch.py
```

Pin transformers if needed: `pip install --break-system-packages 'transformers>=4.37,<5'`

GPU explorer path (once waypoints work):

```bash
ros2 launch vlm_pipeline_live pipeline_b.launch.py
```
