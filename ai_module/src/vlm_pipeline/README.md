# VLM Pipeline (Track A)

Zero-shot **find**, **count**, and **instruction-following** pipeline for the [CMU VLN Challenge 2026](https://github.com/jainanshu0912/CMU-VLN-Challenge-2026). It answers challenge questions using pre-loaded VLA-3D scene data (`object_result.csv` + `scene_graph.json`) and a rule-based query parser—no API keys required by default.

## What it does

| Question type | Example | Output |
|---------------|---------|--------|
| **Find** (object reference) | `Find the pillow on the chair closest to the TV.` | Bounding-box marker + navigation waypoint |
| **Count** (numerical) | `How many sofas are below a window?` | Integer on `/numerical_response` |
| **Navigate** (instruction-following) | `Go near the stool and stop at the table.` | Ordered waypoints on `/way_point_with_heading` |

Pipeline flow:

```
/challenge_question → classify → parse → graph search / count / navigate → ROS publishers
```

Data is loaded once at startup from `vla3d_data_root/<scene_name>/`.

## Prerequisites

- Ubuntu 24.04 + ROS 2 Jazzy (or the challenge `iros2026_ai_module` container)
- VLA-3D Unity training data on disk, e.g. `~/vla3d_data/Unity/<scene_name>/` with:
  - `<scene>_object_result.csv`
  - `<scene>_scene_graph.json`
- For simulation: base autonomy stack running + Unity scene matching `scene_name`

## Build

```bash
cd /path/to/CMU-VLN-Challenge-2026/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline
source install/setup.bash
```

Inside the AI module Docker container, use `/home/docker/ai_module` as the workspace path.

## Launch

```bash
ros2 launch vlm_pipeline vlm_pipeline.launch.py scene_name:=arabic_room
```

### Launch arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `scene_name` | `chinese_room` | Unity scene folder under `vla3d_data_root` |
| `vla3d_data_root` | `/home/docker/vla3d_data/Unity` | Root containing per-scene CSV/JSON |
| `use_llm_parser` | `false` | `true` to parse questions with an LLM instead of rules |
| `vlm_backend` | `ollama` | Backend when `use_llm_parser:=true` (`ollama`, `gemini`, `gpt-4o`, `claude`) |
| `vlm_model` | `""` | Optional model name override |

Node parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `waypoint_standoff_m` | `0.7` | Base clearance from object center (also grows with object size) |
| `waypoint_reach_m` | `0.45` | Distance to advance to the next navigate leg |
| `waypoint_skip_if_within_m` | `0.9` | Skip / complete leg if this close to the object |
| `waypoint_no_republish_within_m` | `0.6` | Don’t republish inside this radius (prevents spin lock) |
| `waypoint_reach_dwell_sec` | `0.3` | Must stay inside reach radius this long before advancing |
| `waypoint_timeout_sec` | `120.0` | Per-leg timeout while following a path |
| `waypoint_republish_sec` | `2.0` | Re-publish rate while still far from the goal |

Example with host data path and larger standoff:

```bash
ros2 launch vlm_pipeline vlm_pipeline.launch.py \
  scene_name:=arabic_room \
  vla3d_data_root:=/home/anshu/vla3d_data/Unity
```

```bash
ros2 run vlm_pipeline vlm_pipeline_node --ros-args \
  -p scene_name:=arabic_room \
  -p waypoint_standoff_m:=1.2
```

## ROS interface

Matches the challenge `dummy_vlm` interface.

### Subscriptions

| Topic | Type | Notes |
|-------|------|-------|
| `/challenge_question` | `std_msgs/String` | Question text (BEST_EFFORT QoS) |
| `/state_estimation` | `nav_msgs/Odometry` | Robot pose for standoff + path following |

### Publications

| Topic | Type | When |
|-------|------|------|
| `/selected_object_marker` | `visualization_msgs/Marker` | Find — cube in `map` frame |
| `/way_point_with_heading` | `geometry_msgs/Pose2D` | Find — standoff goal; Navigate — sequential legs |
| `/numerical_response` | `std_msgs/Int32` | Count |

### Example questions

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find the pillow closest to the book on the stool.'}"

ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'How many sofas are below a window?'}"

ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Go near the stool under the picture and stop at the small table farthest from the columns.'}"
```

## Offline tests (no ROS)

Uses `questions/questions.json` from the repo root and local VLA-3D data.

```bash
cd ai_module/src/vlm_pipeline
python tests/test_offline_find.py
python tests/test_offline_count.py
python tests/test_offline_navigate.py
```

Set data root if needed:

```bash
export VLA3D_DATA_ROOT=~/vla3d_data/Unity
python tests/test_offline_find.py
```

Offline results (rule-based parser): **30/30** find, **15/15** count, **30/30** navigate questions across training scenes.

## Simulation workflow

1. Start the simulator + autonomy stack (`iros2026_system`, `system_simulation.sh`).
2. Swap Unity mesh to match `scene_name` (see challenge README).
3. Enable **Waypoint mode** in RViz.
4. Launch the pipeline with the same `scene_name`.
5. Send questions with `ros2 topic pub` (see examples above).

Wait ~2 s after launching the pipeline so DDS discovery completes.

## Supported training scenes

`arabic_room`, `chinese_room`, `home_building_1`, `home_building_2`, `hotel_room_1`, `hotel_room_2`, `japanese_room`, `livingroom_1`, `livingroom_2`, `livingroom_3`, `livingroom_4`, `loft`, `office_1`, `office_2`, `studio`

The Unity visual scene and `scene_name` must match for correct answers in simulation.

## Optional LLM query parser

Default is a free rule-based parser. To use an LLM for parsing find/count (still uses graph search for answers):

```bash
ros2 launch vlm_pipeline vlm_pipeline.launch.py \
  scene_name:=chinese_room \
  use_llm_parser:=true \
  vlm_backend:=ollama
```

Set the appropriate API key or Ollama endpoint for the chosen backend (see `vlm_pipeline/vlm_backends/`).

## Package layout

```
vlm_pipeline/
├── launch/
│   └── vlm_pipeline.launch.py      # Main node
├── vlm_pipeline/
│   ├── main_node.py                # ROS node
│   ├── scene_loader.py             # VLA-3D CSV + scene graph
│   ├── question_classifier.py
│   ├── query_parser.py             # Find/count rule-based + optional LLM JSON
│   ├── navigate_parser.py          # Instruction-following leg parser
│   ├── navigate_pipeline.py        # Resolve legs → waypoints
│   ├── graph_search.py             # Find
│   ├── count_pipeline.py           # Count
│   └── vlm_backends/               # Ollama, OpenAI, Claude, Gemini
└── tests/
    ├── test_offline_find.py
    ├── test_offline_count.py
    └── test_offline_navigate.py
```

## Limitations (Track A)

- Static scene graph only—no live camera or LiDAR perception
- Avoid-path constraints are parsed/logged but not geometrically enforced yet
- Find/navigate waypoints use a standoff near object centers; cluttered goals may still be hard for the base planner
- Test-time held-out scenes are not available in the released VLA-3D training set

Track B (live scene graph from sensors) is not included in this package.
