# Sequential pipeline (`vlm_sequential`)

One launch for evaluation: **explore → build a live scene graph → answer** `/challenge_question`.

Eval does **not** give VLA-3D CSV/JSON. This package runs Pipeline B (coverage tour + detector + graph) then Pipeline A (find / count / navigate) on the exported graph.

```text
System container (separate terminal)
  RViz + Unity + autonomy

AI container — one command
  ros2 launch vlm_sequential vlm_sequential.launch.py
        │
        ├─ explorer      drive 5–6 viewpoints, detect at each stop
        ├─ live_detector GroundingDINO / YOLO on 360° crops + LiDAR
        ├─ scene_graph   VLA-3D-style JSON
        └─ sequential    wait for export, then answer the question
```

## Launch (AI container)

```bash
cd /home/docker/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to vlm_sequential
source install/setup.bash

# RViz: Waypoint / Resume Navigation to Goal
ros2 launch vlm_sequential vlm_sequential.launch.py
```

Equivalent eval-style command (same launch, via dummy package):

```bash
ros2 launch dummy_vlm dummy_vlm.launch
```

Held-out scenes (unknown room type):

```bash
ros2 launch vlm_sequential vlm_sequential.launch.py \
  scene_type:=indoor \
  detector_backend:=grounding_dino
```

Known training scene (better detector captions):

```bash
ros2 launch vlm_sequential vlm_sequential.launch.py \
  scene_name:=live_scene \
  scene_type:=office_2 \
  num_viewpoints:=6
```

### Arguments

| Arg | Default | Meaning |
|-----|---------|---------|
| `scene_name` | `live_scene` | Export folder name under `/tmp/vla3d_live/` |
| `scene_type` | `indoor` | Detector vocab (`indoor`, `office`, `hotel`, or a Unity scene id) |
| `detector_backend` | `grounding_dino` | `grounding_dino` \| `yolo_world` \| `yoloe` |
| `num_viewpoints` | `6` | Coverage stops (including start) |
| `save_snapshots` | `false` | Desktop debug crops |
| `pipeline_a_export_root` | `/tmp/vla3d_live` | Where B writes A's CSV/JSON |

## What happens

1. Explorer plans viewpoints from `/registered_scan` and publishes `/way_point_with_heading`.
2. At each stop it triggers `/vlm_live/run_detection`; the graph accumulates.
3. When the tour ends, B writes `/tmp/vla3d_live/live_scene/*_object_result.csv` + `*_scene_graph.json`, then `/vlm_live/exploration_complete`.
4. Sequential loads that folder (does **not** use `~/vla3d_data`).
5. The first `/challenge_question` (eval publishes at 1 Hz from startup) is buffered, then answered:
   - **Find** → `/selected_object_marker` (`Marker` cube, `map`) + `/way_point_with_heading` (`Pose2D`)
   - **Count** → `/numerical_response` (`Int32`)
   - **Navigate** → sequential `Pose2D` on `/way_point_with_heading`

Do not start Pipeline A (`vlm_pipeline.launch.py`) at the same time — two nodes would fight over waypoints.

## System terminal (not this package)

```bash
docker exec -it iros2026_system bash
# then sim_start / system_simulation.sh as usual
```

Enable **Waypoint mode** in RViz before the AI launch.

## Tests

```bash
cd /home/docker/ai_module/src/vlm_sequential
PYTHONPATH=../vlm_pipeline:$PWD:$PYTHONPATH python3 -m unittest tests.test_export_paths
```
