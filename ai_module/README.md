# AI module — sequential VLN pipeline

Evaluators: see **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for the few extra steps on top of the official `docker/README.md`.

Evaluation launch (same command as the challenge dummy VLM):

```bash
ros2 launch dummy_vlm dummy_vlm.launch
```

That starts **explore → live scene graph → answer** `/challenge_question`. Eval does not provide VLA-3D CSV/JSON.

```text
iros2026_system     Unity + autonomy + RViz (Waypoint mode)
iros2026_ai_module  dummy_vlm.launch
                      explorer → GroundingDINO + LiDAR → scene graph
                      sequential answers the buffered question
```

## ROS interface (scoring)

| Direction | Topic | Type | Use |
|-----------|--------|------|-----|
| Sub | `/challenge_question` | `std_msgs/String` | Eval publishes at 1 Hz |
| Sub | `/state_estimation` | `nav_msgs/Odometry` | Robot pose |
| Pub | `/way_point_with_heading` | `geometry_msgs/Pose2D` | Find + navigate |
| Pub | `/selected_object_marker` | `visualization_msgs/Marker` | Find: `CUBE` in `map` |
| Pub | `/numerical_response` | `std_msgs/Int32` | Count |

Find: cube marker plus a standoff Pose2D. Count: Int32 only. Navigate: sequential Pose2D goals. The same question string is answered **once** per launch (eval’s 1 Hz repeats are ignored).

`/vlm_live/*` topics are internal.

## How to run (same as evaluation)

From the repo `docker/` folder, GPU host:

```bash
xhost +
docker compose -f compose_gpu.yml up -d
```

System container:

```bash
docker exec -it iros2026_system bash
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

In RViz: **Resume Navigation to Goal** (Waypoint mode). Then AI container:

```bash
docker exec -it iros2026_ai_module bash
source /opt/ros/jazzy/setup.bash
source /home/docker/ai_module/install/setup.bash
ros2 launch dummy_vlm dummy_vlm.launch
```

Eval publishes `/challenge_question` at 1 Hz from startup. The node buffers it until the live graph is written under `/tmp/vla3d_live/live_scene/`.

Local one-shot (after the graph exists, or while exploring to test buffering):

```bash
ros2 run vlm_pipeline pub_challenge_question "Find the sofa."
ros2 run vlm_pipeline pub_challenge_question "How many chairs are there?"
ros2 run vlm_pipeline pub_challenge_question "Go to the plant."
```

## Packages

| Package | Role |
|---------|------|
| `dummy_vlm` | Eval entrypoint; launch file includes `vlm_sequential` |
| `vlm_sequential` | Waits for the live export, then find/count/navigate |
| `vlm_pipeline_live` | Coverage explorer, GroundingDINO, scene graph |
| `vlm_pipeline` | Question classifier, graph search, waypoints |

Launch arguments (optional on `vlm_sequential.launch.py` / `dummy_vlm.launch`):

| Arg | Default | Meaning |
|-----|---------|---------|
| `scene_name` | `live_scene` | Export folder under `/tmp/vla3d_live/` |
| `scene_type` | `indoor` | Detector vocab (`indoor`, `office`, `hotel`, or a Unity id) |
| `detector_backend` | `grounding_dino` | Open-vocab 2D detector |
| `num_viewpoints` | `6` | Coverage stops (including start) |
| `pipeline_a_export_root` | `/tmp/vla3d_live` | Where the live CSV/JSON are written |

## Docker image

Hub: [jainanshu0912/cmu-vln-ai-module](https://hub.docker.com/r/jainanshu0912/cmu-vln-ai-module) (`:latest`). Pull that instead of `--build` when possible (see [INSTRUCTIONS.md](INSTRUCTIONS.md)).

`ai_module/docker/Dockerfile` is self-contained: PyTorch, GroundingDINO, BERT, FastDDS UDP (no shared-memory data sharing), and a colcon build of the four packages. Weights are wget’d if `docker/models/groundingdino_swint_ogc.pth` is not staged.

```bash
# from ai_module/
./docker/build.sh
```

Official compose (`docker/compose_gpu.yml`) builds from this Dockerfile. Do not use bind-mount dev compose for eval.

## Offline Pipeline A only

If you already have CSV/JSON (Unity GT or a previous live export):

```bash
ros2 launch vlm_pipeline vlm_pipeline.launch.py \
  scene_name:=live_scene \
  vla3d_data_root:=/home/docker/ai_module/src/vlm_pipeline_live/data
```

