# Sample live scene (`live_scene`)

Captured with Pipeline B (GroundingDINO + LiDAR fusion + live scene graph) in the CMU VLN sim.

| File | Description |
|------|-------------|
| `live_scene_scene_graph.json` | VLA-3D-style scene graph (~125 objects) — same schema as `/tmp/vlm_live_scene_graph.json` |
| `live_scene_object_result.csv` | Object list for Pipeline A `SceneLoader` |
| `object_list.txt` | Human-readable label dump |

## Use with Pipeline A

```bash
# From ai_module workspace (or copy this folder into your vla3d root)
ros2 launch vlm_pipeline vlm_pipeline.launch.py \
  scene_name:=live_scene \
  vla3d_data_root:=$(pwd)/src/vlm_pipeline_live/data
```

Or export a fresh graph from a live run:

```bash
ros2 run vlm_pipeline_live write_object_list_from_scene_graph -- \
  --graph /tmp/vlm_live_scene_graph.json \
  --out-dir /tmp/vla3d_live/live_scene \
  --scene-name live_scene
```
