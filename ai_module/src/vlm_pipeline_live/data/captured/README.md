# Captured live scene graphs (manual Pipeline B)

Each run is kept under a timestamped folder:

```text
data/captured/<scene_name>/<YYYYMMDD_HHMMSS>/
  <scene_name>_scene_graph.json
  <scene_name>_object_result.csv
  object_list.txt
  capture_meta.json
data/captured/<scene_name>/latest.json   # pointer to newest run
```

Runtime (before archive) uses the same idea under `/tmp/vlm_live_captures/`.

Compare against official VLA-3D graphs under `~/vla3d_data/Unity/<scene_name>/`.
