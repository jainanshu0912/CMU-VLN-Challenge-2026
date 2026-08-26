# Pipeline B — Live Scene Graph (`vlm_pipeline_live`)

ROS 2 package that builds a **live** indoor scene representation from the robot’s
onboard **360° camera + LiDAR**, then turns it into a **VLA-3D-style scene graph**
that Pipeline A (`vlm_pipeline`) can query for find / count / navigate.

This is separate from Track A’s static offline graphs under `~/vla3d_data`.

## What's new (recent changes)

| Change | Detail |
|--------|--------|
| **Pluggable 2D backends** | Same fusion / NMS / graph path; swap vision with `detector_backend:=grounding_dino` \| `yolo_world` \| `yoloe` \| `owlvit`. |
| **YOLOE / YOLO-World** | Ultralytics open-vocab (`yoloe-11s-seg.pt` / `yolov8s-worldv2.pt`) for A/B vs GroundingDINO. |
| **Gemini label verify** | Free Flash API (`gemini_verify:=true`) keep/relabel/drop boxes **after** detection, before LiDAR fusion. |
| **Backend A/B compare** | `compare_backend_runs` scores DINO / DINO+Gemini / YOLOE graphs vs Unity GT in one table. |
| **Class-aware 3D NMS + label canonicalize** | Same-label merges with size-aware radius; cleans noisy DINO/YOLO phrases before the graph. |
| **Desktop debug snapshots** | Annotated crops + cyan fused centers on host `~/Desktop/vlm_live_snapshots`. |
| **GPU-only** | CPU launch files removed; vision backends require CUDA. |
| **Unique graph captures** | Timestamped folders under `/tmp/vlm_live_captures/` + `compare_scene_graphs` vs Unity GT. |

---

## What this pipeline does (big picture)

```text
 /camera/image  (equirect 360°)
        │
        ▼
  4× 90° perspective crops
        │
        ▼
  Open-vocab 2D backend  ──►  2D boxes + labels on each crop
  (grounding_dino | yolo_world | yoloe)
        │
        ▼
  Optional Gemini verify (keep / relabel / drop)
        │
        ▼
  LiDAR fusion (/registered_scan + /state_estimation)
        │                 LiDAR points that fall inside each 2D box
        ▼                 → 3D AABB in map frame
  Accumulated 3D objects (class-aware NMS across views)
        │
        ▼
  Live scene graph (near / on / closest / …)
        │
        ▼
  Optional: export CSV/JSON → Pipeline A find/count/navigate
```

**Important split (read this first):**

| Stage | What it is | Used for navigation? |
|-------|------------|----------------------|
| 2D boxes on crop images | **Chosen backend only** (vision) | No |
| Cyan crosses on crop images | Fused **3D centers** reprojected into the crop | Diagnostic only |
| `/vlm_live/detections_json` + RViz markers | **Fused 3D** boxes in `map` | Yes — these positions feed the graph / later nav |
| Scene graph relations | Geometry on those 3D boxes | Querying (find/count), not motion itself |

If labels/boxes look wrong on the Desktop PNGs → vision / prompt / crop / **backend** issue.  
If boxes look fine but cyan is far outside the box → camera–LiDAR alignment / projection issue.  
If both look fine but the robot still goes to the wrong place → object choice, standoff, or autonomy/trav — not necessarily the 2D model.

---

## Recommended mode right now: **manual teleop**

Manual teleop is still the best way to debug labels. The autonomous explorer now plans **5–6 XY standpoints** from `/registered_scan` and runs a 360° detect at each stop.

### 1. Build (inside `iros2026_ai_module`)

```bash
cd /home/docker/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline vlm_pipeline_live
source install/setup.bash
```

### 2. Launch detector + scene graph

```bash
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2 \
  scene_type:=office \
  detector_backend:=yoloe \
  save_snapshots:=true
```

Defaults worth knowing:

| Launch arg | Default | Meaning |
|------------|---------|---------|
| `detector_backend` | `grounding_dino` | `grounding_dino` \| `yolo_world` \| `yoloe` \| `owlvit` |
| `yolo_model` | `""` | Empty → backend default (`yolov8s-worldv2.pt` or `yoloe-11s-seg.pt`) |
| `gemini_verify` | `false` | Gemini Flash keep/relabel/drop after 2D detection |
| `gemini_model` | `gemini-3.6-flash` | Free-tier verify model |
| `scene_type` | `office` | Exact Unity scene (`office_2`, `hotel_room_1`, …) or `office` \| `hotel` \| `livingroom` \| `home` \| `cultural` \| `indoor` |
| `detection_prompt` | `""` | If set, overrides `scene_type` caption |
| `scene_name` | `office_2` | Name stamped into saved graphs |
| `save_snapshots` | `true` | Write annotated crops to disk |
| `snapshot_dir` | `/tmp/vlm_live_snapshots` | **Bind-mounted to host `~/Desktop/vlm_live_snapshots`** |
| `graph_output_dir` | `/tmp/vlm_live_captures` | Unique timestamped graph folders |

### 3. Map the room

1. Teleop to a good viewpoint; wait a moment for camera + scan to update.
2. Trigger one detection:

```bash
ros2 topic pub --once /vlm_live/run_detection std_msgs/msg/Bool "{data: true}"
```

3. Move to a **new** viewpoint (don’t spam detect from the same pose).
4. Trigger again — detections **accumulate** with class-aware 3D NMS.
5. Start a fresh capture:

```bash
ros2 topic pub --once /vlm_live/clear_detections std_msgs/msg/Bool "{data: true}"
```

### 4. Inspect debug images on the host Desktop

After each trigger, open:

```text
~/Desktop/vlm_live_snapshots/snapshot_XXXX_<ts>/
  snapshot_XXXX_equirect.png           # full 360°
  snapshot_XXXX_crop_h000.png          # raw 90° crop
  snapshot_XXXX_crop_h000_boxes.png    # ← look here
  snapshot_XXXX_crop_h090_boxes.png
  snapshot_XXXX_crop_h180_boxes.png
  snapshot_XXXX_crop_h270_boxes.png
  snapshot_XXXX_meta.json
```

**How to read `*_boxes.png`:**

- **Colored rectangles + text** (`chair 0.87`) = **2D backend** result on that crop only.
- **Cyan crosses** = 3D fused object centers (from LiDAR points inside boxes) projected back into the crop.
  - Cyan inside the box → fusion is at least consistent with that detection.
  - Cyan far outside / missing → suspect LiDAR–camera geometry or empty/wrong point association.

---

## Pluggable 2D backends (A/B comparison)

All backends share one API (`detection_backend.create_detection_backend`). Fusion, NMS, scene graph, and snapshots stay the same — only the 2D boxes/labels change.

| Backend | Launch | Notes |
|---------|--------|-------|
| GroundingDINO | `detector_backend:=grounding_dino` (default) | Needs config + checkpoint under `/home/docker/models/` |
| YOLO-World v2 | `detector_backend:=yolo_world` | Ultralytics `YOLOWorld`; default `yolov8s-worldv2.pt` |
| **YOLOE (YOLO11 open-vocab)** | `detector_backend:=yoloe` | Ultralytics `YOLOE`; default `yoloe-11s-seg.pt` — closest to “YOLO v11 World” |
| **OWL-ViT v2** | `detector_backend:=owlvit` | Hugging Face `google/owlv2-base-patch16` (override with `yolo_model:=…`) |
| Gemini as detector | `detector_backend:=gemini` | Not implemented — Gemini is a **verifier**, not a box proposer |

**About “YOLO v11 World”:** There is no separate Ultralytics `yolov11-world` checkpoint. Open-vocab on the YOLO11 line is **YOLOE** (`yoloe-11*-seg.pt`). Classic **YOLO-World** remains `yolov8*-worldv2.pt`. Prefer `yoloe` for the YOLO11 comparison; use `yolo_model:=yoloe-11m-seg.pt` / `yoloe-11l-seg.pt` for stronger models. Do **not** use `*-seg-pf.pt` (prompt-free) — those reject `set_classes()`.

### Gemini label verification (free Flash API)

After DINO/YOLO proposes boxes, optionally ask **Gemini Flash** to **keep / relabel / drop** each box before LiDAR fusion. Boxes still come from the detector; Gemini only cleans labels.

```bash
# Host: create a key at https://aistudio.google.com/apikey then export + recreate AI container
export GEMINI_API_KEY="AIza..."   # must be a real Generative Language API key
# (or GOOGLE_API_KEY)
echo "key len=${#GEMINI_API_KEY}"  # sanity: should be > 20
cd ~/CMU-VLN-Challenge-2026/docker
docker compose -f compose_gpu.dev.yml up -d --force-recreate ai_module

# Inside AI container — confirm the key arrived
echo "container key len=${#GEMINI_API_KEY} suffix=${GEMINI_API_KEY: -4}"
pip3 install -U --break-system-packages google-genai pillow

ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2 \
  detector_backend:=yoloe \
  gemini_verify:=true \
  gemini_model:=gemini-3.6-flash \
  save_snapshots:=true
```

If logs say `API_KEY_INVALID`, Gemini never ran — the pipeline fail-opens and keeps detector boxes. Create a new key in AI Studio, export it on the **host**, recreate the container, and confirm `container key len=...` is non-zero before launching. If logs say the model is `no longer available`, switch to `gemini_model:=gemini-3.6-flash` (current default).

| Arg | Default | Meaning |
|-----|---------|---------|
| `gemini_verify` | `false` | Enable Gemini keep/relabel/drop |
| `gemini_model` | `gemini-3.6-flash` | Free-tier Flash model |
| `gemini_api_key` | `""` | Optional override; else env `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `gemini_fail_open` | `true` | On API/parse errors, keep original detector boxes |

Logs show `[gemini_verify] N → M (keep=… relabel=… drop=…)`. Desktop `*_boxes.png` are **after** verification.

If you see `Expecting ',' delimiter` or `503 UNAVAILABLE`, rebuild — the verifier now uses JSON schema mode, batches ≤12 boxes/call, retries short 503s, and repairs mildly broken JSON. A `dino_gemini` capture only counts as different from plain DINO when those success logs appear.

**Free-tier quota:** `gemini-3.6-flash` is often capped around **20 requests/day**. One detection trigger uses several calls (4 crops × batches). After `429 RESOURCE_EXHAUSTED` / `GenerateRequestsPerDay`, the verifier **disables for the rest of the run** and keeps DINO boxes. For multi-pose mapping today, use `gemini_verify:=false`, or wait for quota reset / try another model / enable billing.

### Install YOLO deps (inside `iros2026_ai_module`)

```bash
# PEP 668: use --break-system-packages inside the AI container (same as Dockerfile torch/DINO installs)
pip3 install -U --break-system-packages ultralytics
# YOLOE text prompts also need Ultralytics CLIP (set_classes); auto-install fails under PEP 668
pip3 install -U --break-system-packages git+https://github.com/ultralytics/CLIP.git
```

First run downloads weights + a MobileCLIP text encoder for YOLOE (needs network once).

### Compare backends on the same teleop tour

1. Run with DINO, trigger detections from a few poses, archive/compare the graph.
2. Clear, relaunch with `detector_backend:=yoloe` (or `yolo_world`), repeat the **same** viewpoints.
3. Diff Desktop `*_boxes.png` and `compare_scene_graphs` Label P/R + matched counts.

```bash
# GroundingDINO (default)
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2 detector_backend:=grounding_dino save_snapshots:=true

# YOLOE / YOLO11 open-vocab (recommended YOLO path)
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2 detector_backend:=yoloe \
  yolo_model:=yoloe-11s-seg.pt save_snapshots:=true

# OWL-ViT v2 (zero-shot; first run downloads google/owlv2-base-patch16)
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2 detector_backend:=owlvit \
  box_threshold:=0.2 save_snapshots:=true

# Classic YOLO-World v2
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2 detector_backend:=yolo_world \
  yolo_model:=yolov8s-worldv2.pt save_snapshots:=true
```

---

## Module-by-module

### `equirect_to_perspective.py`

- Input: `/camera/image` equirect (~1920×640, 360°×120°).
- Output: four **640×640** crops at headings **0° / 90° / 180° / 270°** (~90° HFOV).
- Also builds per-pixel **rays** in the camera optical frame for projecting LiDAR into the crop.

### `detection_backend.py` + backends + `label_utils.py`

- Factory: `create_detection_backend("grounding_dino" | "yolo_world" | "yoloe" | "owlvit" | …)`.
- Optional: `gemini_label_verifier.py` — Gemini Flash keep/relabel/drop per crop (`gemini_verify:=true`).
- Same dotted caption for all backends (`chair . desk . lamp`); YOLO splits it via `set_classes`.
- **`scene_type:=office_2` / `hotel_room_1` / …** → per-scene caption built from that Unity GT vocab.
- **`scene_type:=office` / `hotel` / `livingroom` / `home` / `cultural`** → aggregated type prompts.
- **`scene_type:=indoor`** → livingroom aggregate (general indoor).
- Regenerate captions from GT: `python3 scripts/generate_scene_prompts.py` (then paste into `label_utils.py`).
- **`canonicalize_label()`** cleans phrases (`trash bin` → `trash can`, `cabinet shelf` → `cabinet`) before NMS / graph.

Wrong bbox labels here are a **vision / backend** problem, not navigation math yet.

### `lidar_camera_fusion.py`

For each 2D box:

1. Transform `/registered_scan` points into the camera frame using `/state_estimation`.
2. Project points into that crop.
3. Keep points that fall **inside the 2D box**.
4. If enough points (`min_lidar_points`, default 8) → fit a 3D axis-aligned box in **map** frame.

Then **class-aware 3D NMS**:

- Only merge detections with the **same canonical label**.
- Merge radius grows with object size (desks merge more aggressively than cups).
- Prefer higher confidence, then more LiDAR points.

These 3D boxes are what RViz markers and the scene graph use.

### `live_detector.py`

ROS node wiring:

| Subscribe | Publish |
|-----------|---------|
| `/camera/image` | `/vlm_live/detections_json` |
| `/registered_scan` | `/vlm_live/detection_markers` |
| `/state_estimation` | `/vlm_live/detection_complete` |
| `/vlm_live/run_detection` | |
| `/vlm_live/clear_detections` | |
| `/vlm_live/exploration_complete` (optional extra snap) | |

Also writes Desktop snapshots when `save_snapshots:=true`.

### `live_scene_graph.py` / `live_scene_graph_node.py`

Builds VLA-3D-compatible relations (`near`, `on`, `above`, `closest`, …) from the fused 3D boxes.

Saves uniquely so runs are not overwritten:

```text
/tmp/vlm_live_captures/<scene_name>/<YYYYMMDD_HHMMSS>/scene_graph.json
/tmp/vlm_live_captures/<scene_name>/latest_scene_graph.json   # pointer only
```

### Eval helpers

| Tool | Role |
|------|------|
| `archive_captured_scene` | Copy a graph into `data/captured/<scene>/<run_id>/` |
| `compare_scene_graphs` | Score live graph vs `~/vla3d_data/Unity/<scene>/` |
| `write_object_list_from_scene_graph` | Export Pipeline A CSV + scene folder |

Example compare:

```bash
ros2 run vlm_pipeline_live compare_scene_graphs -- \
  --pred /tmp/vlm_live_captures/office_2/latest_scene_graph.json \
  --gt ~/vla3d_data/Unity/office_2/office_2_scene_graph.json \
  --out /tmp/office_2_compare.json
```

**Label P / R** in the report = how well predicted **class counts** match GT (ignores XYZ).  
**Matched** = same canonical label **and** close in XY.  
Navigation quality is closer to **matched 3D positions**, not label P/R alone.

---

## Data flow vs Pipeline A

```text
Pipeline B (this package)          Pipeline A (vlm_pipeline)
─────────────────────────          ────────────────────────
live 3D objects + scene graph  →   SceneLoader / GraphSearchMatcher
                                   publish /way_point_with_heading
                                   autonomy drives there
```

Export for A:

```bash
ros2 run vlm_pipeline_live write_object_list_from_scene_graph -- \
  --graph /tmp/vlm_live_captures/office_2/latest_scene_graph.json \
  --out-dir /tmp/vla3d_live/office_2 \
  --scene-name office_2

ros2 launch vlm_pipeline vlm_pipeline.launch.py \
  scene_name:=office_2 \
  vla3d_data_root:=/tmp/vla3d_live
```

---

## Package layout

```text
vlm_pipeline_live/
├── README.md
├── launch/
│   ├── pipeline_b_manual.launch.py      # teleop + manual detect
│   ├── pipeline_b.launch.py             # coverage explorer + detect-per-stop
│   ├── explorer.launch.py
│   ├── live_detector.launch.py
│   └── live_scene_graph.launch.py
├── data/
│   ├── live_scene/                      # sample checked-in graph
│   └── captured/                        # archived runs
└── vlm_pipeline_live/
    ├── equirect_to_perspective.py
    ├── grounding_dino_backend.py
    ├── label_utils.py
    ├── lidar_camera_fusion.py
    ├── detection_vis.py                 # boxes + cyan overlays
    ├── live_detector.py
    ├── live_scene_graph.py
    ├── live_scene_graph_node.py
    ├── capture_paths.py
    ├── archive_captured_scene.py
    ├── compare_scene_graphs.py
    ├── write_object_list_from_scene_graph.py
    ├── explorer.py
    ├── viewpoint_planner.py
    └── scan_stability.py
```

---

## Docker note (snapshots on Desktop)

`compose_gpu.dev.yml` mounts:

```text
~/Desktop/vlm_live_snapshots  →  /tmp/vlm_live_snapshots
```

Recreate `ai_module` after compose changes:

```bash
cd ~/CMU-VLN-Challenge-2026/docker
docker compose -f compose_gpu.dev.yml up -d --force-recreate ai_module
```

---

## Explorer mode (coverage tour)

Plans XY standpoints from `/registered_scan` (free-space grid + farthest-point sampling over the **full scanned room**, not an 8 m bubble). The 360° camera already covers all headings at each stop, so the explorer does **not** rotate in place.

After the start snap, the **next stop is the free cell farthest from every snap already taken**. Failed goals are blacklisted. The explorer does **not** throw away the tour and nearest-neighbor replan after every detect (that kept livingroom tours jittering around spawn).

At each stop: settle → `/vlm_live/run_detection` → wait for `/vlm_live/detection_complete` → accumulate with 3D NMS → next uncovered goal.

If the robot wedges (table/cupboard gap), the explorer **backs up ~2.5 m** then republishes the target. It does **not** pull the goal closer — that sat inside the 2 m reach radius and looked “reached” without moving.

A stop within **2.0 m** of the viewpoint counts as reached; detect still runs if you got within ~2.5 m or made ≥1 m of progress. Obstacle inflation for viewpoint free space is **0.30 m** (`free_clearance_m`) with **0.40 m** wall inset. Long rooms can raise the snap budget (`auto_num_viewpoints:=true`, cap 8).

When the tour ends it exports a Pipeline A folder:

`/tmp/vla3d_live/<scene_name>/` (`*_object_result.csv` + `*_scene_graph.json`)

```bash
ros2 launch vlm_pipeline_live pipeline_b.launch.py \
  scene_name:=arabic_room \
  scene_type:=arabic_room \
  num_viewpoints:=6 \
  detector_backend:=grounding_dino
```

Then:

```bash
ros2 launch vlm_pipeline vlm_pipeline.launch.py \
  scene_name:=arabic_room \
  vla3d_data_root:=/tmp/vla3d_live
```

Planned snaps show as spheres on `/vlm_live/explorer_viewpoints` (green = start).

Useful explorer params: `stuck_backup_m` (2.5), `free_clearance_m` (0.30), `wall_inset_m` (0.40), `max_plan_radius_m` (25), `min_viewpoint_spacing_m` (3.0).

---

## Debugging checklist

1. **Wrong labels / silly boxes on Desktop crops** → tune prompt, thresholds, canonicalize; not LiDAR first.  
2. **Good boxes, cyan far away** → investigate `lidar_camera_fusion` / sensor frame / projection.  
3. **Good boxes + cyan OK, bad RViz map boxes** → NMS / accumulate / multi-view duplicates.  
4. **Good 3D markers, bad robot motion** → Pipeline A standoff / autonomy / traversable area.

---

## Backend A/B/C vs Unity GT (office_2)

Capture three live graphs with the **same teleop viewpoints**, then score them against
`~/vla3d_data/Unity/office_2/office_2_scene_graph.json` (mounted as `/home/docker/vla3d_data/...`).

### 1. Build

```bash
cd /home/docker/ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline_live
source install/setup.bash
```

### 2. Capture (one launch at a time)

Use identical viewpoints and the same number of `/vlm_live/run_detection` triggers for each.

```bash
# A) GroundingDINO
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2_dino scene_type:=office \
  detector_backend:=grounding_dino gemini_verify:=false \
  save_snapshots:=true

# B) GroundingDINO + Gemini
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2_dino_gemini scene_type:=office \
  detector_backend:=grounding_dino gemini_verify:=true \
  gemini_model:=gemini-3.6-flash save_snapshots:=true

# C) YOLOE (YOLO11 open-vocab)
ros2 launch vlm_pipeline_live pipeline_b_manual.launch.py \
  scene_name:=office_2_yoloe scene_type:=office \
  detector_backend:=yoloe gemini_verify:=false \
  save_snapshots:=true
```

Each run writes `/tmp/vlm_live_captures/<scene_name>/latest_scene_graph.json`.

### 3. Compare all three vs GT

```bash
ros2 run vlm_pipeline_live compare_backend_runs -- \
  --gt /home/docker/vla3d_data/Unity/office_2/office_2_scene_graph.json \
  --run grounding_dino:/tmp/vlm_live_captures/office_2_dino/latest_scene_graph.json \
  --run dino_gemini:/tmp/vlm_live_captures/office_2_dino_gemini/latest_scene_graph.json \
  --run yoloe:/tmp/vlm_live_captures/office_2_yoloe/latest_scene_graph.json \
  --out /tmp/vlm_backend_compare/office_2_backend_compare.json
```

Or print the guided script then compare:

```bash
bash /home/docker/ai_module/src/vlm_pipeline_live/scripts/run_backend_ab_experiment.sh
```

### GroundingDINO vs OWL-ViT (arabic_room explorer)

Use different `scene_name` values so graphs do not overwrite each other. OWL scores are usually lower than DINO — start at `box_threshold:=0.2`. First OWL run downloads `google/owlv2-base-patch16`.

```bash
# 1) GroundingDINO
ros2 launch vlm_pipeline_live pipeline_b.launch.py \
  scene_name:=arabic_room_dino scene_type:=arabic_room \
  detector_backend:=grounding_dino box_threshold:=0.35

# 2) OWL-ViT v2
ros2 launch vlm_pipeline_live pipeline_b.launch.py \
  scene_name:=arabic_room_owlvit scene_type:=arabic_room \
  detector_backend:=owlvit box_threshold:=0.2

# 3) Score both vs Unity GT
ros2 run vlm_pipeline_live compare_backend_runs -- \
  --gt /home/docker/vla3d_data/Unity/arabic_room/arabic_room_scene_graph.json \
  --run grounding_dino:/tmp/vlm_live_captures/arabic_room_dino/latest_scene_graph.json \
  --run owlvit:/tmp/vlm_live_captures/arabic_room_owlvit/latest_scene_graph.json \
  --out /tmp/arabic_room_dino_vs_owlvit.json
```

Helper that prints the same commands and compares if both graphs already exist:

```bash
bash /home/docker/ai_module/src/vlm_pipeline_live/scripts/compare_dino_owlvit.sh
```

**How to read the table:** higher **matched** + **labP/labR** is better; **extra** high means over-detecting; **meanXY** is mean distance of matched pairs (lower better).

---

## Tests

```bash
cd /home/docker/ai_module/src/vlm_pipeline_live
PYTHONPATH=../vlm_pipeline:$PWD:$PYTHONPATH python3 -m unittest \
  tests.test_label_utils \
  tests.test_lidar_fusion \
  tests.test_detection_vis \
  tests.test_capture_paths \
  tests.test_compare_scene_graphs \
  tests.test_live_scene_graph
```
