# CMU VLN Challenge 2026 — AI Module Pipeline Plan

> **Three parallel pipelines for find (object reference) and count (numerical) questions.**
> Pipeline A handles the 15 training scenes using pre-built VLA-3D data.
> Pipeline B handles the 3 held-out simulation test scenes and the real-robot stage using live sensor data.
> Pipeline C (SORT3D-style) is a stronger reasoning engine that can replace Pipeline A's query layer for both static and live scenes.

---

## 1. Challenge Overview


| Item                           | Detail                                                            |
| ------------------------------ | ----------------------------------------------------------------- |
| Simulation submission deadline | Aug 15, 2026                                                      |
| Real-robot stage               | After simulation round — real-world office environments           |
| Training scenes                | 15 Unity indoor scenes (full VLA-3D data available)               |
| Test scenes (held out)         | 3 Unity scenes — no pre-loaded data, must use live sensing        |
| Questions per scene            | 5 total: 1 numerical, 2 object reference, 2 instruction-following |
| Time limit per question        | 10 minutes (exploration + answering combined)                     |
| AI module interface            | ROS 2 Jazzy — `/challenge_question` in → 3 output topics          |


### Question Types and Scoring


| Type                  | Trigger Pattern                 | Output Topic                         | Score                            |
| --------------------- | ------------------------------- | ------------------------------------ | -------------------------------- |
| Numerical             | "How many..." / "Count..."      | `/numerical_response` (`Int32`)      | 0 or 1 (exact match)             |
| Object Reference      | "Find the..." / "The [noun]..." | `/selected_object_marker` (`Marker`) | 0–2 (IoU with ground-truth bbox) |
| Instruction Following | Path/trajectory commands        | `/way_point_with_heading` (`Pose2D`) | 0–6 (path constraints, order)    |


**Sprint focus: Numerical + Object Reference (find + count). Instruction-following is a stub.**

---

## 2. Confirmed Data and Constraints


| Source                        | Contents                                                                                                               | Available for           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `~/vla3d_data/Unity/<scene>/` | `object_result.csv` (bboxes + labels + colors), `scene_graph.json` (pre-computed spatial relations), `object_list.txt` | 15 training scenes only |
| `/camera/image` ROS topic     | 1920×640 equirectangular, 360° HFOV, 120° VFOV, 10 Hz                                                                  | Runtime (all scenes)    |
| `/registered_scan` ROS topic  | Accumulated 3D LiDAR point cloud in `map` frame, 5 Hz                                                                  | Runtime (all scenes)    |
| `/terrain_map` ROS topic      | Traversable area point cloud around robot, 5 Hz                                                                        | Runtime (all scenes)    |
| `/state_estimation` ROS topic | Robot pose (Odometry), 100–200 Hz                                                                                      | Runtime (all scenes)    |


**Confirmed constraint:** The AI module container cannot read the simulator container's filesystem and has no Docker binary. The 3 test scenes' `object_list.txt` files are inaccessible before evaluation. Live sensing is mandatory for test scenes and the entire real-robot stage.

---

## 3. ROS Interface (All Pipelines Share This)

### Subscribed Topics


| Topic                 | Type                      | Used by                                      |
| --------------------- | ------------------------- | -------------------------------------------- |
| `/challenge_question` | `std_msgs/String`         | All pipelines — question entry point         |
| `/state_estimation`   | `nav_msgs/Odometry`       | Pose tracking, standoff waypoint calculation |
| `/camera/image`       | `sensor_msgs/Image`       | Pipelines B and C (live sensing)             |
| `/registered_scan`    | `sensor_msgs/PointCloud2` | Pipelines B and C (LiDAR-Camera fusion)      |


### Published Topics


| Topic                     | Type                        | For question type                                              |
| ------------------------- | --------------------------- | -------------------------------------------------------------- |
| `/numerical_response`     | `std_msgs/Int32`            | Numerical (count)                                              |
| `/selected_object_marker` | `visualization_msgs/Marker` | Object Reference (find)                                        |
| `/way_point_with_heading` | `geometry_msgs/Pose2D`      | Object Reference (nav to found object) + Instruction Following |


---

## 4. Pipeline A — Static (15 Training Scenes)

### When it applies

All 15 training scenes where VLA-3D data is pre-loaded. Scene name is set via the `scene_name` ROS launch parameter.

### Architecture

```
/challenge_question
        │
        ▼
QuestionClassifier
  ├── "How many..." → CountPipeline
  └── "Find the..." / "The [noun]..." → FindPipeline
              │
              ▼
        QueryParser
     (rule-based + LLM JSON fallback)
              │
              ▼
     GraphSearchMatcher
              │
        SceneLoader
    (VLA-3D CSV + JSON)
              │
   ┌──────────┴──────────┐
   ▼                     ▼
Marker               Int32
/selected_object_marker  /numerical_response
+
Pose2D
/way_point_with_heading
```

### Stage 1 — Question Classification

Keyword routing (fixes edge cases the dummy model misclassifies):


| Input pattern                                        | Classified as     |
| ---------------------------------------------------- | ----------------- |
| Starts with "How many" / "Count"                     | `COUNT`           |
| Starts with "Find" / "The [noun]..." / article-first | `FIND`            |
| Everything else                                      | `NAVIGATE` (stub) |


Edge cases handled:

- `japanese_room`: `"The lantern between the vase and the stone decoration..."` — starts with "The", not "Find"
- `loft`: `"The blue chair that is closest to the cup of coffee."` — same pattern

### Stage 2 — Query Parsing

**Rule-based** (no API needed, always available):

- Extracts target class, relation, anchor(s), color/size attributes via regex patterns
- Handles: above, below, closest, farthest, between, near, on, under, with

**LLM fallback** (optional, activated via `use_llm_parser: true`):

- Calls configured backend with a one-shot JSON schema prompt
- Output: `{target_class, anchors: [{class, role}], relation, attributes: {color, size}}`
- Falls back to rule-based on API failure

### Stage 3 — Graph Search (Find)

Uses VLA-3D `scene_graph.json` + `object_result.csv`:

1. Filter candidates by `target_class` (fuzzy label matching across `raw_label`, `nyu_label`, `nyu40_label` with synonym expansion)
2. Check scene graph edges: does a relation edge exist from this candidate to the anchor class?
3. Geometric fallback: compute relation geometrically if not in scene graph
4. Attribute tie-break: filter by color scheme or size label
5. Return single best candidate → publish `Marker` (CUBE, map frame) + `Pose2D` standoff waypoint

**Spatial relation matching — both graph-based and geometric fallback:**


| Relation           | Scene graph lookup                | Geometric fallback                                   |
| ------------------ | --------------------------------- | ---------------------------------------------------- |
| above / below      | `above` / `below` edges           | Z-center comparison + horizontal bbox overlap        |
| on                 | `on` edges                        | Z-contact within 15 cm + XY proximity                |
| in                 | `in` edges                        | XY bbox containment                                  |
| near               | `near` / `beside` edges           | Euclidean center distance < 1.5 m                    |
| closest / farthest | `closest` / `farthest` rank lists | Distance ranking                                     |
| between            | `between` pair edges              | Perpendicular distance to anchor–anchor line < 0.5 m |


### Stage 3 — Count Pipeline (Numerical)

1. Filter `object_result.csv` by `target_class` (same fuzzy matching)
2. Apply attribute filters (color scheme, size label)
3. Apply spatial constraint via scene graph (e.g., "below a window" → check `below` edges)
4. Return `len(matching_objects)` as `Int32`

### Components Built


| File                     | Status | Description                                       |
| ------------------------ | ------ | ------------------------------------------------- |
| `scene_loader.py`        | ✅ Done | Parses VLA-3D CSV + JSON → `SceneData`            |
| `question_classifier.py` | ✅ Done | FIND / COUNT / NAVIGATE routing                   |
| `query_parser.py`        | ✅ Done | Rule-based + LLM JSON fallback                    |
| `graph_search.py`        | ✅ Done | Full geometric + graph-based matching (587 lines) |
| `count_pipeline.py`      | ✅ Done | Counting with attribute + spatial filters         |
| `main_node.py`           | ✅ Done | Full ROS2 node, all publishers/subscribers wired  |
| `vlm_backends/`          | ✅ Done | Ollama, OpenAI, Claude, Gemini backends           |


---

## 5. Pipeline B — Live (3 Test Scenes + Real Robot)

### When it applies

- 3 held-out simulation test scenes (no pre-loaded data available at evaluation time)
- Entire real-robot stage (real-world office environments)

### Architecture

```
On startup (90s exploration phase):
  Explorer publishes 4 rotate-in-place waypoints (0° / 90° / 180° / 270°)
         │
         ├── /registered_scan accumulates → full scene point cloud
         │
         └── /camera/image frames →
               equirect_to_perspective (4×90° crops)
                     │
               GroundingDINO (open-vocab detection)
                     │
               LiDAR-Camera Fusion (2D bbox → 3D bbox in map frame)
                     │
               live_scene_graph.py (compute spatial relations from 3D bboxes)
                     │
               SceneData object (identical schema to VLA-3D)
                     │
               [same GraphSearchMatcher / CountPipeline as Pipeline A]
```

### Stage 1 — Exploration

`explorer.py` publishes 4 rotate-in-place waypoints sequentially:

- Each waypoint: `(robot_x, robot_y, heading)` — same XY position, different heading
- Headings: 0°, 90°, 180°, 270°
- Triggers detection pass when `/registered_scan` stabilises (< 5% change in 5 s window)

### Stage 2 — Equirectangular → Perspective Crops

The 360° camera produces equirectangular images that standard detectors cannot process directly.

`equirect_to_perspective.py`:

- Extracts 4 perspective crops, each ~90° HFOV, ~640×640 px
- Stores inverse pixel map: `(crop_px, crop_py) → 3D unit ray in camera frame`
- Used by LiDAR-Camera fusion to back-project detections

Library: `py360convert` or manual spherical projection math.

### Stage 3 — Object Detection (GroundingDINO)

Text prompt derived from the incoming question:

```
Question: "Find the pillow closest to the book on the stool."
→ prompt: "pillow . book . stool"     (GroundingDINO period-separated class list)
```

Run on all 4 crops. Merge detections using the inverse pixel map. Deduplicate objects appearing at crop boundaries (NMS in map-frame 3D space).

### Stage 4 — LiDAR-Camera Fusion (3D Bboxes)

Uses camera-LiDAR extrinsics from the system README.

For each 2D detection box:

1. Project accumulated `/registered_scan` (map frame) into crop image via extrinsics
2. Keep LiDAR points whose 2D projection falls inside the detection box
3. 3D bbox center = median of those points; bbox dimensions = point cluster extent
4. Result is already in map frame (registered scan is map-frame)

### Stage 5 — Live Scene Graph

`live_scene_graph.py` computes spatial relations geometrically from detected 3D bboxes using VLA-3D geometry thresholds:


| Relation      | Rule                                                             |
| ------------- | ---------------------------------------------------------------- |
| above / below | `Δcz > (h1+h2)/2 − ε` AND horizontal bbox overlap                |
| on            | above + vertical contact gap < 5 cm                              |
| in            | XY bbox of target contained in anchor XY bbox                    |
| near          | Euclidean center-to-center distance < 1.5 m                      |
| closest       | Minimum distance among all anchor-class instances                |
| farthest      | Maximum distance                                                 |
| between       | Perpendicular distance from target to anchor–anchor line < 0.5 m |


Output: `SceneData` object with identical schema to VLA-3D output from `scene_loader.py`.
This feeds directly into the same `GraphSearchMatcher` and `CountPipeline` from Pipeline A.

### Real Robot Adaptations

The live pipeline applies to the real-robot stage with these tuning adjustments:


| Concern                     | Adjustment                                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Real LiDAR noise            | Apply `remove_statistical_outlier` (Open3D) before fusion                                                       |
| Lighting/material variation | Lower GroundingDINO confidence threshold to 0.25–0.35                                                           |
| Larger spaces               | Add forward exploration waypoints beyond rotate-in-place                                                        |
| System topic changes        | Re-check `/camera/image` resolution + `/registered_scan` frame name against real-robot README before submission |


### Components to Build


| File                              | Status        | Description                                                    |
| --------------------------------- | ------------- | -------------------------------------------------------------- |
| `live/explorer.py`                | ❌ Not started | Rotate-in-place waypoint publisher, scan stabilisation trigger |
| `live/equirect_to_perspective.py` | ❌ Not started | 360° → 4 perspective crops + inverse pixel map                 |
| `live/live_detector.py`           | ❌ Not started | GroundingDINO on crops + LiDAR-Camera fusion → 3D bboxes       |
| `live/live_scene_graph.py`        | ❌ Not started | 3D bboxes → spatial relations → SceneData                      |
| `main_node.py` (update)           | ❌ Pending     | Add `scene_mode` param dispatch (static vs live)               |


---

## 6. Pipeline C — SORT3D-Style (LLM Spatial Toolbox)

> Based on: [SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using LLMs](https://arxiv.org/abs/2504.18684) (IROS 2025, CMU)
>
> Key insight: instead of parsing the question into a rigid JSON schema and running a fixed graph search, the LLM performs **chain-of-thought reasoning** and calls heuristic spatial functions step by step. The LLM handles language; the toolbox handles geometry.

### When it applies

Pipeline C is a drop-in upgrade to the reasoning layer of Pipelines A and B. The scene data source (VLA-3D static vs. live detected) stays the same. Only the query parsing and matching steps change.

### Architecture

```
/challenge_question
        │
        ▼
QuestionClassifier (unchanged)
        │
        ▼
Stage 1: LLM Object Filter
  Extract object nouns → filter relevant objects from scene inventory
        │
        ▼
Stage 2: VLM Captioner (Track B / real robot only)
  Crop camera image per object → Qwen2-VL-7B → "The pillow is red, soft, square..."
  (Track A: use VLA-3D color_scheme + raw_label as simplified caption)
        │
        ▼
Stage 3: LLM Spatial Toolbox Reasoner
  LLM receives:
    - Question
    - Filtered object list with IDs, names, positions, sizes, captions
    - One in-context example of toolbox usage
  LLM outputs chain-of-thought + sequential toolbox function calls
        │
        ▼
  Spatial Toolbox (heuristic functions — geometry only, no LLM):
    find_all(class)            → [obj_ids]
    find_near(targets, anchor) → [obj_ids]
    find_on(targets, anchor)   → [obj_ids]
    find_above(targets, anchor)→ [obj_ids]
    find_below(targets, anchor)→ [obj_ids]
    find_between(t, a1, a2)    → [obj_ids]
    find_closest(targets, ref) → [obj_id]
    find_farthest(targets, ref)→ [obj_id]
    order_bottom_to_top(t)     → [obj_ids]  (for "2nd closest" etc.)
    order_smallest_to_largest(t)→ [obj_ids]
    find_left(t, anchor, pose) → [obj_ids]  (view-dependent)
    find_right(t, anchor, pose)→ [obj_ids]  (view-dependent)
    count(objects)             → int
    go_near(obj_id)            → Pose2D waypoint
    go_between(id1, id2)       → Pose2D waypoint
        │
        ▼
  Executed result → object ID or count
        │
   ┌────┴────┐
   ▼         ▼
 Marker   Int32
```

### Example Reasoning Trace

```
Question: "Find the bowl on the table closest to the folding screen."

LLM chain-of-thought:
  screens = find_all("folding screen")          # → [obj_3]
  tables  = find_all("table")                   # → [obj_5, obj_8, obj_12]
  t       = find_closest(tables, screens)       # → [obj_8]  (table closest to screen)
  bowls   = find_all("bowl")                    # → [obj_1, obj_4]
  result  = find_on(bowls, t)                   # → [obj_1]  (bowl on that table)
  answer  = result[0]                           # → obj_1

Action: go_near(obj_1) → publish Marker + Pose2D
```

```
Question: "How many computer monitors are on the table closest to the map wall decal?"

LLM chain-of-thought:
  decals  = find_all("map wall decal")          # → [obj_15]
  tables  = find_all("table")                   # → [obj_2, obj_7, obj_9]
  t       = find_closest(tables, decals)        # → [obj_9]
  monitors= find_all("computer monitor")        # → [obj_10, obj_11, obj_12]
  on_t    = find_on(monitors, t)               # → [obj_10, obj_11]
  answer  = count(on_t)                        # → 2

Action: publish Int32(2)
```

### Why This Is Stronger Than Pipelines A and B


| Capability                  | Pipeline A/B              | Pipeline C                                       |
| --------------------------- | ------------------------- | ------------------------------------------------ |
| Nested spatial chains       | Partial (single relation) | Natural (chained tool calls)                     |
| View-dependent (left/right) | Not handled               | `find_left/right` with robot pose                |
| Complex attribute matching  | Color scheme only         | VLM captions: color, material, shape, affordance |
| Explainability              | Black-box graph traversal | Full chain-of-thought visible                    |
| Handles "2nd closest" etc.  | Not handled               | `order_bottom_to_top`                            |
| Training data needed        | None                      | None (one in-context example)                    |


### What Pipeline C Reuses (Nothing Is Thrown Away)


| Existing component                   | Role in Pipeline C                                       |
| ------------------------------------ | -------------------------------------------------------- |
| `scene_loader.py`                    | Unchanged — provides object list for static scenes       |
| `graph_search.py` geometry functions | Become the spatial toolbox implementations               |
| `question_classifier.py`             | Unchanged                                                |
| `main_node.py` ROS publishers        | Unchanged — same output topics                           |
| `vlm_backends/`                      | Used for both LLM filter call and toolbox reasoning call |
| Live pipeline (B) `live_detector.py` | Unchanged — provides object list for live scenes         |


### VLM Captioning (Stage 2, Track B Only)

- Model: `Qwen2.5-VL-Instruct-3B` (quantized, runs on 4090)
- Input: cropped image of each detected object (best CLIP-similarity viewpoint)
- Prompt: `"Describe the <object> in this image using color, material, shape, affordances. Format: 'The <object> is <color>, <material>, <shape>'"`
- Output appended to object representation before LLM reasoning

For Track A (static scenes): use VLA-3D `raw_label` + `color_scheme` label as simplified caption — no VLM needed.

### Components to Build


| File                         | Status        | Description                                                       |
| ---------------------------- | ------------- | ----------------------------------------------------------------- |
| `sort3d/spatial_toolbox.py`  | ❌ Not started | Standalone geometry functions (refactored from `graph_search.py`) |
| `sort3d/toolbox_reasoner.py` | ❌ Not started | LLM chain-of-thought with tool execution loop                     |
| `sort3d/object_filter.py`    | ❌ Not started | LLM call #1: extract nouns, filter relevant objects               |
| `sort3d/object_captioner.py` | ❌ Not started | Qwen2-VL crop captioner (Track B / real robot only)               |
| `main_node.py` (update)      | ❌ Pending     | Add `reasoning_mode` param (`pipeline_a` vs `sort3d`)             |


---

## 7. Pipeline Comparison Summary


|                     | Pipeline A                 | Pipeline B                           | Pipeline C                                |
| ------------------- | -------------------------- | ------------------------------------ | ----------------------------------------- |
| Scene data source   | VLA-3D static (pre-loaded) | Live sensors (GroundingDINO + LiDAR) | Either (same reasoning layer)             |
| Query understanding | Rule-based + LLM JSON      | Same as A                            | LLM chain-of-thought + toolbox            |
| Object attributes   | VLA-3D color scheme        | LiDAR cluster extent                 | VLM captions (richer)                     |
| Spatial reasoning   | Graph search (single-pass) | Online graph                         | Sequential toolbox calls (chainable)      |
| Answer latency      | ~2–5 s                     | ~90 s explore + ~5 s answer          | ~3–8 s (extra LLM filter call)            |
| Applies to          | 15 training scenes         | 3 test scenes + real robot           | All scenes (replaces A/B reasoning layer) |
| Status              | **Largely complete**       | Not started                          | Not started                               |


---

## 8. Full File Structure

```
ai_module/src/vlm_pipeline/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/vlm_pipeline
├── launch/
│   └── vlm_pipeline.launch.py          ← scene_mode, scene_name, reasoning_mode params
├── config/
│   └── pipeline.yaml                   ← VLM backend, thresholds, camera intrinsics
├── vlm_pipeline/
│   ├── __init__.py
│   │
│   ├── # ── SHARED (all pipelines) ─────────────────────────────────────────
│   ├── main_node.py                    ✅ ROS2 node entry point
│   ├── question_classifier.py          ✅ FIND / COUNT / NAVIGATE routing
│   │
│   ├── # ── PIPELINE A (static) ────────────────────────────────────────────
│   ├── scene_loader.py                 ✅ VLA-3D CSV + JSON → SceneData
│   ├── query_parser.py                 ✅ Rule-based + LLM JSON fallback
│   ├── graph_search.py                 ✅ Spatial constraint matching
│   ├── object_matcher.py               ✅ Candidate ranking
│   ├── count_pipeline.py               ✅ Counting with filters
│   │
│   ├── # ── VLM BACKENDS (shared) ──────────────────────────────────────────
│   └── vlm_backends/
│       ├── __init__.py                 ✅
│       ├── base.py                     ✅ Abstract VlmBackend
│       ├── ollama_backend.py           ✅ Local (default, no API key)
│       ├── openai_backend.py           ✅ GPT-4o / GPT-4o-mini
│       ├── claude_backend.py           ✅ Claude Opus / Sonnet
│       └── gemini_backend.py           ✅ Gemini Flash / Pro
│
│   ├── # ── PIPELINE B (live) ──────────────────────────────────────────────
│   └── live/
│       ├── __init__.py                 ❌
│       ├── explorer.py                 ❌ Rotate-in-place waypoint publisher
│       ├── equirect_to_perspective.py  ❌ 360° → 4 perspective crops
│       ├── live_detector.py            ❌ GroundingDINO + LiDAR-Camera fusion
│       └── live_scene_graph.py         ❌ 3D bboxes → SceneData
│
│   ├── # ── PIPELINE C (SORT3D) ────────────────────────────────────────────
│   └── sort3d/
│       ├── __init__.py                 ❌
│       ├── spatial_toolbox.py          ❌ Standalone geometry functions
│       ├── toolbox_reasoner.py         ❌ LLM tool-calling loop
│       ├── object_filter.py            ❌ LLM noun extraction + relevance filter
│       └── object_captioner.py         ❌ Qwen2-VL crop captioner
│
└── tests/
    ├── test_offline_find.py            ❌ All 30 find questions offline
    ├── test_offline_count.py           ❌ All 15 count questions offline
    ├── test_live_smoke.py              ❌ Pipeline B on chinese_room w/o VLA-3D data
    └── test_sort3d_reasoning.py        ❌ Pipeline C on all 45 questions
```

---

## 9. Task Checklist

### Pipeline A — Static

- [x] `scene_loader.py` — parse VLA-3D CSV + JSON for all 15 scenes
- [x] `question_classifier.py` — FIND / COUNT / NAVIGATE with edge case fixes
- [x] `query_parser.py` — rule-based parser + LLM JSON fallback
- [x] `graph_search.py` — full geometric + graph-based spatial matching
- [x] `count_pipeline.py` — counting with attribute + spatial filters
- [x] `main_node.py` — ROS2 node with all topic publishers and subscribers
- [x] `vlm_backends/` — Ollama, OpenAI, Claude, Gemini
- [x] `tests/test_offline_find.py` — run all 30 find questions, report accuracy
- [x] `tests/test_offline_count.py` — run all 15 count questions, report accuracy
- [x] End-to-end test in docker compose — all 15 training scenes pass
- [ ] Multi-VLM accuracy benchmark — GPT-4o vs Claude vs Gemini vs Ollama table

### Pipeline B — Live

- [ ] `live/explorer.py` — 4 rotate-in-place waypoints, scan stabilisation signal
- [ ] `live/equirect_to_perspective.py` — equirectangular → 4×90° perspective crops + inverse pixel map
- [ ] `live/live_detector.py` — GroundingDINO on crops + LiDAR-Camera fusion → 3D bboxes in map frame
- [ ] `live/live_scene_graph.py` — 3D bboxes → spatial relations → SceneData
- [ ] `main_node.py` update — `scene_mode` param dispatch (static / live)
- [ ] `launch/vlm_pipeline.launch.py` update — expose `scene_mode` param
- [ ] `tests/test_live_smoke.py` — Pipeline B on chinese_room without VLA-3D data
- [ ] End-to-end test in simulator — live mode answers correctly for ≥ 2 scenes
- [ ] Real-robot tuning — LiDAR outlier removal, GroundingDINO threshold tuning

### Pipeline C — SORT3D

- [ ] `sort3d/spatial_toolbox.py` — refactor geometry functions from `graph_search.py` as standalone callables
- [ ] `sort3d/object_filter.py` — LLM call #1: extract nouns + filter relevant object IDs
- [ ] `sort3d/toolbox_reasoner.py` — LLM chain-of-thought + execute returned function calls
- [ ] `sort3d/object_captioner.py` — Qwen2-VL-7B crop captioner (Track B / real robot only)
- [ ] `main_node.py` update — `reasoning_mode` param (`pipeline_a` vs `sort3d`)
- [ ] In-context example — write the single few-shot toolbox usage example for the LLM prompt
- [ ] `tests/test_sort3d_reasoning.py` — run all 45 find+count questions through Pipeline C
- [ ] Accuracy comparison — Pipeline A vs Pipeline C on all 45 training questions

### Submission

- [ ] `docker/Dockerfile` — add GroundingDINO, Open3D, py360convert, Qwen2-VL dependencies
- [ ] Verify all three pipelines complete within 10-minute time limit
- [ ] `docker/run.sh` — expose `scene_mode` and `reasoning_mode` env vars
- [ ] Final Docker image build and smoke test
- [ ] GitHub submission — push ai_module, update README with usage instructions

---

## 10. Key Dependencies

```bash
# Pipeline A (already needed, lightweight)
pip install openai anthropic google-generativeai ollama

# Pipeline B — Live Detection
pip install groundingdino-py          # open-vocabulary object detection
pip install open3d                    # point cloud processing + outlier removal
pip install py360convert              # equirectangular projection
pip install numpy scipy opencv-python

# Pipeline C — SORT3D reasoning
pip install qwen-vl                   # or: pip install transformers accelerate
pip install transformers accelerate   # for Qwen2-VL captioner
pip install torch torchvision         # GPU inference

# ROS2 Python utilities
pip install transforms3d              # 3D rotation math for extrinsics
```

---

## 11. Critical Numbers


| Parameter                       | Value                   | Source                   |
| ------------------------------- | ----------------------- | ------------------------ |
| Camera image size               | 1920 × 640 px           | Challenge README         |
| Camera HFOV                     | 360°                    | Challenge README         |
| Camera VFOV                     | 120°                    | Challenge README         |
| Camera rate                     | 10 Hz                   | Challenge README         |
| LiDAR scan rate                 | 5 Hz                    | Challenge README         |
| Time limit per question         | 10 minutes              | Challenge README         |
| VLA-3D near threshold           | ~1.5 m                  | VLA-3D scene_graph code  |
| VLA-3D on contact threshold     | ~0.05 m                 | VLA-3D scene_graph code  |
| GroundingDINO default threshold | 0.3 (sim) / 0.25 (real) | Tunable                  |
| Pipeline A answer latency       | ~2–5 s                  | LLM API + graph search   |
| Pipeline B total time           | ~90–100 s               | 90 s explore + detection |
| Pipeline C answer latency       | ~3–8 s                  | 2 LLM calls + toolbox    |


---

## 12. Reference Links


| Resource                | Link                                                                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| CMU VLN Challenge 2026  | [ai-meets-autonomy.com/cmu-vln-challenge](https://www.ai-meets-autonomy.com/cmu-vln-challenge)                                                 |
| Challenge GitHub        | [github.com/Yuxin916/CMU-VLN-Challenge-2026](https://github.com/Yuxin916/CMU-VLN-Challenge-2026)                                               |
| VLA-3D Dataset          | [github.com/HaochenZ11/VLA-3D](https://github.com/HaochenZ11/VLA-3D)                                                                           |
| IRef-VLA Benchmark      | [github.com/HaochenZ11/IRef-VLA](https://github.com/HaochenZ11/IRef-VLA)                                                                       |
| SORT3D Paper (arXiv)    | [arxiv.org/abs/2504.18684](https://arxiv.org/abs/2504.18684)                                                                                   |
| SORT3D GitHub           | [github.com/nzantout/SORT3D](https://github.com/nzantout/SORT3D)                                                                               |
| GroundingDINO           | [github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)                                                       |
| Qwen2-VL                | [huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)                                                   |
| Semantic Mapping Module | [github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar](https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar) |


---

*Last updated: Jul 3, 2026*