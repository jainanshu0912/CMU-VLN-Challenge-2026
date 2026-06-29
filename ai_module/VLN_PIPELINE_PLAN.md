# CMU VLN Challenge 2026 — AI Module Pipeline Plan

**Team Plan | 9-Day Sprint (Jun 21 – Jun 29, 2026)**

---

## 1. Challenge Overview


| Item                    | Detail                                                       |
| ----------------------- | ------------------------------------------------------------ |
| Simulation deadline     | Aug 15, 2026                                                 |
| Real-robot stage        | After simulation round, real-world office environments       |
| Scenes                  | 18 Unity indoor scenes (15 training + 3 held-out test)       |
| Questions per scene     | 5 (1 numerical, 2 object reference, 2 instruction-following) |
| Time limit per question | 10 minutes (explore + answer combined)                       |
| AI module interface     | ROS 2 Jazzy, `/challenge_question` in → 3 topic types out    |


### Question Types and Scoring


| Type                             | Trigger                         | Output Topic              | Score                  |
| -------------------------------- | ------------------------------- | ------------------------- | ---------------------- |
| Numerical ("How many...")        | `std_msgs/Int32`                | `/numerical_response`     | 0 or 1 (exact)         |
| Object Reference ("Find the...") | `visualization_msgs/Marker`     | `/selected_object_marker` | 0–2 (IoU overlap)      |
| Instruction Following (path)     | `geometry_msgs/Pose2D` sequence | `/way_point_with_heading` | 0–6 (path constraints) |


**Sprint focus: Numerical + Object Reference (find + count). Instruction-following is a stub.**

---

## 2. Key Data Available


| Source                                               | What it provides                                                                                                       | Scenes covered                          |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `~/vla3d_data/Unity/<scene>/`                        | `object_result.csv` (bboxes + labels + colors), `scene_graph.json` (pre-computed spatial relations), `object_list.txt` | 15 training scenes                      |
| `object_list.txt` inside running simulator container | Object bounding boxes + labels for the current scene                                                                   | 1 scene at a time (read-only from host) |
| `/camera/image` ROS topic                            | 1920×640 equirectangular, 360° HFOV, 120° VFOV, 10 Hz                                                                  | Runtime only                            |
| `/registered_scan` ROS topic                         | Accumulated 3D LiDAR point cloud in map frame, 5 Hz                                                                    | Runtime only                            |
| `/terrain_map` ROS topic                             | Traversable area point cloud around vehicle, 5 Hz                                                                      | Runtime only                            |
| `/state_estimation` ROS topic                        | Robot pose (Odometry), 100–200 Hz                                                                                      | Runtime only                            |


**Critical constraint confirmed:** The AI module container has NO filesystem access to the simulator container and NO docker binary. The 3 test scenes' `object_list.txt` files cannot be pre-extracted. **Live detection is mandatory for test scenes and real-robot stage.**

---

## 3. Architecture — Two-Track Pipeline

```
                        ┌─────────────────────────────────────────────┐
                        │           vlm_pipeline ROS2 Node             │
/challenge_question ──► │                                              │
                        │  QuestionClassifier                          │
                        │     │ find      │ count    │ navigate        │
                        │     ▼           ▼          ▼                 │
                        │  LLM QueryParser           NavigateStub      │
                        │     │                                        │
                        │     ▼                                        │
                        │  GraphSearchMatcher ◄── SceneData            │
                        │                              ▲               │
                        │           ┌──────────────────┴──────────┐    │
                        │           │ Track A          │ Track B   │    │
                        │           │ (Static)         │ (Live)    │    │
                        └───────────┼──────────────────┼───────────┘    
                                    │                  │
/numerical_response ◄───────────────┘                  │
/selected_object_marker ◄──────────────────────────────┘
/way_point_with_heading ◄──────────────────────────────┘
```

### Track A — Static (15 Training Scenes)

- **Input**: Pre-loaded VLA-3D `object_result.csv` + `scene_graph.json`
- **Scene identification**: `scene_name` set via ROS launch parameter
- **Answer time**: ~2–5 s (just one LLM API call + graph search)
- **Applies to**: All 15 training scenes in simulation

### Track B — Live (3 Test Scenes + Real Robot)

- **Input**: `/camera/image` + `/registered_scan` ROS topics
- **Scene identification**: Not needed — builds scene graph from scratch
- **Answer time**: ~90 s explore + ~5 s answer
- **Applies to**: 3 held-out simulation test scenes + all real-robot environments

---

## 4. Track A Pipeline Detail

### 4.1 Question Classifier

Keyword routing (handles edge cases the dummy model misclassifies):

```
"How many..." / "Count..."          → numerical
"Find..." / "The [noun]..." / article-first → object_reference  
Everything else                     → instruction_following (stub)
```

Edge cases fixed:

- japanese_room: `"The lantern between the vase and the stone decoration..."` — starts with "The", not "Find"
- loft: `"The blue chair that is closest to the cup of coffee."` — same pattern

### 4.2 LLM Query Parser

Calls GPT-4o / Claude / Gemini with a structured extraction prompt.

**Output schema (JSON)**:

```json
{
  "question_type": "find",
  "target_class": "pillow",
  "anchors": [
    {"class": "book", "role": "anchor1"},
    {"class": "stool", "role": "anchor2"}
  ],
  "relation": "closest",
  "attributes": {
    "color": null,
    "size": null
  }
}
```

For count questions, the schema adds:

```json
{
  "question_type": "count",
  "target_class": "sofa",
  "attribute_filters": {"color": null, "size": null},
  "spatial_filter": {"relation": "below", "anchor": "window"}
}
```

### 4.3 Graph Search Matcher (Find)

Uses VLA-3D `scene_graph.json` structure:

```
regions → relationships → relation_type → {subject_id: [target_ids]}
```

Algorithm:

1. Filter all objects by `target_class` (label match against `raw_label`, `nyu_label`, `nyu40_label`)
2. For each candidate, check scene graph: does a relation edge exist from this candidate to an anchor of the right class?
3. If multiple candidates remain, apply attribute filters (color scheme from `object_color_scheme*` columns)
4. Return the single remaining candidate → publish its bbox as `visualization_msgs/Marker` (CUBE in `map` frame) + `Pose2D` waypoint to center

### 4.4 Count Pipeline

1. Filter `object_result.csv` by `target_class`
2. Apply attribute filters (color, size)
3. If spatial constraint exists, check scene graph for relation to anchor class
4. Return `len(remaining_objects)` as `std_msgs/Int32`

---

## 5. Track B Pipeline Detail

### 5.1 Exploration Phase (90 seconds)

`explorer.py` publishes 4 rotate-in-place waypoints in sequence:

```
(robot_x, robot_y, 0°) → (robot_x, robot_y, 90°) → 180° → 270°
```

Triggers detection when `/registered_scan` stabilizes (< 5% change over a 5 s window).

### 5.2 Equirectangular → Perspective Crops

The 360° camera produces equirectangular images that standard detectors can't process directly.

`equirect_to_perspective.py` extracts 4 perspective crops:

- Crop 0: 0°–90° (forward)
- Crop 1: 90°–180° (right)
- Crop 2: 180°–270° (rear)
- Crop 3: 270°–360° (left)

Each crop is ~960×640 with ~90° HFOV. An inverse pixel map is stored: `(crop_px, crop_py) → 3D unit ray direction in camera frame` — needed for LiDAR projection.

### 5.3 Object Detection (GroundingDINO)

Text prompt is derived from the question's relevant object classes:

```
Question: "Find the pillow closest to the book on the stool."
→ prompt: "pillow . book . stool"    (GroundingDINO period-separated format)
```

Run on all 4 crops. Merge detections back using the inverse pixel map. Deduplicate objects that appear in overlapping crop boundaries.

### 5.4 LiDAR-Camera Fusion (3D Bboxes)

Uses camera-LiDAR extrinsics from the system README.

For each 2D detection box:

1. Project accumulated `/registered_scan` points into the camera crop using extrinsics
2. Keep LiDAR points whose 2D projection falls inside the detection box
3. 3D bbox center = median of those points; dimensions = extent of the point cluster
4. Transform to `map` frame (already in map frame since we use `/registered_scan`)

### 5.5 Live Scene Graph (Spatial Relations)

`live_scene_graph.py` computes all VLA-3D relations geometrically from the detected 3D bboxes:


| Relation      | Geometric Rule                                                   |
| ------------- | ---------------------------------------------------------------- |
| above / below | `Δcz > (h1+h2)/2 − ε` AND horizontal bbox overlap                |
| on            | above + vertical contact gap < 5 cm                              |
| in            | XY bbox of target contained in anchor's XY bbox                  |
| near          | Euclidean center-to-center distance < 1.5 m                      |
| closest       | Minimum distance among all anchor-class instances                |
| farthest      | Maximum distance                                                 |
| between       | Perpendicular distance from target to anchor–anchor line < 0.5 m |


Produces a `SceneData` object with identical schema to Track A → feeds directly into `GraphSearchMatcher`.

---

## 6. Real Robot Stage

The second stage uses real-world office environments. Track B applies with these adaptations:

- **LiDAR noise**: Apply statistical outlier removal (Open3D `remove_statistical_outlier`) before fusion
- **GroundingDINO threshold**: Start at 0.3, tune down if misses; tune up if false positives
- **Larger spaces**: Explorer may need additional linear waypoints (forward, backward) beyond rotate-in-place
- **System topic differences**: Re-check `/camera/image` resolution and `/registered_scan` frame name against real-robot system README before submission

---

## 7. VLM Backend Architecture

Abstract interface — swap model by changing `config/pipeline.yaml`:

```python
class VLMBackend(ABC):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
```

Implementations: `OpenAIBackend` (GPT-4o), `ClaudeBackend` (claude-opus-4), `GeminiBackend` (gemini-2.5-pro)

Benchmarking plan: run all 45 find+count questions through all 3 backends on Day 8, compare:

- Parse accuracy (did the JSON come out correctly structured?)
- Match accuracy (did the correct object get selected?)
- Latency (API round-trip time)

---

## 8. Package File Structure

```
ai_module/src/vlm_pipeline/
├── package.xml                                  ← ROS2 Python package metadata
├── setup.py                                     ← ament Python install
├── setup.cfg
├── resource/vlm_pipeline
├── launch/
│   └── vlm_pipeline.launch.py                  ← scene_mode, scene_name, vla3d_data_root
├── config/
│   └── pipeline.yaml                           ← VLM backend, thresholds, camera params
├── vlm_pipeline/
│   ├── __init__.py
│   ├── main_node.py                            ← ROS2 node entry point  [EXISTS]
│   ├── question_classifier.py                  ← find/count/navigate routing  [EXISTS partial]
│   ├── scene_loader.py                         ← VLA-3D CSV+JSON → SceneData  [DONE]
│   ├── scene_graph_builder.py                  ← object_list.txt → SceneData (dev tool)
│   ├── query_parser.py                         ← LLM structured query extraction
│   ├── graph_search.py                         ← spatial constraint matching (Track A+B)
│   ├── object_matcher.py                       ← candidate ranking → 1 result
│   ├── count_pipeline.py                       ← counting pipeline
│   ├── vlm_backends/
│   │   ├── base.py                             ← abstract VLMBackend
│   │   ├── openai_backend.py
│   │   ├── claude_backend.py
│   │   └── gemini_backend.py
│   └── live/                                   ← Track B (test scenes + real robot)
│       ├── explorer.py                         ← rotate-in-place waypoint publisher
│       ├── equirect_to_perspective.py          ← 360° → 4 perspective crops
│       ├── live_detector.py                    ← GroundingDINO + LiDAR-Camera fusion
│       └── live_scene_graph.py                 ← 3D bboxes → SceneData
└── tests/
    ├── test_offline_find.py                    ← all 30 find questions offline
    ├── test_offline_count.py                   ← all 15 count questions offline
    └── test_live_smoke.py                      ← Track B smoke test
```

---

## 9. ROS Interface Summary

### Subscribed Topics


| Topic                 | Type                          | Used By                             |
| --------------------- | ----------------------------- | ----------------------------------- |
| `/challenge_question` | `std_msgs/msg/String`         | All — question entry point          |
| `/state_estimation`   | `nav_msgs/msg/Odometry`       | Explorer (robot pose for waypoints) |
| `/camera/image`       | `sensor_msgs/msg/Image`       | Track B — live detection            |
| `/registered_scan`    | `sensor_msgs/msg/PointCloud2` | Track B — LiDAR-Camera fusion       |


### Published Topics


| Topic                     | Type                            | For Question Type                                                   |
| ------------------------- | ------------------------------- | ------------------------------------------------------------------- |
| `/numerical_response`     | `std_msgs/msg/Int32`            | Numerical (count)                                                   |
| `/selected_object_marker` | `visualization_msgs/msg/Marker` | Object Reference (find)                                             |
| `/way_point_with_heading` | `geometry_msgs/msg/Pose2D`      | Object Reference (navigate to found object) + Instruction Following |


---

## 10. 9-Day Sprint Schedule


| Day | Date   | Focus                                   | Done When                                                                                          |
| --- | ------ | --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1   | Jun 21 | Package setup + SceneLoader             | `scene_loader.py` loads all 15 scenes without error ✅                                              |
| 2   | Jun 22 | QuestionClassifier + VLM backends       | All 75 questions classified correctly; all 3 API backends callable                                 |
| 3   | Jun 23 | LLM QueryParser                         | Structured JSON extracted correctly for all 30 find questions                                      |
| 4   | Jun 24 | GraphSearch + ObjectMatcher             | All 30 find questions matched to correct object offline                                            |
| 5   | Jun 25 | CountPipeline + offline tests           | All 15 count questions return correct integer; accuracy report generated                           |
| 6   | Jun 26 | ROS2 integration (Track A)              | Node runs in docker compose; all 15 training scenes answer correctly end-to-end                    |
| 7   | Jun 27 | Track B: equirect crops + live detector | GroundingDINO detects relevant objects in simulator 360° images; LiDAR fusion returns 3D positions |
| 8   | Jun 28 | Track B: live scene graph + integration | chinese_room find/count correct with Track B (no VLA-3D data); multi-VLM accuracy table            |
| 9   | Jun 29 | Timing + polish + submission prep       | Both tracks complete in < 10 min; Docker image rebuilt; README updated                             |


---

## 11. Task Checklist

### Core Pipeline (Track A)

- [x] `vlm_pipeline` ROS2 package skeleton (package.xml, setup.py, main_node.py)
- [x] `scene_loader.py` — parses VLA-3D CSV + JSON for all 15 training scenes
- [x] `question_classifier.py` — find/count/navigate routing with edge case fixes
- [x] `vlm_backends/base.py` + `openai_backend.py` + `claude_backend.py` + `gemini_backend.py`
- [x] `query_parser.py` — LLM prompt → structured query JSON
- [x] `graph_search.py` — spatial constraint matching
- [x] `object_matcher.py` — candidate ranking
- [x] `count_pipeline.py` — counting with filters
- [x] `tests/test_offline_find.py` + `test_offline_count.py`
- [x] `main_node.py` wired to all publishers/subscribers
- [x] End-to-end test: all 15 training scenes pass in docker compose

### Live Pipeline (Track B)

- [ ] `live/explorer.py` — rotate-in-place waypoint publisher
- [ ] `live/equirect_to_perspective.py` — 360° → 4 perspective crops
- [ ] `live/live_detector.py` — GroundingDINO + LiDAR-Camera fusion
- [ ] `live/live_scene_graph.py` — 3D bboxes → SceneData
- [ ] `main_node.py` scene_mode dispatch
- [ ] Smoke test: Track B on chinese_room without VLA-3D data

### Development Tools

- [ ] `scene_graph_builder.py` — object_list.txt → SceneData (offline validation tool)

### Submission

- [ ] Multi-VLM benchmark: GPT-4o vs Claude vs Gemini accuracy table
- [ ] Timing validation: both tracks complete in < 10 min
- [ ] `docker/Dockerfile` updated with new dependencies
- [ ] `docker/run.sh` updated
- [ ] README.md updated with usage instructions

---

## 12. Key Dependencies to Install

```bash
# VLM APIs
pip install openai anthropic google-generativeai

# Track B — Live Detection
pip install groundingdino-py          # or: pip install groundingdino
pip install open3d                    # for point cloud processing
pip install py360convert              # equirectangular projection
pip install numpy scipy opencv-python

# ROS2 Python
pip install transforms3d              # for extrinsic transforms
```

---

## 13. Critical Numbers to Know


| Parameter                       | Value         | Source                             |
| ------------------------------- | ------------- | ---------------------------------- |
| Camera image size               | 1920 × 640 px | Challenge README                   |
| Camera HFOV                     | 360°          | Challenge README                   |
| Camera VFOV                     | 120°          | Challenge README                   |
| Camera publish rate             | 10 Hz         | Challenge README                   |
| LiDAR scan rate                 | 5 Hz          | Challenge README                   |
| Time limit per question         | 10 min        | Challenge README                   |
| Training scenes                 | 15            | Challenge README                   |
| Test scenes (held out)          | 3             | Challenge README                   |
| VLA-3D near threshold           | ~1.5 m        | VLA-3D scene_graph code            |
| VLA-3D on contact threshold     | ~0.05 m       | VLA-3D scene_graph code            |
| GroundingDINO default threshold | 0.3           | Tunable in config.yaml             |
| Track A answer latency          | ~2–5 s        | Estimate (LLM API + graph search)  |
| Track B total time              | ~90–100 s     | Estimate (90s explore + detection) |


---

## 14. Reference Links

- [CMU VLN Challenge 2026](https://www.ai-meets-autonomy.com/cmu-vln-challenge)
- [Challenge GitHub Repo](https://github.com/Yuxin916/CMU-VLN-Challenge-2026)
- [VLA-3D Dataset](https://github.com/HaochenZ11/VLA-3D)
- [IRef-VLA Benchmark (ICRA 2025)](https://github.com/HaochenZ11/IRef-VLA)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

---

*Last updated: Jun 22, 2026*