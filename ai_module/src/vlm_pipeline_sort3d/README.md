# Pipeline C — SORT3D-Style Spatial Reasoning

Independent ROS 2 package (`vlm_pipeline_sort3d`). Does **not** depend on Pipeline A (`vlm_pipeline`) or Pipeline B (`vlm_pipeline_live`), so you can build, test, and compare it in isolation.

Paper: [SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-Shot 3D Grounding Using LLMs](https://arxiv.org/abs/2504.18684) · [GitHub](https://github.com/nzantout/SORT3D)

---

## What this method actually is

Pipelines A/B treat language understanding as **parse → fixed matcher**:

```
question → JSON {target, relation, anchors} → one-shot graph/geometry search → answer
```

That breaks down on nested language (“bowl on the **table closest to** the screen”), ordinals (“2nd closest”), and view-dependent refs (“to the left of…”).

**SORT3D flips the split of labor:**

| Role | Who does it |
|------|-------------|
| Language + planning | LLM (chain-of-thought) |
| Geometry / spatial predicates | Deterministic **toolbox** functions |
| Appearance attributes | Short text captions (VLA-3D labels offline, VLM crops live) |

The LLM never invents 3D math. It writes a short program of tool calls; each tool returns object IDs (or a count / waypoint); the LLM chains the next call on those IDs.

```
Question
   │
   ▼
① Object filter     LLM extracts nouns → keep only relevant objects
   │
   ▼
② Captioner         (optional) VLM crop → "The pillow is red, soft..."
   │
   ▼
③ Toolbox reasoner  LLM ↔ SpatialToolbox loop until finish_find / finish_count
   │
   ▼
Marker / Int32 / Pose2D   (same challenge topics as A/B)
```

### Example (from the paper style)

> Find the bowl on the table closest to the folding screen.

```
screens = find_all("folding screen")     → [3]
tables  = find_all("table")              → [5, 8]
t       = find_closest(tables, screens)  → [8]
bowls   = find_all("bowl")               → [1, 4]
result  = find_on(bowls, t)              → [1]
finish_find("1")
```

Nested “on X closest to Y” becomes **two tools**, not one overloaded parser rule.

### Why it can beat Pipeline A

- **Compositional** relations are natural (tool chaining).
- **View-dependent** left/right use robot pose.
- **Ordinals** via `order_*` then index.
- **Attributes** can use rich captions, not only color_scheme enums.
- Still **zero-shot** — no text–3D training; one in-context example is enough.

Latency cost: ~2 LLM round-trips (filter + reasoner steps) vs A’s single parse.

---

## Package layout

```
vlm_pipeline_sort3d/
├── scene_inventory.py     # Object list loader (VLA-3D CSV → InventoryObject)
├── object_filter.py       # Stage 1 — noun extract + relevance filter
├── object_captioner.py    # Stage 2 — VLM captions (stub for live)
├── spatial_toolbox.py     # Geometry tools (implemented)
├── toolbox_reasoner.py    # Stage 3 — LLM tool-call loop + parser
├── question_classifier.py # FIND / COUNT / NAVIGATE
└── main_node.py           # ROS node (LLM backend wiring still TODO)
```

No dependency on `vlm_pipeline` or `vlm_pipeline_live` in `package.xml`.

---

## Status

| Module | Status |
|--------|--------|
| `scene_inventory.py` | Loads VLA-3D `object_result.csv` into SORT3D inventory |
| `spatial_toolbox.py` | Core tools implemented (near/on/above/below/between/closest/…) |
| `object_filter.py` | Rule-based nouns + optional LLM JSON filter |
| `toolbox_reasoner.py` | Prompt, tool-call parser, execution loop |
| `object_captioner.py` | Stub (static captions from CSV for now) |
| `main_node.py` | ROS I/O wired; **LLM backend not plugged in yet** |
| Offline A-vs-C benchmark | Not started |

---

## Build

```bash
cd ai_module
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vlm_pipeline_sort3d
source install/setup.bash
```

## Run toolbox unit tests (no ROS / no LLM)

```bash
cd ai_module/src/vlm_pipeline_sort3d
PYTHONPATH=. python3 -m pytest tests/test_spatial_toolbox.py -q
```

## Launch (after LLM is wired)

```bash
ros2 launch vlm_pipeline_sort3d pipeline_c.launch.py scene_name:=chinese_room
```

---

## Next steps

1. Plug an LLM backend into `ObjectFilter` + `ToolboxReasoner` (Ollama / OpenAI / … — copy pattern from Pipeline A backends *without* importing that package, or vendor a thin client here).
2. Offline eval on all 45 find+count training questions vs Pipeline A.
3. Optional: accept a live object list JSON from Pipeline B later, still without hard-coupling packages.
l