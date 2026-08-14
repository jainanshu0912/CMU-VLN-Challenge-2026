"""Build a VLA-3D-compatible scene graph from live 3D detections.

Relation heuristics follow HaochenZ11/VLA-3D ``scene_graph/generate_scene_info.py``
(and IRef-VLA, which reuses the same graph format), adapted for axis-aligned
boxes from LiDAR–camera fusion. Absolute distance thresholds match
Pipeline A's ``graph_search`` geometric fallback so live graphs work with
``GraphSearchMatcher`` / ``CountPipeline``.

Output schema matches ``vlm_pipeline.scene_loader.SceneData`` /
``*_scene_graph.json``:

  {
    "scene_name": "...",
    "regions": {
      "0": {
        "region_id": "0",
        "region_name": "live_map",
        "objects": [...],
        "relationships": {
          "above": {"0": ["3"], ...},
          "below": {...},
          "near": {...},
          "on": {...},
          "in": {...},
          "closest": {...},
          "farthest": {...},
          "between": {"5": [["2", "7"]], ...},
          ...
        }
      }
    }
  }
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from vlm_pipeline.scene_loader import (
  RELATION_TYPES,
  SceneData,
  SceneObject,
  SceneRegion,
)
from vlm_pipeline_live.lidar_camera_fusion import DetectedObject3D

# Absolute thresholds (Pipeline A / plan). VLA-3D uses region-relative near;
# live maps prefer meters.
NEAR_DISTANCE_M = 1.5
ON_CONTACT_TOLERANCE_M = 0.05
ON_PROXIMITY_TOLERANCE_M = 0.15
BETWEEN_PERP_TOLERANCE_M = 0.5
OVERLAP_EPSILON_M = 0.05
VERTICAL_IOM = 0.5  # VLA-3D --vertical_iom default
DEFAULT_REGION_ID = "0"
DEFAULT_REGION_NAME = "live_map"

# Label heuristics when NYU IDs are unavailable (live GroundingDINO labels).
IN_CONTAINER_LABELS = frozenset({
  "box", "basket", "bin", "cabinet", "drawer", "sink", "bathtub", "container",
  "refrigerator", "fridge", "microwave", "oven", "dishwasher", "nightstand",
  "night stand", "dresser",
})
SUPPORT_SURFACE_LABELS = frozenset({
  "table", "desk", "shelf", "shelves", "bookshelf", "cabinet", "sofa", "couch",
  "bed", "chair", "stool", "bench", "counter", "stand", "tv stand", "tv cabinet",
  "nightstand", "night stand", "dresser", "ottoman", "coffee table",
})
STRUCTURE_BLACKLIST_LABELS = frozenset({
  "wall", "floor", "ceiling", "door", "doorway", "door way", "door frame",
})


@dataclass(frozen=True)
class SceneGraphBuildResult:
  scene: SceneData
  graph_dict: dict
  num_objects: int
  num_relations: int


def detections_to_scene_objects(
  detections: Sequence[DetectedObject3D],
  region_id: str = DEFAULT_REGION_ID,
) -> list[SceneObject]:
  """Convert fused 3D detections into Pipeline A ``SceneObject`` nodes."""
  objects: list[SceneObject] = []
  for index, det in enumerate(detections):
    label = (det.label or "object").strip().lower()
    if not label:
      label = "object"
    objects.append(
      SceneObject(
        object_id=str(index),
        region_id=region_id,
        raw_label=label,
        nyu_id="-1",
        nyu40_id="-1",
        nyu_label=label,
        nyu40_label=label,
        cx=float(det.cx),
        cy=float(det.cy),
        cz=float(det.cz),
        x_length=max(float(det.x_length), 0.05),
        y_length=max(float(det.y_length), 0.05),
        z_length=max(float(det.z_length), 0.05),
        heading=float(det.heading),
        front_heading=None,
        color_schemes=[],
      )
    )
  return objects


class LiveSceneGraphBuilder:
  """Compute VLA-3D-style spatial relations from live 3D object boxes."""

  def __init__(
    self,
    *,
    near_distance_m: float = NEAR_DISTANCE_M,
    on_contact_tolerance_m: float = ON_CONTACT_TOLERANCE_M,
    on_proximity_tolerance_m: float = ON_PROXIMITY_TOLERANCE_M,
    between_perp_tolerance_m: float = BETWEEN_PERP_TOLERANCE_M,
    vertical_iom: float = VERTICAL_IOM,
    region_id: str = DEFAULT_REGION_ID,
    region_name: str = DEFAULT_REGION_NAME,
    scene_name: str = "live_scene",
  ) -> None:
    self.near_distance_m = float(near_distance_m)
    self.on_contact_tolerance_m = float(on_contact_tolerance_m)
    self.on_proximity_tolerance_m = float(on_proximity_tolerance_m)
    self.between_perp_tolerance_m = float(between_perp_tolerance_m)
    self.vertical_iom = float(vertical_iom)
    self.region_id = region_id
    self.region_name = region_name
    self.scene_name = scene_name

  def build_from_detections(
    self,
    detections: Sequence[DetectedObject3D],
  ) -> SceneGraphBuildResult:
    objects = detections_to_scene_objects(detections, region_id=self.region_id)
    return self.build_from_objects(objects)

  def build_from_objects(self, objects: Sequence[SceneObject]) -> SceneGraphBuildResult:
    relationships = self._compute_relationships(list(objects))
    object_ids = [obj.object_id for obj in objects]
    region = SceneRegion(
      region_id=self.region_id,
      region_name=self.region_name,
      object_ids=object_ids,
      relationships=relationships,
    )
    scene = SceneData(
      scene_name=self.scene_name,
      objects={obj.object_id: obj for obj in objects},
      regions={self.region_id: region},
      data_root=Path("."),
    )
    graph_dict = scene_data_to_vla3d_json(scene)
    return SceneGraphBuildResult(
      scene=scene,
      graph_dict=graph_dict,
      num_objects=len(objects),
      num_relations=_count_relations(relationships),
    )

  def _compute_relationships(
    self,
    objects: list[SceneObject],
  ) -> dict[str, dict[str, list]]:
    relations: dict[str, dict[str, list]] = {
      name: {obj.object_id: [] for obj in objects} for name in RELATION_TYPES
    }
    if not objects:
      return relations

    by_label: dict[str, list[SceneObject]] = defaultdict(list)
    for obj in objects:
      by_label[obj.raw_label.lower()].append(obj)

    for anchor in objects:
      aid = anchor.object_id
      for other in objects:
        if other.object_id == aid:
          continue
        if _is_structure_label(other) or _is_structure_label(anchor):
          # Still allow near/closest involving furniture; skip structure-as-target
          # for on/above/in when the *target* is a wall/floor/ceiling.
          pass

        if self._is_above(other, anchor) and not _is_structure_label(other):
          relations["above"][aid].append(other.object_id)
        if self._is_above(anchor, other) and not _is_structure_label(other):
          relations["below"][aid].append(other.object_id)
        if self._is_near(anchor, other):
          relations["near"][aid].append(other.object_id)
        if self._is_beside(anchor, other):
          relations["beside"][aid].append(other.object_id)
        if self._is_on(other, anchor) and not _is_structure_label(other):
          relations["on"][aid].append(other.object_id)
        if self._is_in(other, anchor) and not _is_structure_label(other):
          relations["in"][aid].append(other.object_id)

      # Closest / farthest: one ranking list per other label class (VLA-3D schema 2 → first).
      closest_ids: list[str] = []
      farthest_ids: list[str] = []
      for label, group in by_label.items():
        peers = [obj for obj in group if obj.object_id != aid]
        if not peers:
          continue
        ranked = sorted(peers, key=lambda obj: _xy_distance(anchor, obj))
        closest_ids.append(ranked[0].object_id)
        farthest_ids.append(ranked[-1].object_id)
      relations["closest"][aid] = closest_ids
      relations["farthest"][aid] = farthest_ids

      # Between: ternary pairs of other objects that sandwich the anchor.
      between_pairs: list[tuple[str, str]] = []
      for i, left in enumerate(objects):
        if left.object_id == aid:
          continue
        for right in objects[i + 1 :]:
          if right.object_id == aid:
            continue
          if self._is_between(anchor, left, right):
            between_pairs.append((left.object_id, right.object_id))
      relations["between"][aid] = between_pairs

      # Hanging-on: elevated object near a support, not resting on anything.
      for other in objects:
        if other.object_id == aid:
          continue
        if self._is_hanging_on(other, anchor, relations["on"], relations["in"]):
          relations["hanging_on"][aid].append(other.object_id)

    return relations

  def _xy_iom(self, a: SceneObject, b: SceneObject) -> float:
    return _axis_aligned_iom_xy(a, b)

  def _is_near(self, a: SceneObject, b: SceneObject) -> bool:
    return _xy_distance(a, b) < self.near_distance_m

  def _is_beside(self, a: SceneObject, b: SceneObject) -> bool:
    if not self._is_near(a, b):
      return False
    if self._is_above(a, b) or self._is_above(b, a):
      return False
    return self._xy_iom(a, b) < self.vertical_iom

  def _is_above(self, upper: SceneObject, lower: SceneObject) -> bool:
    if self._xy_iom(upper, lower) < self.vertical_iom:
      return False
    upper_bottom = upper.cz - upper.z_length / 2.0
    lower_top = lower.cz + lower.z_length / 2.0
    # VLA-3D: max_z_lower + on_thres <= min_z_upper, with XY IOM.
    return lower_top + OVERLAP_EPSILON_M <= upper_bottom or (
      upper.cz - lower.cz > (upper.z_length + lower.z_length) / 2.0 - OVERLAP_EPSILON_M
      and self._xy_iom(upper, lower) >= self.vertical_iom
    )

  def _is_on(self, upper: SceneObject, lower: SceneObject) -> bool:
    if _is_structure_label(lower):
      return False
    if not _looks_like_support(lower) and lower.raw_label.lower() not in SUPPORT_SURFACE_LABELS:
      # Still allow geometric on for unknown labels with contact + IOM.
      pass
    upper_bottom = upper.cz - upper.z_length / 2.0
    lower_top = lower.cz + lower.z_length / 2.0
    lower_bottom = lower.cz - lower.z_length / 2.0
    if abs(upper_bottom - lower_top) > self.on_proximity_tolerance_m:
      return False
    if upper_bottom < lower_bottom:
      return False
    if self._xy_iom(upper, lower) < self.vertical_iom:
      return False
    # Support should be larger in XY footprint (VLA-3D).
    if (lower.x_length * lower.y_length) < (upper.x_length * upper.y_length):
      return False
    return True

  def _is_in(self, inner: SceneObject, container: SceneObject) -> bool:
    if not _looks_like_container(container):
      return False
    if not (
      inner.cx >= container.cx - container.x_length / 2.0
      and inner.cx <= container.cx + container.x_length / 2.0
      and inner.cy >= container.cy - container.y_length / 2.0
      and inner.cy <= container.cy + container.y_length / 2.0
    ):
      return False
    if not (
      container.x_length > inner.x_length
      and container.y_length > inner.y_length
      and container.z_length > inner.z_length
    ):
      return False
    inner_top = inner.cz + inner.z_length / 2.0
    inner_bottom = inner.cz - inner.z_length / 2.0
    container_top = container.cz + container.z_length / 2.0
    container_bottom = container.cz - container.z_length / 2.0
    return inner_top < container_top and inner_bottom > container_bottom

  def _is_between(
    self,
    target: SceneObject,
    anchor_a: SceneObject,
    anchor_b: SceneObject,
  ) -> bool:
    ax, ay = anchor_a.cx, anchor_a.cy
    bx, by = anchor_b.cx, anchor_b.cy
    px, py = target.cx, target.cy
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-6:
      return False
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if t < 0.0 or t > 1.0:
      return False
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y) <= self.between_perp_tolerance_m

  def _is_hanging_on(
    self,
    target: SceneObject,
    anchor: SceneObject,
    on_map: dict[str, list],
    in_map: dict[str, list],
  ) -> bool:
    """Simplified hanging_on: elevated near anchor, not on/in anything."""
    if _is_structure_label(target):
      return False
    if _xy_distance(target, anchor) > self.near_distance_m:
      return False
    target_bottom = target.cz - target.z_length / 2.0
    anchor_bottom = anchor.cz - anchor.z_length / 2.0
    anchor_top = anchor.cz + anchor.z_length / 2.0
    if target_bottom <= anchor_bottom + 0.2:
      return False
    if target.cz + target.z_length / 2.0 >= anchor_top:
      return False
    for targets in on_map.values():
      if target.object_id in targets:
        return False
    for targets in in_map.values():
      if target.object_id in targets:
        return False
    return True


def scene_data_to_vla3d_json(scene: SceneData) -> dict:
  """Serialize ``SceneData`` to the VLA-3D / IRef-VLA scene_graph.json schema."""
  regions_out: dict = {}
  for region_id, region in scene.regions.items():
    objects_out = []
    for object_id in region.object_ids:
      obj = scene.objects.get(object_id)
      if obj is None:
        continue
      objects_out.append(_object_to_graph_dict(obj))

    relationships_out: dict = {}
    for relation, subject_map in region.relationships.items():
      rel_map: dict = {}
      for subject_id, targets in subject_map.items():
        if relation == "between":
          rel_map[subject_id] = [
            list(pair) if isinstance(pair, tuple) else pair for pair in targets
          ]
        else:
          rel_map[subject_id] = list(targets)
      relationships_out[relation] = rel_map

    regions_out[region_id] = {
      "region_id": region.region_id,
      "region_name": region.region_name,
      "region_bbox": _region_bbox_corners(
        [scene.objects[oid] for oid in region.object_ids if oid in scene.objects]
      ),
      "objects": objects_out,
      "relationships": relationships_out,
    }

  return {
    "scene_name": scene.scene_name,
    "regions": regions_out,
  }


def save_scene_graph_json(scene: SceneData, path: str | Path) -> Path:
  out = Path(path)
  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", encoding="utf-8") as handle:
    json.dump(scene_data_to_vla3d_json(scene), handle, indent=2)
  return out


def scene_graph_to_json_string(scene: SceneData) -> str:
  return json.dumps(scene_data_to_vla3d_json(scene))


def _object_to_graph_dict(obj: SceneObject) -> dict:
  return {
    "object_id": obj.object_id,
    "raw_label": obj.raw_label,
    "nyu_id": obj.nyu_id,
    "nyu40_id": obj.nyu40_id,
    "nyu_label": obj.nyu_label,
    "nyu40_label": obj.nyu40_label,
    "color_vals": [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
    "color_labels": ["N/A", "N/A", "N/A"],
    "color_percentages": ["0", "0", "0"],
    "bbox_center": [obj.cx, obj.cy, obj.cz],
    "bbox_size": [obj.x_length, obj.y_length, obj.z_length],
    "bbox_heading": obj.heading,
  }


def _region_bbox_corners(objects: Sequence[SceneObject]) -> list[list[float]]:
  if not objects:
    return [[0.0, 0.0, 0.0]] * 8
  xs, ys, zs = [], [], []
  for obj in objects:
    xs.extend([obj.cx - obj.x_length / 2.0, obj.cx + obj.x_length / 2.0])
    ys.extend([obj.cy - obj.y_length / 2.0, obj.cy + obj.y_length / 2.0])
    zs.extend([obj.cz - obj.z_length / 2.0, obj.cz + obj.z_length / 2.0])
  xmin, xmax = min(xs), max(xs)
  ymin, ymax = min(ys), max(ys)
  zmin, zmax = min(zs), max(zs)
  # Same 8-corner ordering style as VLA-3D samples (axis-aligned).
  return [
    [xmin, ymax, zmax],
    [xmax, ymax, zmax],
    [xmax, ymin, zmax],
    [xmin, ymin, zmax],
    [xmin, ymax, zmin],
    [xmax, ymax, zmin],
    [xmax, ymin, zmin],
    [xmin, ymin, zmin],
  ]


def _xy_distance(a: SceneObject, b: SceneObject) -> float:
  return math.hypot(a.cx - b.cx, a.cy - b.cy)


def _axis_aligned_iom_xy(a: SceneObject, b: SceneObject) -> float:
  ax0, ax1 = a.cx - a.x_length / 2.0, a.cx + a.x_length / 2.0
  ay0, ay1 = a.cy - a.y_length / 2.0, a.cy + a.y_length / 2.0
  bx0, bx1 = b.cx - b.x_length / 2.0, b.cx + b.x_length / 2.0
  by0, by1 = b.cy - b.y_length / 2.0, b.cy + b.y_length / 2.0
  ix0, iy0 = max(ax0, bx0), max(ay0, by0)
  ix1, iy1 = min(ax1, bx1), min(ay1, by1)
  inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
  if inter <= 0.0:
    return 0.0
  area_a = max(a.x_length * a.y_length, 1e-9)
  area_b = max(b.x_length * b.y_length, 1e-9)
  return inter / min(area_a, area_b)


def _is_structure_label(obj: SceneObject) -> bool:
  return obj.raw_label.lower() in STRUCTURE_BLACKLIST_LABELS


def _looks_like_container(obj: SceneObject) -> bool:
  label = obj.raw_label.lower()
  return any(token in label for token in IN_CONTAINER_LABELS)


def _looks_like_support(obj: SceneObject) -> bool:
  label = obj.raw_label.lower()
  return any(token in label for token in SUPPORT_SURFACE_LABELS)


def _count_relations(relationships: dict[str, dict[str, list]]) -> int:
  total = 0
  for subject_map in relationships.values():
    for targets in subject_map.values():
      total += len(targets)
  return total


def main() -> None:
  """Tiny offline sanity check with synthetic detections."""
  detections = [
    DetectedObject3D("table", 0.9, 0.0, 0.0, 0.4, 1.2, 0.8, 0.5, 0.0, 20, 0.0),
    DetectedObject3D("book", 0.8, 0.1, 0.0, 0.75, 0.2, 0.15, 0.05, 0.0, 10, 0.0),
    DetectedObject3D("chair", 0.85, 1.0, 0.0, 0.45, 0.5, 0.5, 0.9, 0.0, 15, 90.0),
    DetectedObject3D("lamp", 0.7, 0.0, 1.2, 0.9, 0.2, 0.2, 0.4, 0.0, 8, 180.0),
  ]
  result = LiveSceneGraphBuilder(scene_name="synthetic_live").build_from_detections(detections)
  print(
    f"objects={result.num_objects} relations={result.num_relations}\n"
    f"on={result.scene.regions['0'].relationships['on']}\n"
    f"near={result.scene.regions['0'].relationships['near']}"
  )


if __name__ == "__main__":
  main()
