"""Load VLA-3D static scene data (object CSV + scene graph JSON) for Unity scenes."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

UNITY_SCENE_NAMES: Tuple[str, ...] = (
  "arabic_room",
  "chinese_room",
  "home_building_1",
  "home_building_2",
  "hotel_room_1",
  "hotel_room_2",
  "japanese_room",
  "livingroom_1",
  "livingroom_2",
  "livingroom_3",
  "livingroom_4",
  "loft",
  "office_1",
  "office_2",
  "studio",
)

RELATION_TYPES: Tuple[str, ...] = (
  "above",
  "below",
  "closest",
  "farthest",
  "between",
  "beside",
  "near",
  "in",
  "on",
  "hanging_on",
)

BetweenPair = Tuple[str, str]
RelationTarget = Union[str, BetweenPair]


def _parse_optional_float(value: str) -> Optional[float]:
  if not value or value == "_":
    return None
  return float(value)


def _parse_optional_str(value: str) -> Optional[str]:
  if not value or value == "_":
    return None
  return value


@dataclass(frozen=True)
class ColorScheme:
  red: Optional[float]
  green: Optional[float]
  blue: Optional[float]
  label: Optional[str]
  percentage: Optional[float]


@dataclass
class SceneObject:
  object_id: str
  region_id: str
  raw_label: str
  nyu_id: str
  nyu40_id: str
  nyu_label: str
  nyu40_label: str
  cx: float
  cy: float
  cz: float
  x_length: float
  y_length: float
  z_length: float
  heading: float
  front_heading: Optional[float]
  color_schemes: List[ColorScheme] = field(default_factory=list)

  @property
  def labels(self) -> Tuple[str, str, str]:
    return (self.raw_label, self.nyu_label, self.nyu40_label)

  def label_matches(self, query: str) -> bool:
    q = query.strip().lower()
    return any(label.lower() == q for label in self.labels)


@dataclass
class SceneRegion:
  region_id: str
  region_name: str
  object_ids: List[str] = field(default_factory=list)
  relationships: Dict[str, Dict[str, List[RelationTarget]]] = field(default_factory=dict)


@dataclass
class SceneData:
  scene_name: str
  objects: Dict[str, SceneObject]
  regions: Dict[str, SceneRegion]
  data_root: Path

  def get_object(self, object_id: str) -> Optional[SceneObject]:
    return self.objects.get(object_id)

  def objects_by_label(self, label: str) -> List[SceneObject]:
    return [obj for obj in self.objects.values() if obj.label_matches(label)]

  def objects_in_region(self, region_id: str) -> List[SceneObject]:
    region = self.regions.get(region_id)
    if region is None:
      return []
    return [self.objects[oid] for oid in region.object_ids if oid in self.objects]

  def region_for_object(self, object_id: str) -> Optional[SceneRegion]:
    obj = self.objects.get(object_id)
    if obj is None or obj.region_id == "-1":
      return None
    return self.regions.get(obj.region_id)

  def get_relations(
    self,
    subject_id: str,
    relation: str,
  ) -> List[RelationTarget]:
    region = self.region_for_object(subject_id)
    if region is None:
      return []
    relation_map = region.relationships.get(relation, {})
    return list(relation_map.get(subject_id, []))

  def related_object_ids(
    self,
    subject_id: str,
    relation: str,
  ) -> List[str]:
    targets = self.get_relations(subject_id, relation)
    ids: List[str] = []
    for target in targets:
      if isinstance(target, str):
        ids.append(target)
      else:
        ids.extend(target)
    return ids


class SceneLoader:
  """Parse VLA-3D Unity scene folders into structured SceneData."""

  def __init__(self, data_root: Optional[str] = None) -> None:
    self.data_root = Path(data_root or os.path.expanduser("~/vla3d_data/Unity"))

  def scene_dir(self, scene_name: str) -> Path:
    return self.data_root / scene_name

  def object_csv_path(self, scene_name: str) -> Path:
    return self.scene_dir(scene_name) / f"{scene_name}_object_result.csv"

  def scene_graph_path(self, scene_name: str) -> Path:
    return self.scene_dir(scene_name) / f"{scene_name}_scene_graph.json"

  def object_list_path(self, scene_name: str) -> Path:
    return self.scene_dir(scene_name) / "object_list.txt"

  def list_scenes(self) -> List[str]:
    return [
      scene_name
      for scene_name in UNITY_SCENE_NAMES
      if self.object_csv_path(scene_name).is_file()
      and self.scene_graph_path(scene_name).is_file()
    ]

  def load(self, scene_name: str) -> SceneData:
    csv_path = self.object_csv_path(scene_name)
    graph_path = self.scene_graph_path(scene_name)

    if not csv_path.is_file():
      raise FileNotFoundError(f"Missing object CSV for scene '{scene_name}': {csv_path}")
    if not graph_path.is_file():
      raise FileNotFoundError(f"Missing scene graph for scene '{scene_name}': {graph_path}")

    objects = self._parse_object_csv(csv_path)
    regions = self._parse_scene_graph(graph_path)

  # Ensure every scene-graph object is present in the CSV map.
    for region in regions.values():
      for object_id in region.object_ids:
        if object_id not in objects:
          raise ValueError(
            f"Scene graph object '{object_id}' missing from CSV for '{scene_name}'"
          )

    return SceneData(
      scene_name=scene_name,
      objects=objects,
      regions=regions,
      data_root=self.data_root,
    )

  def load_all(self) -> Dict[str, SceneData]:
    return {scene_name: self.load(scene_name) for scene_name in self.list_scenes()}

  def validate_coordinates(self, scene_name: str, tolerance: float = 1e-4) -> bool:
    """
    Compare object_bbox_c[xyz] in the CSV against object_list.txt when present.
    object_list.txt uses map-frame coordinates exported for the simulator.
    """
    object_list_path = self.object_list_path(scene_name)
    if not object_list_path.is_file():
      return True

    scene = self.load(scene_name)
    mismatches = 0

    with object_list_path.open("r", encoding="utf-8") as handle:
      for line_number, line in enumerate(handle):
        parts = line.strip().split()
        if len(parts) < 4:
          continue

        object_id = parts[0]
        expected_x = float(parts[1])
        expected_y = float(parts[2])
        expected_z = float(parts[3])
        obj = scene.objects.get(object_id)
        if obj is None:
          mismatches += 1
          continue

        if (
          abs(obj.cx - expected_x) > tolerance
          or abs(obj.cy - expected_y) > tolerance
          or abs(obj.cz - expected_z) > tolerance
        ):
          mismatches += 1

    if mismatches:
      raise ValueError(
        f"Coordinate mismatch for '{scene_name}': {mismatches} objects differ from object_list.txt"
      )
    return True

  def _parse_object_csv(self, csv_path: Path) -> Dict[str, SceneObject]:
    objects: Dict[str, SceneObject] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
      for row in csv.DictReader(handle):
        color_schemes: List[ColorScheme] = []
        for index in (1, 2, 3):
          color_schemes.append(
            ColorScheme(
              red=_parse_optional_float(row[f"object_color_r{index}"]),
              green=_parse_optional_float(row[f"object_color_g{index}"]),
              blue=_parse_optional_float(row[f"object_color_b{index}"]),
              label=_parse_optional_str(row[f"object_color_scheme{index}"]),
              percentage=_parse_optional_float(
                row[f"object_color_scheme_percentage{index}"]
              ),
            )
          )

        object_id = row["object_id"]
        objects[object_id] = SceneObject(
          object_id=object_id,
          region_id=row["region_id"],
          raw_label=row["raw_label"],
          nyu_id=row["nyu_id"],
          nyu40_id=row["nyu40_id"],
          nyu_label=row["nyu_label"],
          nyu40_label=row["nyu40_label"],
          cx=float(row["object_bbox_cx"]),
          cy=float(row["object_bbox_cy"]),
          cz=float(row["object_bbox_cz"]),
          x_length=float(row["object_bbox_xlength"]),
          y_length=float(row["object_bbox_ylength"]),
          z_length=float(row["object_bbox_zlength"]),
          heading=float(row["object_bbox_heading"]),
          front_heading=_parse_optional_float(row["object_front_heading"]),
          color_schemes=color_schemes,
        )

    return objects

  def _parse_scene_graph(self, graph_path: Path) -> Dict[str, SceneRegion]:
    with graph_path.open("r", encoding="utf-8") as handle:
      graph = json.load(handle)

    regions: Dict[str, SceneRegion] = {}

    for region_id, region_data in graph.get("regions", {}).items():
      object_ids = [obj["object_id"] for obj in region_data.get("objects", [])]
      relationships = self._parse_relationships(region_data.get("relationships", {}))
      regions[region_id] = SceneRegion(
        region_id=region_id,
        region_name=region_data.get("region_name", ""),
        object_ids=object_ids,
        relationships=relationships,
      )

    return regions

  def _parse_relationships(
    self,
    raw_relationships: Dict[str, Dict[str, Sequence[RelationTarget]]],
  ) -> Dict[str, Dict[str, List[RelationTarget]]]:
    parsed: Dict[str, Dict[str, List[RelationTarget]]] = {}

    for relation_type, subject_map in raw_relationships.items():
      parsed_subjects: Dict[str, List[RelationTarget]] = {}
      for subject_id, targets in subject_map.items():
        parsed_subjects[subject_id] = self._normalize_relation_targets(
          relation_type,
          targets,
        )
      parsed[relation_type] = parsed_subjects

    return parsed

  def _normalize_relation_targets(
    self,
    relation_type: str,
    targets: Sequence[RelationTarget],
  ) -> List[RelationTarget]:
    normalized: List[RelationTarget] = []

    for target in targets:
      if relation_type == "between":
        if isinstance(target, list) and len(target) == 2:
          normalized.append((str(target[0]), str(target[1])))
        continue

      if isinstance(target, str):
        normalized.append(target)
      elif isinstance(target, list) and target:
        normalized.append(str(target[0]))

    return normalized
