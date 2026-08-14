"""Minimal object inventory for Pipeline C (independent of Pipeline A's SceneData).

SORT3D does not need a pre-built scene graph. It only needs an object-centric
inventory: id, name, 3D center, size, and an optional text caption. The LLM
then calls geometry tools against this list.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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


@dataclass
class InventoryObject:
  """One object in the SORT3D inventory."""

  object_id: str
  name: str
  cx: float
  cy: float
  cz: float
  x_length: float
  y_length: float
  z_length: float
  caption: str = ""
  aliases: Tuple[str, ...] = ()

  @property
  def center(self) -> Tuple[float, float, float]:
    return (self.cx, self.cy, self.cz)

  @property
  def size(self) -> Tuple[float, float, float]:
    return (self.x_length, self.y_length, self.z_length)

  def volume(self) -> float:
    return max(self.x_length, 1e-6) * max(self.y_length, 1e-6) * max(self.z_length, 1e-6)

  def display_name(self) -> str:
    return self.caption or self.name


@dataclass
class SceneInventory:
  """Object list the LLM spatial toolbox operates on."""

  scene_name: str
  objects: List[InventoryObject] = field(default_factory=list)

  def by_id(self) -> Dict[str, InventoryObject]:
    return {obj.object_id: obj for obj in self.objects}

  def ids(self) -> List[str]:
    return [obj.object_id for obj in self.objects]

  def filter_by_ids(self, object_ids: Sequence[str]) -> "SceneInventory":
    keep = set(object_ids)
    return SceneInventory(
      scene_name=self.scene_name,
      objects=[obj for obj in self.objects if obj.object_id in keep],
    )


def _parse_float(value: str, default: float = 0.0) -> float:
  if not value or value == "_":
    return default
  return float(value)


def _default_vla3d_root() -> Path:
  env = os.environ.get("VLA3D_ROOT")
  if env:
    return Path(env).expanduser()
  return Path.home() / "vla3d_data" / "Unity"


def load_vla3d_inventory(
  scene_name: str,
  vla3d_root: Optional[str | Path] = None,
) -> SceneInventory:
  """Load a SORT3D inventory from a VLA-3D `object_result.csv` (Track A offline).

  Captions are a simple string from raw_label + color_scheme — no VLM required.
  """
  root = Path(vla3d_root) if vla3d_root else _default_vla3d_root()
  csv_path = root / scene_name / "object_result.csv"
  if not csv_path.is_file():
    raise FileNotFoundError(f"VLA-3D object CSV not found: {csv_path}")

  objects: List[InventoryObject] = []
  with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
      object_id = row.get("object_id") or row.get("id") or ""
      if not object_id:
        continue
      raw = (row.get("raw_label") or "").strip()
      nyu = (row.get("nyu_label") or "").strip()
      nyu40 = (row.get("nyu40_label") or "").strip()
      name = raw or nyu or nyu40 or "object"
      color = (row.get("color_scheme") or row.get("color_label") or "").strip()
      if color and color != "_":
        caption = f"The {name} is {color}."
      else:
        caption = f"The {name}."

      aliases = tuple(a for a in (raw, nyu, nyu40) if a and a != "_")
      objects.append(
        InventoryObject(
          object_id=str(object_id),
          name=name,
          cx=_parse_float(row.get("cx", "0")),
          cy=_parse_float(row.get("cy", "0")),
          cz=_parse_float(row.get("cz", "0")),
          x_length=_parse_float(row.get("x_length", "0.1"), 0.1),
          y_length=_parse_float(row.get("y_length", "0.1"), 0.1),
          z_length=_parse_float(row.get("z_length", "0.1"), 0.1),
          caption=caption,
          aliases=aliases,
        )
      )

  return SceneInventory(scene_name=scene_name, objects=objects)


def inventory_prompt_block(inventory: SceneInventory) -> str:
  """Compact text block listing objects for the LLM prompt."""
  lines = []
  for obj in inventory.objects:
    lines.append(
      f"- id={obj.object_id} name={obj.name!r} "
      f"center=({obj.cx:.2f},{obj.cy:.2f},{obj.cz:.2f}) "
      f"size=({obj.x_length:.2f},{obj.y_length:.2f},{obj.z_length:.2f}) "
      f"caption={obj.display_name()!r}"
    )
  return "\n".join(lines)
