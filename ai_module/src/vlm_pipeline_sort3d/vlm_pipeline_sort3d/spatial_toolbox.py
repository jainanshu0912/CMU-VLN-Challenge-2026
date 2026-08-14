"""Heuristic spatial toolbox — geometry only, no LLM.

This is the core idea behind SORT3D: the LLM plans *which* spatial operations
to run, but every geometric predicate is a deterministic function of 3D boxes.

Each tool takes object IDs (or class names) and returns object IDs (or a count /
waypoint). The reasoner executes calls sequentially so nested relations become
a chain of tool results.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from vlm_pipeline_sort3d.scene_inventory import InventoryObject, SceneInventory


@dataclass
class ToolboxConfig:
  near_threshold_m: float = 1.5
  on_contact_threshold_m: float = 0.05
  between_threshold_m: float = 0.5
  overlap_epsilon_m: float = 0.02
  standoff_m: float = 0.8


@dataclass
class Waypoint:
  x: float
  y: float
  yaw: float


def _norm(text: str) -> str:
  return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _xy_overlap(a: InventoryObject, b: InventoryObject, eps: float) -> bool:
  ax0, ax1 = a.cx - a.x_length / 2, a.cx + a.x_length / 2
  ay0, ay1 = a.cy - a.y_length / 2, a.cy + a.y_length / 2
  bx0, bx1 = b.cx - b.x_length / 2, b.cx + b.x_length / 2
  by0, by1 = b.cy - b.y_length / 2, b.cy + b.y_length / 2
  return ax0 < bx1 + eps and ax1 + eps > bx0 and ay0 < by1 + eps and ay1 + eps > by0


def _xy_contained(inner: InventoryObject, outer: InventoryObject, eps: float) -> bool:
  ix0, ix1 = inner.cx - inner.x_length / 2, inner.cx + inner.x_length / 2
  iy0, iy1 = inner.cy - inner.y_length / 2, inner.cy + inner.y_length / 2
  ox0, ox1 = outer.cx - outer.x_length / 2, outer.cx + outer.x_length / 2
  oy0, oy1 = outer.cy - outer.y_length / 2, outer.cy + outer.y_length / 2
  return ix0 >= ox0 - eps and ix1 <= ox1 + eps and iy0 >= oy0 - eps and iy1 <= oy1 + eps


def _dist(a: InventoryObject, b: InventoryObject) -> float:
  return math.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2 + (a.cz - b.cz) ** 2)


def _dist_xy(a: InventoryObject, b: InventoryObject) -> float:
  return math.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2)


def _perp_dist_to_segment(
  px: float, py: float,
  ax: float, ay: float,
  bx: float, by: float,
) -> float:
  abx, aby = bx - ax, by - ay
  apx, apy = px - ax, py - ay
  ab2 = abx * abx + aby * aby
  if ab2 < 1e-12:
    return math.sqrt(apx * apx + apy * apy)
  t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
  qx, qy = ax + t * abx, ay + t * aby
  return math.sqrt((px - qx) ** 2 + (py - qy) ** 2)


class SpatialToolbox:
  """Deterministic spatial operators used by the LLM reasoner."""

  def __init__(
    self,
    inventory: SceneInventory,
    config: Optional[ToolboxConfig] = None,
    robot_pose: Optional[Tuple[float, float, float]] = None,
  ) -> None:
    self.inventory = inventory
    self.config = config or ToolboxConfig()
    self.robot_pose = robot_pose  # (x, y, yaw)
    self._by_id: Dict[str, InventoryObject] = inventory.by_id()

  def _resolve(self, object_ids: Sequence[str]) -> List[InventoryObject]:
    out: List[InventoryObject] = []
    for oid in object_ids:
      obj = self._by_id.get(str(oid))
      if obj is not None:
        out.append(obj)
    return out

  def _matches_class(self, obj: InventoryObject, class_name: str) -> bool:
    needle = _norm(class_name)
    if not needle:
      return False
    candidates = (obj.name,) + obj.aliases
    for cand in candidates:
      hay = _norm(cand)
      if needle == hay or needle in hay or hay in needle:
        return True
    return False

  # ── catalog tools ────────────────────────────────────────────────────────

  def find_all(self, class_name: str) -> List[str]:
    """Return all object IDs whose label matches ``class_name``."""
    return [
      obj.object_id
      for obj in self.inventory.objects
      if self._matches_class(obj, class_name)
    ]

  def count(self, object_ids: Sequence[str]) -> int:
    return len(self._resolve(object_ids))

  # ── relation tools ───────────────────────────────────────────────────────

  def find_near(self, targets: Sequence[str], anchors: Sequence[str]) -> List[str]:
    anchor_objs = self._resolve(anchors)
    if not anchor_objs:
      return []
    thr = self.config.near_threshold_m
    hits: List[str] = []
    for t in self._resolve(targets):
      if any(_dist(t, a) < thr for a in anchor_objs):
        hits.append(t.object_id)
    return hits

  def find_on(self, targets: Sequence[str], anchors: Sequence[str]) -> List[str]:
    """Target rests on an anchor: above + vertical contact + XY overlap."""
    anchor_objs = self._resolve(anchors)
    if not anchor_objs:
      return []
    contact = self.config.on_contact_threshold_m
    eps = self.config.overlap_epsilon_m
    hits: List[str] = []
    for t in self._resolve(targets):
      t_bottom = t.cz - t.z_length / 2
      for a in anchor_objs:
        a_top = a.cz + a.z_length / 2
        if t_bottom + contact >= a_top and t_bottom - contact <= a_top + contact:
          if _xy_overlap(t, a, eps):
            hits.append(t.object_id)
            break
    return hits

  def find_above(self, targets: Sequence[str], anchors: Sequence[str]) -> List[str]:
    anchor_objs = self._resolve(anchors)
    if not anchor_objs:
      return []
    eps = self.config.overlap_epsilon_m
    hits: List[str] = []
    for t in self._resolve(targets):
      for a in anchor_objs:
        if t.cz > a.cz and _xy_overlap(t, a, eps):
          hits.append(t.object_id)
          break
    return hits

  def find_below(self, targets: Sequence[str], anchors: Sequence[str]) -> List[str]:
    anchor_objs = self._resolve(anchors)
    if not anchor_objs:
      return []
    eps = self.config.overlap_epsilon_m
    hits: List[str] = []
    for t in self._resolve(targets):
      for a in anchor_objs:
        if t.cz < a.cz and _xy_overlap(t, a, eps):
          hits.append(t.object_id)
          break
    return hits

  def find_between(
    self,
    targets: Sequence[str],
    anchor_a: Sequence[str],
    anchor_b: Sequence[str],
  ) -> List[str]:
    a_objs = self._resolve(anchor_a)
    b_objs = self._resolve(anchor_b)
    if not a_objs or not b_objs:
      return []
    thr = self.config.between_threshold_m
    hits: List[str] = []
    for t in self._resolve(targets):
      ok = False
      for a in a_objs:
        for b in b_objs:
          d = _perp_dist_to_segment(t.cx, t.cy, a.cx, a.cy, b.cx, b.cy)
          if d < thr:
            ok = True
            break
        if ok:
          break
      if ok:
        hits.append(t.object_id)
    return hits

  def find_closest(self, targets: Sequence[str], references: Sequence[str]) -> List[str]:
    refs = self._resolve(references)
    tgts = self._resolve(targets)
    if not refs or not tgts:
      return []
    best = min(tgts, key=lambda t: min(_dist(t, r) for r in refs))
    return [best.object_id]

  def find_farthest(self, targets: Sequence[str], references: Sequence[str]) -> List[str]:
    refs = self._resolve(references)
    tgts = self._resolve(targets)
    if not refs or not tgts:
      return []
    best = max(tgts, key=lambda t: min(_dist(t, r) for r in refs))
    return [best.object_id]

  def find_left(
    self,
    targets: Sequence[str],
    anchors: Sequence[str],
    robot_pose: Optional[Tuple[float, float, float]] = None,
  ) -> List[str]:
    """View-dependent: target is to the left of anchor from robot yaw."""
    return self._find_side(targets, anchors, side="left", robot_pose=robot_pose)

  def find_right(
    self,
    targets: Sequence[str],
    anchors: Sequence[str],
    robot_pose: Optional[Tuple[float, float, float]] = None,
  ) -> List[str]:
    return self._find_side(targets, anchors, side="right", robot_pose=robot_pose)

  def _find_side(
    self,
    targets: Sequence[str],
    anchors: Sequence[str],
    side: str,
    robot_pose: Optional[Tuple[float, float, float]],
  ) -> List[str]:
    pose = robot_pose or self.robot_pose
    if pose is None:
      return []
    _, _, yaw = pose
    # Robot forward in map XY; left is +90° from forward.
    fx, fy = math.cos(yaw), math.sin(yaw)
    lx, ly = -fy, fx
    anchor_objs = self._resolve(anchors)
    if not anchor_objs:
      return []
    hits: List[str] = []
    for t in self._resolve(targets):
      for a in anchor_objs:
        dx, dy = t.cx - a.cx, t.cy - a.cy
        lateral = dx * lx + dy * ly
        if side == "left" and lateral > 0:
          hits.append(t.object_id)
          break
        if side == "right" and lateral < 0:
          hits.append(t.object_id)
          break
    return hits

  # ── ordering tools ───────────────────────────────────────────────────────

  def order_bottom_to_top(self, targets: Sequence[str]) -> List[str]:
    objs = self._resolve(targets)
    objs.sort(key=lambda o: o.cz)
    return [o.object_id for o in objs]

  def order_smallest_to_largest(self, targets: Sequence[str]) -> List[str]:
    objs = self._resolve(targets)
    objs.sort(key=lambda o: o.volume())
    return [o.object_id for o in objs]

  def order_closest_to_farthest(
    self,
    targets: Sequence[str],
    references: Sequence[str],
  ) -> List[str]:
    refs = self._resolve(references)
    objs = self._resolve(targets)
    if not refs:
      return [o.object_id for o in objs]
    objs.sort(key=lambda t: min(_dist(t, r) for r in refs))
    return [o.object_id for o in objs]

  # ── navigation helpers ───────────────────────────────────────────────────

  def go_near(self, object_id: str) -> Optional[Waypoint]:
    obj = self._by_id.get(str(object_id))
    if obj is None:
      return None
    pose = self.robot_pose
    if pose is None:
      # Approach from -Y by default.
      return Waypoint(x=obj.cx, y=obj.cy - self.config.standoff_m, yaw=math.pi / 2)
    rx, ry, _ = pose
    dx, dy = obj.cx - rx, obj.cy - ry
    dist = math.sqrt(dx * dx + dy * dy) or 1.0
    ux, uy = dx / dist, dy / dist
    return Waypoint(
      x=obj.cx - ux * self.config.standoff_m,
      y=obj.cy - uy * self.config.standoff_m,
      yaw=math.atan2(uy, ux),
    )

  def go_between(self, id1: str, id2: str) -> Optional[Waypoint]:
    a = self._by_id.get(str(id1))
    b = self._by_id.get(str(id2))
    if a is None or b is None:
      return None
    mx, my = (a.cx + b.cx) / 2, (a.cy + b.cy) / 2
    yaw = math.atan2(b.cy - a.cy, b.cx - a.cx)
    return Waypoint(x=mx, y=my, yaw=yaw)

  # ── dispatch for the reasoner ────────────────────────────────────────────

  TOOL_NAMES = (
    "find_all",
    "find_near",
    "find_on",
    "find_above",
    "find_below",
    "find_between",
    "find_closest",
    "find_farthest",
    "find_left",
    "find_right",
    "order_bottom_to_top",
    "order_smallest_to_largest",
    "order_closest_to_farthest",
    "count",
    "go_near",
    "go_between",
  )

  def call(self, name: str, *args, **kwargs):
    if name not in self.TOOL_NAMES:
      raise KeyError(f"Unknown toolbox function: {name}")
    return getattr(self, name)(*args, **kwargs)
