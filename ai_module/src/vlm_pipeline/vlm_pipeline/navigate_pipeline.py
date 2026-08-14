"""Resolve instruction-following legs to map waypoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from vlm_pipeline.graph_search import GraphSearchMatcher, objects_matching_label, resolve_anchor_objects
from vlm_pipeline.navigate_parser import (
  BETWEEN_KIND,
  NEAR_PATH_KIND,
  OBJECT_KIND,
  NavLeg,
  ParsedNavigate,
)
from vlm_pipeline.query_parser import QueryParser
from vlm_pipeline.question_classifier import QuestionType
from vlm_pipeline.scene_loader import SceneData, SceneObject

WaypointXY = Tuple[float, float]


@dataclass
class ResolvedWaypoint:
  x: float
  y: float
  theta: float
  label: str
  object_id: Optional[str] = None
  kind: str = OBJECT_KIND


class NavigatePipeline:
  """Turn parsed navigate queries into ordered map waypoints."""

  def __init__(
    self,
    matcher: Optional[GraphSearchMatcher] = None,
    query_parser: Optional[QueryParser] = None,
    standoff_m: float = 0.8,
  ) -> None:
    self.matcher = matcher or GraphSearchMatcher()
    self.query_parser = query_parser or QueryParser(use_llm=False)
    self.standoff_m = standoff_m

  def resolve(
    self,
    scene: SceneData,
    parsed: ParsedNavigate,
    robot_x: float = 0.0,
    robot_y: float = 0.0,
  ) -> List[ResolvedWaypoint]:
    waypoints: List[ResolvedWaypoint] = []
    cur_x, cur_y = robot_x, robot_y

    for leg in parsed.legs:
      resolved = self._resolve_leg(scene, leg, cur_x, cur_y)
      if resolved is None:
        continue
      waypoints.append(resolved)
      cur_x, cur_y = resolved.x, resolved.y

    # Face toward the next waypoint (or keep last theta).
    for index in range(len(waypoints) - 1):
      nxt = waypoints[index + 1]
      waypoints[index].theta = math.atan2(nxt.y - waypoints[index].y, nxt.x - waypoints[index].x)

    return waypoints

  def _resolve_leg(
    self,
    scene: SceneData,
    leg: NavLeg,
    robot_x: float,
    robot_y: float,
  ) -> Optional[ResolvedWaypoint]:
    if leg.kind == BETWEEN_KIND:
      return self._resolve_between(scene, leg, robot_x, robot_y)
    if leg.kind in (OBJECT_KIND, NEAR_PATH_KIND):
      return self._resolve_object(scene, leg.phrase, robot_x, robot_y, kind=leg.kind)
    return None

  def _resolve_object(
    self,
    scene: SceneData,
    phrase: str,
    robot_x: float,
    robot_y: float,
    kind: str = OBJECT_KIND,
  ) -> Optional[ResolvedWaypoint]:
    if not phrase:
      return None

    # Prefer the full find parser (relations / attributes).
    find_question = f"Find the {phrase}"
    parsed = None
    match: Optional[SceneObject] = None
    try:
      parsed = self.query_parser.parse(find_question, QuestionType.FIND)
      match = self.matcher.find(scene, parsed)
    except Exception:
      match = None

    if match is None:
      anchors = resolve_anchor_objects(scene, phrase, pick_best=True)
      match = anchors[0] if anchors else None

    if match is None and parsed is not None and parsed.target_class:
      # Soft fallback: class match, prefer nearest to first anchor if any.
      pool = objects_matching_label(scene, parsed.target_class)
      if pool and parsed.anchors:
        refs: List[SceneObject] = []
        for anchor in parsed.anchors:
          refs.extend(resolve_anchor_objects(scene, anchor.class_name, pick_best=True))
        if refs:
          ref = refs[0]
          pool = sorted(
            pool,
            key=lambda obj: (obj.cx - ref.cx) ** 2 + (obj.cy - ref.cy) ** 2,
          )
      if pool:
        match = pool[0]

    if match is None:
      return None

    x, y = _standoff_xy(match, robot_x, robot_y, self.standoff_m)
    theta = math.atan2(match.cy - robot_y, match.cx - robot_x)
    return ResolvedWaypoint(
      x=x,
      y=y,
      theta=theta,
      label=match.raw_label,
      object_id=str(match.object_id),
      kind=kind,
    )

  def _resolve_between(
    self,
    scene: SceneData,
    leg: NavLeg,
    robot_x: float,
    robot_y: float,
  ) -> Optional[ResolvedWaypoint]:
    left = _pick_anchor(scene, leg.anchor1)
    right = _pick_anchor(scene, leg.anchor2, exclude_id=left.object_id if left else None)
    if left is None or right is None:
      return None

    # Midpoint between the two anchors.
    mx = 0.5 * (left.cx + right.cx)
    my = 0.5 * (left.cy + right.cy)
    theta = math.atan2(my - robot_y, mx - robot_x)
    return ResolvedWaypoint(
      x=mx,
      y=my,
      theta=theta,
      label=f"between({left.raw_label},{right.raw_label})",
      object_id=None,
      kind=BETWEEN_KIND,
    )


def _pick_anchor(
  scene: SceneData,
  phrase: str,
  exclude_id: Optional[str] = None,
) -> Optional[SceneObject]:
  if not phrase:
    return None

  candidates = resolve_anchor_objects(scene, phrase, pick_best=False)
  if not candidates:
    candidates = objects_matching_label(scene, phrase)
  if exclude_id is not None:
    filtered = [obj for obj in candidates if str(obj.object_id) != str(exclude_id)]
    if filtered:
      candidates = filtered
  if not candidates:
    return None
  return candidates[0]


def _standoff_xy(
  obj: SceneObject,
  robot_x: float,
  robot_y: float,
  standoff_m: float,
) -> WaypointXY:
  dx = robot_x - obj.cx
  dy = robot_y - obj.cy
  dist = math.hypot(dx, dy)
  if dist < 0.05:
    return obj.cx, obj.cy

  standoff = max(0.0, standoff_m)
  if dist <= standoff:
    return robot_x, robot_y

  scale = standoff / dist
  return obj.cx + dx * scale, obj.cy + dy * scale


def format_waypoint_summary(waypoints: Sequence[ResolvedWaypoint]) -> str:
  parts = []
  for index, wp in enumerate(waypoints):
    parts.append(
      f"{index + 1}:{wp.kind}:{wp.label}@({wp.x:.2f},{wp.y:.2f})"
    )
  return " -> ".join(parts)
