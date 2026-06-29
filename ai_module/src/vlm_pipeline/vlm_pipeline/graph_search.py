"""Match parsed queries against VLA-3D scene graphs."""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Set

from vlm_pipeline.query_parser import (
  AttributeFilters,
  ParsedQuery,
  _clean_phrase,
  _split_by_relation,
)
from vlm_pipeline.scene_loader import SceneData, SceneObject

NEAR_DISTANCE_M = 1.5
ON_CONTACT_TOLERANCE_M = 0.05
ON_PROXIMITY_TOLERANCE_M = 0.15
BETWEEN_PERP_TOLERANCE_M = 0.5
OVERLAP_EPSILON_M = 0.05

THAT_IS_CLOSEST_RE = re.compile(r"^(.*?)\s+that is closest to (.+)$", re.IGNORECASE)
THAT_IS_FARTHEST_RE = re.compile(r"^(.*?)\s+that is farthest from (.+)$", re.IGNORECASE)
WITH_ON_IT_RE = re.compile(r"^(?:a|an|the)\s+(.+?)\s+on it$", re.IGNORECASE)

COLOR_ALIASES: Dict[str, Set[str]] = {
  "red": {"red", "maroon", "crimson"},
  "blue": {"blue", "navy"},
  "black": {"black"},
  "white": {"white"},
  "brown": {"brown"},
  "green": {"green"},
  "gray": {"gray", "grey"},
  "yellow": {"yellow"},
}

LABEL_SYNONYMS: Dict[str, List[str]] = {
  "sofa": ["sofa", "couch"],
  "couch": ["sofa", "couch"],
  "cup of coffee": ["coffee cup", "cup of coffee", "cup"],
  "coffee cup": ["coffee cup", "cup of coffee", "cup"],
  "tv cabinet": ["tv cabinet", "tv stand"],
  "tv stand": ["tv cabinet", "tv stand"],
  "potted plant": ["potted plant", "plant"],
  "plant": ["potted plant", "plant"],
  "speaker": ["speaker"],
}

SPATIAL_RELATIONS = frozenset({
  "below", "above", "near", "on", "beside", "in", "hanging_on", "with",
})


class GraphSearchMatcher:
  """Filter scene objects by class, spatial relations, and attributes."""

  def find(self, scene: SceneData, query: ParsedQuery) -> Optional[SceneObject]:
    candidates = self.filter_objects(scene, query)
    if not candidates:
      return None
    if len(candidates) == 1:
      return candidates[0]
    return _tie_break(scene, candidates, query)

  def count(self, scene: SceneData, query: ParsedQuery) -> int:
    from vlm_pipeline.count_pipeline import CountPipeline

    return CountPipeline().count(scene, query)

  def filter_objects(self, scene: SceneData, query: ParsedQuery) -> List[SceneObject]:
    if query.question_type == "count":
      from vlm_pipeline.count_pipeline import CountPipeline

      return CountPipeline().filter_objects(scene, query)
    return _filter_find(scene, query)


def objects_matching_label(
  scene: SceneData,
  label_phrase: str,
  min_score: float = 0.45,
) -> List[SceneObject]:
  scored: List[tuple[float, SceneObject]] = []
  for obj in scene.objects.values():
    score = _label_score(obj, label_phrase)
    if score >= min_score:
      scored.append((score, obj))
  scored.sort(key=lambda item: (-item[0], int(item[1].object_id)))

  if not scored:
    return []

  best_score = scored[0][0]
  if best_score >= 0.85:
    return [obj for score, obj in scored if score >= best_score - 0.05]
  return [obj for _, obj in scored]


def resolve_anchor_objects(
  scene: SceneData,
  phrase: str,
  pick_best: bool = False,
) -> List[SceneObject]:
  if not phrase:
    return []

  phrase = phrase.strip()

  closest_clause = THAT_IS_CLOSEST_RE.match(phrase)
  if closest_clause:
    base, reference = closest_clause.groups()
    pool = objects_matching_label(scene, base)
    pool = _prefer_raw_label(pool, base)
    reference_objects = resolve_anchor_objects(scene, reference, pick_best=True)
    if reference_objects:
      pool = _pick_by_distance_rank(scene, pool, reference_objects, "closest")
    if pick_best:
      return pool[:1] if pool else []
    return pool

  farthest_clause = THAT_IS_FARTHEST_RE.match(phrase)
  if farthest_clause:
    base, reference = farthest_clause.groups()
    pool = objects_matching_label(scene, base)
    pool = _prefer_raw_label(pool, base)
    reference_objects = resolve_anchor_objects(scene, reference, pick_best=True)
    if reference_objects:
      pool = _pick_by_distance_rank(scene, pool, reference_objects, "farthest")
    if pick_best:
      return pool[:1] if pool else []
    return pool

  target_part, relation, anchor_part = _split_by_relation(phrase)
  if relation:
    pool = objects_matching_label(scene, target_part)
    if relation in SPATIAL_RELATIONS:
      inner_anchors = resolve_anchor_objects(scene, anchor_part, pick_best=False)
      if not inner_anchors:
        inner_anchors = resolve_anchor_objects(scene, anchor_part, pick_best=True)
    else:
      inner_anchors = resolve_anchor_objects(scene, anchor_part, pick_best=True)

    if not inner_anchors:
      return pool if not pick_best else (pool[:1] if pool else [])

    if relation in ("closest", "farthest"):
      pool = _pick_by_distance_rank(scene, pool, inner_anchors, relation)
    elif relation == "with":
      on_it_match = WITH_ON_IT_RE.match(anchor_part.strip())
      if on_it_match:
        inner_label = on_it_match.group(1)
        inner_objects = objects_matching_label(scene, inner_label)
        filtered = [
          candidate
          for candidate in pool
          if any(_relation_holds(scene, inner, candidate, "on") for inner in inner_objects)
        ]
        if filtered:
          pool = filtered
        else:
          pool = [
            obj
            for obj in scene.objects.values()
            if any(_relation_holds(scene, inner, obj, "on") for inner in inner_objects)
          ]
      else:
        pool = [
          candidate
          for candidate in pool
          if any(_relation_holds(scene, inner, candidate, "on") for inner in inner_anchors)
        ]
    else:
      pool = [
        candidate
        for candidate in pool
        if any(_relation_holds(scene, candidate, anchor, relation) for anchor in inner_anchors)
      ]

    if pick_best:
      return pool[:1] if pool else []
    return pool

  matches = objects_matching_label(scene, phrase)
  if pick_best and matches:
    return [matches[0]]
  return matches


def _filter_find(scene: SceneData, query: ParsedQuery) -> List[SceneObject]:
  candidates = objects_matching_label(scene, query.target_class)
  candidates = _apply_attribute_filters(candidates, query.attributes)
  if not candidates and query.attributes.color:
    candidates = _apply_attribute_filters(
      objects_matching_label(scene, query.target_class),
      AttributeFilters(color=None, size=query.attributes.size),
    )

  if not query.relation:
    return candidates

  if query.relation == "between" and len(query.anchors) >= 2:
    anchors_a = resolve_anchor_objects(scene, query.anchors[0].class_name)
    anchors_b = resolve_anchor_objects(scene, query.anchors[1].class_name)
    return _filter_between(scene, candidates, anchors_a, anchors_b)

  anchor_phrases = [anchor.class_name for anchor in query.anchors]
  if not anchor_phrases and query.relation in ("closest", "farthest"):
    return candidates

  if query.relation in ("closest", "farthest"):
    anchor_objects: List[SceneObject] = []
    for phrase in anchor_phrases:
      anchor_objects.extend(resolve_anchor_objects(scene, phrase, pick_best=True))
    if not anchor_objects:
      for phrase in anchor_phrases:
        anchor_objects.extend(resolve_anchor_objects(scene, phrase))
    if not anchor_objects:
      for phrase in anchor_phrases:
        anchor_objects.extend(objects_matching_label(scene, phrase)[:5])
    if not anchor_objects:
      return []
    return _pick_by_distance_rank(scene, candidates, anchor_objects, query.relation)

  anchor_objects: List[SceneObject] = []
  for phrase in anchor_phrases:
    anchor_objects.extend(resolve_anchor_objects(scene, phrase))

  if not anchor_objects:
    return []

  matched = [
    candidate
    for candidate in candidates
    if any(_relation_holds(scene, candidate, anchor, query.relation) for anchor in anchor_objects)
  ]

  if len(matched) > 1:
    for phrase in anchor_phrases:
      closest_tail = re.search(r"closest to (.+)$", phrase, flags=re.IGNORECASE)
      if closest_tail:
        reference = resolve_anchor_objects(scene, closest_tail.group(1), pick_best=True)
        if reference:
          return _pick_by_distance_rank(scene, matched, reference, "closest")

  return matched


def _filter_between(
  scene: SceneData,
  candidates: Sequence[SceneObject],
  anchors_a: Sequence[SceneObject],
  anchors_b: Sequence[SceneObject],
) -> List[SceneObject]:
  ids_a = {anchor.object_id for anchor in anchors_a}
  ids_b = {anchor.object_id for anchor in anchors_b}
  matched: List[SceneObject] = []

  for candidate in candidates:
    if _graph_between(scene, candidate.object_id, ids_a, ids_b):
      matched.append(candidate)
      continue
    if any(
      _geometric_between(candidate, anchor_a, anchor_b)
      for anchor_a in anchors_a
      for anchor_b in anchors_b
      if anchor_a.object_id != anchor_b.object_id
    ):
      matched.append(candidate)

  return matched


def _apply_attribute_filters(
  objects: Sequence[SceneObject],
  filters: AttributeFilters,
) -> List[SceneObject]:
  return [
    obj
    for obj in objects
    if _matches_color(obj, filters.color) and _matches_size(obj, filters.size)
  ]


def _matches_color(obj: SceneObject, color: Optional[str]) -> bool:
  if not color:
    return True
  needle = color.lower()
  allowed = COLOR_ALIASES.get(needle, {needle})
  for scheme in obj.color_schemes:
    if scheme.label and scheme.label.lower() in allowed:
      return True
  return any(alias in obj.raw_label.lower() for alias in allowed)


def _matches_size(obj: SceneObject, size: Optional[str]) -> bool:
  if not size:
    return True
  return size.lower() in obj.raw_label.lower()


def _prefer_raw_label(pool: Sequence[SceneObject], label: str) -> List[SceneObject]:
  target = _clean_phrase(label).lower()
  preferred = [obj for obj in pool if obj.raw_label.lower() == target]
  return preferred if preferred else list(pool)


def _label_variants(query: str) -> List[str]:
  cleaned = _clean_phrase(query).lower()
  variants = [cleaned]
  if cleaned in LABEL_SYNONYMS:
    variants.extend(LABEL_SYNONYMS[cleaned])
  return variants


def _label_score(obj: SceneObject, query: str) -> float:
  return max(_label_score_single(obj, variant) for variant in _label_variants(query))


def _label_score_single(obj: SceneObject, query: str) -> float:
  q = query.lower()
  if not q:
    return 0.0

  labels = [obj.raw_label.lower(), obj.nyu_label.lower(), obj.nyu40_label.lower()]
  for label in labels:
    if not label:
      continue
    if label == q:
      return 1.0
    if q in label or label in q:
      return 0.9

  q_words = set(q.split())
  for label in labels:
    if not label:
      continue
    label_words = set(label.split())
    if q_words <= label_words:
      return 0.85
    overlap = len(q_words & label_words) / len(q_words)
    if overlap >= 0.5:
      return 0.5 + overlap * 0.35

  return 0.0


def _relation_holds(
  scene: SceneData,
  candidate: SceneObject,
  anchor: SceneObject,
  relation: str,
) -> bool:
  if relation in ("closest", "farthest", "between"):
    return True

  graph_relation = "on" if relation == "with" else relation
  if _graph_relation_holds(scene, candidate, anchor, graph_relation):
    return True
  return _geometric_relation_holds(candidate, anchor, graph_relation)


def _graph_relation_holds(
  scene: SceneData,
  candidate: SceneObject,
  anchor: SceneObject,
  relation: str,
) -> bool:
  checks: Dict[str, Callable[[], bool]] = {
    "on": lambda: candidate.object_id in scene.related_object_ids(anchor.object_id, "on"),
    "in": lambda: candidate.object_id in scene.related_object_ids(anchor.object_id, "in"),
    "above": lambda: (
      anchor.object_id in scene.related_object_ids(candidate.object_id, "above")
      or candidate.object_id in scene.related_object_ids(anchor.object_id, "below")
    ),
    "below": lambda: (
      anchor.object_id in scene.related_object_ids(candidate.object_id, "below")
      or candidate.object_id in scene.related_object_ids(anchor.object_id, "above")
    ),
    "near": lambda: (
      anchor.object_id in scene.related_object_ids(candidate.object_id, "near")
      or candidate.object_id in scene.related_object_ids(anchor.object_id, "near")
    ),
    "beside": lambda: (
      anchor.object_id in scene.related_object_ids(candidate.object_id, "beside")
      or candidate.object_id in scene.related_object_ids(anchor.object_id, "beside")
    ),
    "hanging_on": lambda: (
      anchor.object_id in scene.related_object_ids(candidate.object_id, "hanging_on")
      or candidate.object_id in scene.related_object_ids(anchor.object_id, "hanging_on")
    ),
  }
  checker = checks.get(relation)
  return checker() if checker else False


def _graph_between(
  scene: SceneData,
  candidate_id: str,
  ids_a: Set[str],
  ids_b: Set[str],
) -> bool:
  for pair in scene.get_relations(candidate_id, "between"):
    if not isinstance(pair, tuple) or len(pair) != 2:
      continue
    left, right = pair
    if (left in ids_a and right in ids_b) or (left in ids_b and right in ids_a):
      return True
  return False


def _pick_by_distance_rank(
  scene: SceneData,
  candidates: Sequence[SceneObject],
  anchor_objects: Sequence[SceneObject],
  relation: str,
) -> List[SceneObject]:
  if not candidates:
    return []

  anchor_ids = {anchor.object_id for anchor in anchor_objects}
  scored: List[tuple[float, float, SceneObject]] = []

  for candidate in candidates:
    rank = _anchor_rank(scene, candidate.object_id, anchor_ids, "closest")
    distance = min(_distance(candidate, anchor) for anchor in anchor_objects)
    scored.append((rank, distance, candidate))

  if relation == "farthest":
    scored.sort(
      key=lambda item: (
        item[0] if item[0] != math.inf else -1.0,
        item[1],
        int(item[2].object_id),
      ),
      reverse=True,
    )
  else:
    scored.sort(key=lambda item: (
      item[0] if item[0] != math.inf else math.inf,
      item[1],
      int(item[2].object_id),
    ))

  best = scored[0][2]
  return [best]


def _anchor_rank(
  scene: SceneData,
  subject_id: str,
  anchor_ids: Set[str],
  relation: str,
) -> float:
  ranked = scene.related_object_ids(subject_id, relation)
  ranks = [index for index, object_id in enumerate(ranked) if object_id in anchor_ids]
  return float(min(ranks)) if ranks else math.inf


def _tie_break(
  scene: SceneData,
  candidates: Sequence[SceneObject],
  query: ParsedQuery,
) -> SceneObject:
  if query.relation in ("closest", "farthest") and query.anchors:
    anchor_objects: List[SceneObject] = []
    for anchor in query.anchors:
      anchor_objects.extend(resolve_anchor_objects(scene, anchor.class_name, pick_best=True))
    if anchor_objects:
      return _pick_by_distance_rank(scene, candidates, anchor_objects, query.relation)[0]

  return min(candidates, key=lambda obj: int(obj.object_id))


def _distance(left: SceneObject, right: SceneObject) -> float:
  return math.hypot(left.cx - right.cx, left.cy - right.cy)


def _geometric_relation_holds(
  candidate: SceneObject,
  anchor: SceneObject,
  relation: str,
) -> bool:
  if relation == "near":
    return _distance(candidate, anchor) < NEAR_DISTANCE_M
  if relation == "on":
    return _proximity_on(candidate, anchor)
  if relation == "above":
    if _is_above(candidate, anchor):
      return True
    return _is_wall_mounted_label(candidate) and _is_wall_above(candidate, anchor)
  if relation == "below":
    return _is_above(anchor, candidate)
  if relation == "in":
    return _xy_contains(anchor, candidate)
  if relation == "beside":
    return _horizontal_overlap(candidate, anchor) and not _is_above(candidate, anchor)
  return False


def _is_wall_mounted_label(obj: SceneObject) -> bool:
  label = obj.raw_label.lower()
  if "record" in label or "decoration" in label:
    return False
  keywords = (
    "picture", "photo", "painting", "calligraphy", "mirror", "lamp", "clock", "frame",
  )
  return any(word in label for word in keywords)


def _proximity_on(upper: SceneObject, lower: SceneObject) -> bool:
  upper_bottom = upper.cz - upper.z_length / 2.0
  lower_top = lower.cz + lower.z_length / 2.0
  if abs(upper_bottom - lower_top) > ON_PROXIMITY_TOLERANCE_M:
    return False
  xy_distance = math.hypot(upper.cx - lower.cx, upper.cy - lower.cy)
  support_size = max(lower.x_length, lower.y_length, lower.z_length, 0.5)
  return xy_distance <= support_size * 2.0


def _geometric_between(
  candidate: SceneObject,
  anchor_a: SceneObject,
  anchor_b: SceneObject,
) -> bool:
  ax, ay = anchor_a.cx, anchor_a.cy
  bx, by = anchor_b.cx, anchor_b.cy
  px, py = candidate.cx, candidate.cy

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
  perp = math.hypot(px - proj_x, py - proj_y)
  return perp <= BETWEEN_PERP_TOLERANCE_M


def _is_above(upper: SceneObject, lower: SceneObject) -> bool:
  if not _horizontal_overlap(upper, lower):
    return False
  return upper.cz - lower.cz > (upper.z_length + lower.z_length) / 2.0 - OVERLAP_EPSILON_M


def _is_wall_above(upper: SceneObject, lower: SceneObject) -> bool:
  xy_distance = math.hypot(upper.cx - lower.cx, upper.cy - lower.cy)
  support_span = max(lower.x_length, lower.y_length, 1.0)
  z_gap = upper.cz - lower.cz
  return xy_distance <= support_span * 1.5 and z_gap > 0.3


def _vertical_contact(upper: SceneObject, lower: SceneObject) -> bool:
  upper_bottom = upper.cz - upper.z_length / 2.0
  lower_top = lower.cz + lower.z_length / 2.0
  return abs(upper_bottom - lower_top) < ON_CONTACT_TOLERANCE_M


def _horizontal_overlap(left: SceneObject, right: SceneObject) -> bool:
  left_x = (left.cx - left.x_length / 2.0, left.cx + left.x_length / 2.0)
  left_y = (left.cy - left.y_length / 2.0, left.cy + left.y_length / 2.0)
  right_x = (right.cx - right.x_length / 2.0, right.cx + right.x_length / 2.0)
  right_y = (right.cy - right.y_length / 2.0, right.cy + right.y_length / 2.0)
  return (
    left_x[0] <= right_x[1] + OVERLAP_EPSILON_M
    and right_x[0] <= left_x[1] + OVERLAP_EPSILON_M
    and left_y[0] <= right_y[1] + OVERLAP_EPSILON_M
    and right_y[0] <= left_y[1] + OVERLAP_EPSILON_M
  )


def _xy_contains(container: SceneObject, inner: SceneObject) -> bool:
  container_x = (container.cx - container.x_length / 2.0, container.cx + container.x_length / 2.0)
  container_y = (container.cy - container.y_length / 2.0, container.cy + container.y_length / 2.0)
  return (
    inner.cx - inner.x_length / 2.0 >= container_x[0]
    and inner.cx + inner.x_length / 2.0 <= container_x[1]
    and inner.cy - inner.y_length / 2.0 >= container_y[0]
    and inner.cy + inner.y_length / 2.0 <= container_y[1]
  )
