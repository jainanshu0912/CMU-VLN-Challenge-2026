"""Count objects matching parsed numerical questions against static scene data."""

from __future__ import annotations

from typing import List

from vlm_pipeline.graph_search import (
  _apply_attribute_filters,
  _pick_by_distance_rank,
  _relation_holds,
  objects_matching_label,
  resolve_anchor_objects,
)
from vlm_pipeline.query_parser import AttributeFilters, ParsedQuery
from vlm_pipeline.scene_loader import SceneData, SceneObject

COUNT_COLOR_ALIASES = {
  "black": {"black", "gray", "grey", "charcoal"},
  "white": {"white", "ivory"},
  "red": {"red", "maroon", "crimson"},
  "blue": {"blue", "navy"},
}


class CountPipeline:
  """Filter scene objects for count questions and return the match count."""

  def count(self, scene: SceneData, query: ParsedQuery) -> int:
    return len(self.filter_objects(scene, query))

  def filter_objects(self, scene: SceneData, query: ParsedQuery) -> List[SceneObject]:
    candidates = objects_matching_label(scene, query.target_class)
    candidates = _apply_count_attribute_filters(candidates, query.attribute_filters)

    relation = query.spatial_filter.relation
    anchor_phrase = query.spatial_filter.anchor
    if not relation or not anchor_phrase:
      return candidates

    if relation in ("closest", "farthest"):
      anchor_objects = resolve_anchor_objects(scene, anchor_phrase, pick_best=True)
      if not anchor_objects:
        return candidates
      return _pick_by_distance_rank(scene, candidates, anchor_objects, relation)

    anchor_objects = resolve_anchor_objects(scene, anchor_phrase)
    if not anchor_objects:
      return []

    if relation == "on" and len(anchor_objects) > 1:
      anchor_objects = _pick_count_support(scene, anchor_phrase, anchor_objects)

    if relation == "with":
      inner_objects = objects_matching_label(scene, anchor_phrase)
      if not inner_objects:
        inner_objects = anchor_objects
      return [
        candidate
        for candidate in candidates
        if any(_relation_holds(scene, inner, candidate, "on") for inner in inner_objects)
      ]

    graph_relation = relation
    return [
      candidate
      for candidate in candidates
      if any(
        _relation_holds(scene, candidate, anchor, graph_relation)
        for anchor in anchor_objects
      )
    ]


def _apply_count_attribute_filters(
  objects: List[SceneObject],
  filters: AttributeFilters,
) -> List[SceneObject]:
  if not filters.color:
    return _apply_attribute_filters(objects, filters)

  matched = _apply_count_color_filter(objects, filters.color)
  if matched:
    return _apply_attribute_filters(matched, AttributeFilters(color=None, size=filters.size))

  return _apply_attribute_filters(objects, filters)


def _apply_count_color_filter(
  objects: List[SceneObject],
  color: str,
) -> List[SceneObject]:
  needle = color.lower()
  allowed = COUNT_COLOR_ALIASES.get(needle, {needle})
  return [
    obj
    for obj in objects
    if any(
      scheme.label and scheme.label.lower() in allowed
      for scheme in obj.color_schemes
    )
    or any(alias in obj.raw_label.lower() for alias in allowed)
  ]


def _pick_count_support(
  scene: SceneData,
  anchor_phrase: str,
  candidates: List[SceneObject],
) -> List[SceneObject]:
  phrase = anchor_phrase.lower()
  if "picture" in phrase or "photo" in phrase:
    pictures = objects_matching_label(scene, "picture")
    if not pictures:
      pictures = objects_matching_label(scene, "photo")
    scored = [
      (
        sum(1 for pic in pictures if _relation_holds(scene, pic, obj, "above")),
        obj,
      )
      for obj in candidates
    ]
    best_count = max(score for score, _ in scored)
    if best_count > 0:
      return [obj for score, obj in scored if score == best_count]
  return candidates
