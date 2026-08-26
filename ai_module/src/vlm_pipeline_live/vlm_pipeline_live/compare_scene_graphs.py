"""Compare a Pipeline B (live) scene graph against a VLA-3D ground-truth graph.

Reports object-count / label-frequency stats, spatial matching (label + XY),
and relation-edge overlap for shared relation types.

Example::

  ros2 run vlm_pipeline_live compare_scene_graphs -- \\
    --pred ai_module/src/vlm_pipeline_live/data/captured/chinese_room/chinese_room_scene_graph.json \\
    --gt ~/vla3d_data/Unity/chinese_room/chinese_room_scene_graph.json \\
    --out /tmp/chinese_room_compare.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlm_pipeline_live.label_utils import canonicalize_label


# Soft label aliases so GroundingDINO phrases can match VLA-3D names.
LABEL_ALIASES: dict[str, str] = {}  # kept for backward compat; use canonicalize_label


@dataclass
class GraphObject:
  object_id: str
  label: str
  canon_label: str
  x: float
  y: float
  z: float
  size: tuple[float, float, float]


@dataclass
class CompareReport:
  pred_path: str
  gt_path: str
  pred_objects: int
  gt_objects: int
  pred_relations: int
  gt_relations: int
  matched: int
  unmatched_pred: int
  unmatched_gt: int
  mean_match_dist_m: float | None
  median_match_dist_m: float | None
  label_precision: float | None
  label_recall: float | None
  relation_precision: float | None
  relation_recall: float | None
  pred_label_counts: dict[str, int]
  gt_label_counts: dict[str, int]
  matched_pairs: list[dict[str, Any]]
  missing_gt_labels: list[dict[str, Any]]
  extra_pred_labels: list[dict[str, Any]]


def _normalize_label(label: str) -> str:
  return canonicalize_label(label)


def _center_and_size(obj: dict[str, Any]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
  if obj.get("bbox_center") is not None and obj.get("bbox_size") is not None:
    c = obj["bbox_center"]
    s = obj["bbox_size"]
    return (float(c[0]), float(c[1]), float(c[2])), (
      float(s[0]),
      float(s[1]),
      float(s[2]),
    )
  if obj.get("center") is not None and obj.get("size") is not None:
    c = obj["center"]
    s = obj["size"]
    return (float(c[0]), float(c[1]), float(c[2])), (
      float(s[0]),
      float(s[1]),
      float(s[2]),
    )
  raise ValueError(f"Object missing center/size: id={obj.get('object_id')}")


def load_graph(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
  if "regions" not in data:
    raise ValueError(f"Missing regions in {path}")
  return data


def iter_objects(graph: dict[str, Any]) -> list[GraphObject]:
  out: list[GraphObject] = []
  for region in graph.get("regions", {}).values():
    for obj in region.get("objects", []):
      label = str(obj.get("raw_label") or obj.get("nyu_label") or "")
      center, size = _center_and_size(obj)
      out.append(
        GraphObject(
          object_id=str(obj.get("object_id", "")),
          label=label,
          canon_label=_normalize_label(label),
          x=center[0],
          y=center[1],
          z=center[2],
          size=size,
        )
      )
  return out


def iter_relation_edges(graph: dict[str, Any]) -> set[tuple[str, str, str]]:
  """Return (relation_type, src_id, dst_id) edges (between uses dst as 'a|b')."""
  edges: set[tuple[str, str, str]] = set()
  for region in graph.get("regions", {}).values():
    relationships = region.get("relationships", {}) or {}
    for rel_type, mapping in relationships.items():
      if not isinstance(mapping, dict):
        continue
      for src, dsts in mapping.items():
        src_id = str(src)
        if not isinstance(dsts, list):
          continue
        for dst in dsts:
          if isinstance(dst, list):
            # between: [id_a, id_b]
            if len(dst) >= 2:
              a, b = sorted((str(dst[0]), str(dst[1])))
              edges.add((str(rel_type), src_id, f"{a}|{b}"))
          else:
            edges.add((str(rel_type), src_id, str(dst)))
  return edges


def _xy_dist(a: GraphObject, b: GraphObject) -> float:
  return math.hypot(a.x - b.x, a.y - b.y)


def match_objects(
  pred: list[GraphObject],
  gt: list[GraphObject],
  *,
  max_dist_m: float = 1.5,
  require_label: bool = True,
) -> tuple[list[tuple[GraphObject, GraphObject, float]], list[GraphObject], list[GraphObject]]:
  """Greedy match: same canonical label (optional), nearest XY within max_dist_m."""
  remaining_gt = list(gt)
  matches: list[tuple[GraphObject, GraphObject, float]] = []
  unmatched_pred: list[GraphObject] = []

  # Prefer rarer / more distinctive labels first by matching nearest overall.
  for p in sorted(pred, key=lambda o: (o.canon_label, o.object_id)):
    best_idx = -1
    best_dist = float("inf")
    for idx, g in enumerate(remaining_gt):
      if require_label and p.canon_label != g.canon_label:
        continue
      dist = _xy_dist(p, g)
      if dist < best_dist:
        best_dist = dist
        best_idx = idx
    if best_idx >= 0 and best_dist <= max_dist_m:
      g = remaining_gt.pop(best_idx)
      matches.append((p, g, best_dist))
    else:
      unmatched_pred.append(p)

  return matches, unmatched_pred, remaining_gt


def remap_relation_edges(
  edges: set[tuple[str, str, str]],
  id_map: dict[str, str],
) -> set[tuple[str, str, str]]:
  """Map pred object ids → gt ids; drop edges whose endpoints are unmatched."""
  remapped: set[tuple[str, str, str]] = set()
  for rel_type, src, dst in edges:
    if src not in id_map:
      continue
    if "|" in dst:
      a, b = dst.split("|", 1)
      if a not in id_map or b not in id_map:
        continue
      aa, bb = sorted((id_map[a], id_map[b]))
      remapped.add((rel_type, id_map[src], f"{aa}|{bb}"))
    else:
      if dst not in id_map:
        continue
      remapped.add((rel_type, id_map[src], id_map[dst]))
  return remapped


def compare_graphs(
  pred_graph: dict[str, Any],
  gt_graph: dict[str, Any],
  *,
  pred_path: str = "",
  gt_path: str = "",
  max_match_dist_m: float = 1.5,
) -> CompareReport:
  pred_objs = iter_objects(pred_graph)
  gt_objs = iter_objects(gt_graph)
  matches, unmatched_pred, unmatched_gt = match_objects(
    pred_objs, gt_objs, max_dist_m=max_match_dist_m, require_label=True
  )

  dists = [d for _, _, d in matches]
  id_map = {p.object_id: g.object_id for p, g, _ in matches}

  pred_edges = iter_relation_edges(pred_graph)
  gt_edges = iter_relation_edges(gt_graph)
  pred_edges_mapped = remap_relation_edges(pred_edges, id_map)
  # Only score relation types present in both graphs among matched nodes.
  shared_types = {e[0] for e in pred_edges_mapped} & {e[0] for e in gt_edges}
  pred_shared = {e for e in pred_edges_mapped if e[0] in shared_types}
  gt_shared = {e for e in gt_edges if e[0] in shared_types}
  inter = pred_shared & gt_shared

  pred_labels = Counter(o.canon_label for o in pred_objs)
  gt_labels = Counter(o.canon_label for o in gt_objs)
  # Multiset precision/recall over canonical labels
  tp = sum(min(pred_labels[l], gt_labels[l]) for l in set(pred_labels) | set(gt_labels))
  label_precision = (tp / sum(pred_labels.values())) if pred_objs else None
  label_recall = (tp / sum(gt_labels.values())) if gt_objs else None

  rel_precision = (len(inter) / len(pred_shared)) if pred_shared else None
  rel_recall = (len(inter) / len(gt_shared)) if gt_shared else None

  def _median(vals: list[float]) -> float | None:
    if not vals:
      return None
    s = sorted(vals)
    mid = len(s) // 2
    if len(s) % 2:
      return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])

  return CompareReport(
    pred_path=pred_path,
    gt_path=gt_path,
    pred_objects=len(pred_objs),
    gt_objects=len(gt_objs),
    pred_relations=len(pred_edges),
    gt_relations=len(gt_edges),
    matched=len(matches),
    unmatched_pred=len(unmatched_pred),
    unmatched_gt=len(unmatched_gt),
    mean_match_dist_m=(sum(dists) / len(dists)) if dists else None,
    median_match_dist_m=_median(dists),
    label_precision=label_precision,
    label_recall=label_recall,
    relation_precision=rel_precision,
    relation_recall=rel_recall,
    pred_label_counts=dict(pred_labels.most_common()),
    gt_label_counts=dict(gt_labels.most_common()),
    matched_pairs=[
      {
        "pred_id": p.object_id,
        "gt_id": g.object_id,
        "pred_label": p.label,
        "gt_label": g.label,
        "canon_label": p.canon_label,
        "dist_xy_m": round(d, 3),
      }
      for p, g, d in matches[:50]  # cap for readability
    ],
    missing_gt_labels=[
      {"id": o.object_id, "label": o.label, "canon": o.canon_label}
      for o in unmatched_gt[:40]
    ],
    extra_pred_labels=[
      {"id": o.object_id, "label": o.label, "canon": o.canon_label}
      for o in unmatched_pred[:40]
    ],
  )


def format_report(report: CompareReport) -> str:
  lines = [
    f"pred: {report.pred_path}",
    f"gt:   {report.gt_path}",
    f"objects: pred={report.pred_objects}  gt={report.gt_objects}  "
    f"matched={report.matched}  extra_pred={report.unmatched_pred}  missing_gt={report.unmatched_gt}",
    f"relations: pred={report.pred_relations}  gt={report.gt_relations}",
  ]
  if report.mean_match_dist_m is not None:
    lines.append(
      f"match XY dist: mean={report.mean_match_dist_m:.3f}m  "
      f"median={report.median_match_dist_m:.3f}m"
    )
  if report.label_precision is not None:
    lines.append(
      f"label multiset: precision={report.label_precision:.3f}  "
      f"recall={report.label_recall:.3f}"
    )
  if report.relation_precision is not None:
    lines.append(
      f"relation edges (matched ids): precision={report.relation_precision:.3f}  "
      f"recall={report.relation_recall:.3f}"
    )
  lines.append("top pred labels: " + ", ".join(
    f"{k}:{v}" for k, v in list(report.pred_label_counts.items())[:10]
  ))
  lines.append("top gt labels:   " + ", ".join(
    f"{k}:{v}" for k, v in list(report.gt_label_counts.items())[:10]
  ))
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Compare live/pred scene graph vs VLA-3D ground-truth graph",
  )
  parser.add_argument("--pred", type=Path, required=True, help="Predicted / live scene graph JSON")
  parser.add_argument("--gt", type=Path, required=True, help="Ground-truth VLA-3D scene graph JSON")
  parser.add_argument(
    "--max-match-dist-m",
    type=float,
    default=1.5,
    help="Max XY distance (m) to accept a label-matched pair",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Optional JSON report output path",
  )
  args = parser.parse_args(argv)

  if not args.pred.is_file():
    raise SystemExit(f"Pred graph not found: {args.pred}")
  if not args.gt.is_file():
    raise SystemExit(f"GT graph not found: {args.gt}")

  report = compare_graphs(
    load_graph(args.pred),
    load_graph(args.gt),
    pred_path=str(args.pred),
    gt_path=str(args.gt),
    max_match_dist_m=args.max_match_dist_m,
  )
  print(format_report(report))

  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
      json.dump(asdict(report), handle, indent=2)
    print(f"Wrote report → {args.out}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
