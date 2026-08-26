"""Compare several Pipeline B backend runs against one VLA-3D GT graph.

Example (inside AI container after three captures)::

  ros2 run vlm_pipeline_live compare_backend_runs -- \\
    --gt /home/docker/vla3d_data/Unity/office_2/office_2_scene_graph.json \\
    --run grounding_dino:/tmp/vlm_live_captures/office_2_dino/latest_scene_graph.json \\
    --run dino_gemini:/tmp/vlm_live_captures/office_2_dino_gemini/latest_scene_graph.json \\
    --run yoloe:/tmp/vlm_live_captures/office_2_yoloe/latest_scene_graph.json \\
    --out /tmp/office_2_backend_compare.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vlm_pipeline_live.compare_scene_graphs import (
  compare_graphs,
  format_report,
  load_graph,
)


def _parse_run(spec: str) -> tuple[str, Path]:
  if ":" not in spec:
    raise argparse.ArgumentTypeError(
      f"Expected name:path, got {spec!r}. Example: yoloe:/tmp/.../latest_scene_graph.json"
    )
  name, path_str = spec.split(":", 1)
  name = name.strip()
  path = Path(path_str.strip()).expanduser()
  if not name:
    raise argparse.ArgumentTypeError(f"Empty run name in {spec!r}")
  return name, path


def _summary_row(name: str, report) -> dict[str, Any]:
  return {
    "name": name,
    "pred_path": report.pred_path,
    "pred_objects": report.pred_objects,
    "gt_objects": report.gt_objects,
    "matched": report.matched,
    "extra_pred": report.unmatched_pred,
    "missing_gt": report.unmatched_gt,
    "mean_match_dist_m": report.mean_match_dist_m,
    "median_match_dist_m": report.median_match_dist_m,
    "label_precision": report.label_precision,
    "label_recall": report.label_recall,
    "relation_precision": report.relation_precision,
    "relation_recall": report.relation_recall,
  }


def _fmt(value: float | None, digits: int = 3) -> str:
  if value is None:
    return "  n/a"
  return f"{value:.{digits}f}"


def format_table(rows: list[dict[str, Any]]) -> str:
  headers = [
    ("backend", 14),
    ("objs", 6),
    ("matched", 8),
    ("extra", 6),
    ("miss", 6),
    ("labP", 6),
    ("labR", 6),
    ("meanXY", 7),
    ("relP", 6),
    ("relR", 6),
  ]
  lines = ["  ".join(h.rjust(w) for h, w in headers)]
  lines.append("-" * len(lines[0]))
  for row in rows:
    cells = [
      (row["name"][:14], 14),
      (str(row["pred_objects"]), 6),
      (str(row["matched"]), 8),
      (str(row["extra_pred"]), 6),
      (str(row["missing_gt"]), 6),
      (_fmt(row["label_precision"]), 6),
      (_fmt(row["label_recall"]), 6),
      (_fmt(row["mean_match_dist_m"]), 7),
      (_fmt(row["relation_precision"]), 6),
      (_fmt(row["relation_recall"]), 6),
    ]
    lines.append("  ".join(text.rjust(width) for text, width in cells))
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Compare multiple live backend scene graphs vs one GT graph",
  )
  parser.add_argument("--gt", type=Path, required=True, help="VLA-3D ground-truth scene graph")
  parser.add_argument(
    "--run",
    action="append",
    required=True,
    type=_parse_run,
    help="Repeatable name:path (e.g. yoloe:/tmp/.../latest_scene_graph.json)",
  )
  parser.add_argument("--max-match-dist-m", type=float, default=1.5)
  parser.add_argument("--out", type=Path, default=None, help="Optional JSON summary path")
  args = parser.parse_args(argv)

  gt_path = args.gt.expanduser()
  if not gt_path.is_file():
    raise SystemExit(f"GT graph not found: {gt_path}")

  gt_graph = load_graph(gt_path)
  rows: list[dict[str, Any]] = []
  details: dict[str, Any] = {"gt": str(gt_path), "runs": {}}

  print(f"GT: {gt_path}\n")
  for name, pred_path in args.run:
    if not pred_path.is_file():
      raise SystemExit(f"Pred graph not found for {name}: {pred_path}")
    report = compare_graphs(
      load_graph(pred_path),
      gt_graph,
      pred_path=str(pred_path),
      gt_path=str(gt_path),
      max_match_dist_m=args.max_match_dist_m,
    )
    print(f"=== {name} ===")
    print(format_report(report))
    print()
    row = _summary_row(name, report)
    rows.append(row)
    details["runs"][name] = {
      "summary": row,
      "report": asdict(report),
    }

  print("=== summary ===")
  print(format_table(rows))

  # Rank by matched, then label F1-ish (P*R), then fewer extras.
  def _rank_key(row: dict[str, Any]) -> tuple:
    p = row["label_precision"] or 0.0
    r = row["label_recall"] or 0.0
    return (row["matched"], p * r, -row["extra_pred"])

  ranked = sorted(rows, key=_rank_key, reverse=True)
  print("\nBest by (matched, label P×R, fewer extras): " + " > ".join(r["name"] for r in ranked))

  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    details["ranking"] = [r["name"] for r in ranked]
    details["table"] = rows
    with args.out.open("w", encoding="utf-8") as handle:
      json.dump(details, handle, indent=2)
    print(f"\nWrote report → {args.out}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
