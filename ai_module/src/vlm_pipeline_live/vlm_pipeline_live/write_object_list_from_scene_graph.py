"""Export Pipeline A object CSV (+ optional VLA-3D scene folder) from a live scene graph.

Pipeline B writes ``/tmp/vlm_live_scene_graph.json`` (objects nested under regions).
Pipeline A expects::

  <vla3d_data_root>/<scene_name>/
    <scene_name>_object_result.csv
    <scene_name>_scene_graph.json

Example::

  ros2 run vlm_pipeline_live write_object_list_from_scene_graph -- \\
    --graph /tmp/vlm_live_scene_graph.json \\
    --out-dir /tmp/vla3d_live/live_scene \\
    --scene-name live_scene

  ros2 launch vlm_pipeline vlm_pipeline.launch.py \\
    scene_name:=live_scene \\
    vla3d_data_root:=/tmp/vla3d_live
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
  "object_id",
  "region_id",
  "raw_label",
  "nyu_id",
  "nyu40_id",
  "nyu_label",
  "nyu40_label",
  "object_bbox_cx",
  "object_bbox_cy",
  "object_bbox_cz",
  "object_bbox_xlength",
  "object_bbox_ylength",
  "object_bbox_zlength",
  "object_bbox_heading",
  "object_front_heading",
  "object_color_r1",
  "object_color_g1",
  "object_color_b1",
  "object_color_scheme1",
  "object_color_scheme_percentage1",
  "object_color_scheme_average_dist1",
  "object_color_r2",
  "object_color_g2",
  "object_color_b2",
  "object_color_scheme2",
  "object_color_scheme_percentage2",
  "object_color_scheme_average_dist2",
  "object_color_r3",
  "object_color_g3",
  "object_color_b3",
  "object_color_scheme3",
  "object_color_scheme_percentage3",
  "object_color_scheme_average_dist3",
]


def load_scene_graph(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
  if "regions" not in data:
    raise ValueError(f"Scene graph missing 'regions': {path}")
  return data


def objects_from_scene_graph(graph: dict[str, Any]) -> list[dict[str, Any]]:
  """Flatten region objects into Pipeline A ``object_result.csv`` rows."""
  rows: list[dict[str, Any]] = []
  for region_id, region in graph.get("regions", {}).items():
    for obj in region.get("objects", []):
      label = obj.get("raw_label") or obj.get("nyu_label") or ""
      center = obj.get("bbox_center") or [0.0, 0.0, 0.0]
      size = obj.get("bbox_size") or [0.0, 0.0, 0.0]
      if len(center) < 3 or len(size) < 3:
        raise ValueError(f"Invalid bbox for object {obj.get('object_id')}: {obj}")

      color_vals = obj.get("color_vals") or [[-1, -1, -1]] * 3
      color_labels = obj.get("color_labels") or ["N/A", "N/A", "N/A"]
      color_pcts = obj.get("color_percentages") or ["0", "0", "0"]

      def _color_channel(idx: int, channel: int) -> str:
        try:
          val = color_vals[idx][channel]
        except (IndexError, TypeError):
          return ""
        if val is None or val == -1:
          return ""
        return str(val)

      def _color_label(idx: int) -> str:
        try:
          return str(color_labels[idx])
        except (IndexError, TypeError):
          return "N/A"

      def _color_pct(idx: int) -> str:
        try:
          return str(color_pcts[idx])
        except (IndexError, TypeError):
          return "0"

      rows.append({
        "object_id": str(obj.get("object_id", "")),
        "region_id": str(region_id),
        "raw_label": label,
        "nyu_id": str(obj.get("nyu_id", "-1")),
        "nyu40_id": str(obj.get("nyu40_id", "-1")),
        "nyu_label": str(obj.get("nyu_label", label)),
        "nyu40_label": str(obj.get("nyu40_label", label)),
        "object_bbox_cx": float(center[0]),
        "object_bbox_cy": float(center[1]),
        "object_bbox_cz": float(center[2]),
        "object_bbox_xlength": float(size[0]),
        "object_bbox_ylength": float(size[1]),
        "object_bbox_zlength": float(size[2]),
        "object_bbox_heading": float(obj.get("bbox_heading", 0.0)),
        "object_front_heading": "",
        "object_color_r1": _color_channel(0, 0),
        "object_color_g1": _color_channel(0, 1),
        "object_color_b1": _color_channel(0, 2),
        "object_color_scheme1": _color_label(0),
        "object_color_scheme_percentage1": _color_pct(0),
        "object_color_scheme_average_dist1": "",
        "object_color_r2": _color_channel(1, 0),
        "object_color_g2": _color_channel(1, 1),
        "object_color_b2": _color_channel(1, 2),
        "object_color_scheme2": _color_label(1),
        "object_color_scheme_percentage2": _color_pct(1),
        "object_color_scheme_average_dist2": "",
        "object_color_r3": _color_channel(2, 0),
        "object_color_g3": _color_channel(2, 1),
        "object_color_b3": _color_channel(2, 2),
        "object_color_scheme3": _color_label(2),
        "object_color_scheme_percentage3": _color_pct(2),
        "object_color_scheme_average_dist3": "",
      })
  return rows


def write_object_result_csv(rows: list[dict[str, Any]], path: Path) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  return path


def write_object_list_txt(rows: list[dict[str, Any]], path: Path) -> Path:
  """Write simulator-style ``object_list.txt`` (id x y z)."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(
        f"{row['object_id']} "
        f"{row['object_bbox_cx']} "
        f"{row['object_bbox_cy']} "
        f"{row['object_bbox_cz']}\n"
      )
  return path


def write_scene_graph_json(graph: dict[str, Any], path: Path, scene_name: str) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  out = dict(graph)
  out["scene_name"] = scene_name
  with path.open("w", encoding="utf-8") as handle:
    json.dump(out, handle, indent=2)
  return path


def export_scene_folder(
  graph_path: Path,
  out_dir: Path,
  scene_name: str,
  *,
  write_object_list: bool = True,
) -> dict[str, Path]:
  """Write CSV + scene_graph.json (+ optional object_list.txt) under ``out_dir``."""
  graph = load_scene_graph(graph_path)
  rows = objects_from_scene_graph(graph)
  if not rows:
    raise ValueError(f"No objects found in scene graph: {graph_path}")

  out_dir.mkdir(parents=True, exist_ok=True)
  written = {
    "csv": write_object_result_csv(rows, out_dir / f"{scene_name}_object_result.csv"),
    "scene_graph": write_scene_graph_json(
      graph, out_dir / f"{scene_name}_scene_graph.json", scene_name
    ),
  }
  if write_object_list:
    written["object_list"] = write_object_list_txt(rows, out_dir / "object_list.txt")
  return written


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Write Pipeline A object_result.csv from a live scene_graph.json",
  )
  parser.add_argument(
    "--graph",
    type=Path,
    default=Path("/tmp/vlm_live_scene_graph.json"),
    help="Input live scene graph JSON (default: /tmp/vlm_live_scene_graph.json)",
  )
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("/tmp/vla3d_live/live_scene"),
    help=(
      "Output scene folder for Pipeline A "
      "(default: /tmp/vla3d_live/live_scene; "
      "~/vla3d_data is usually mounted read-only in Docker)"
    ),
  )
  parser.add_argument(
    "--scene-name",
    default="live_scene",
    help="Scene name used in output filenames (default: live_scene)",
  )
  parser.add_argument(
    "--csv-only",
    action="store_true",
    help="Only write <scene>_object_result.csv (skip graph copy / object_list.txt)",
  )
  parser.add_argument(
    "--no-object-list",
    action="store_true",
    help="Do not write object_list.txt",
  )
  args = parser.parse_args(argv)

  if not args.graph.is_file():
    raise SystemExit(f"Scene graph not found: {args.graph}")

  try:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probe = args.out_dir / ".write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)
  except OSError as exc:
    raise SystemExit(
      f"Cannot write to {args.out_dir}: {exc}\n"
      "  Host mount ~/vla3d_data is read-only in Docker.\n"
      "  Use a writable path, e.g.:\n"
      "    --out-dir /tmp/vla3d_live/live_scene\n"
      "  Then launch Pipeline A with:\n"
      "    vla3d_data_root:=/tmp/vla3d_live scene_name:=live_scene"
    ) from exc

  graph = load_scene_graph(args.graph)
  rows = objects_from_scene_graph(graph)
  if not rows:
    raise SystemExit(f"No objects found in scene graph: {args.graph}")

  csv_path = write_object_result_csv(
    rows, args.out_dir / f"{args.scene_name}_object_result.csv"
  )
  print(f"Wrote {len(rows)} objects → {csv_path}")

  if not args.csv_only:
    graph_out = write_scene_graph_json(
      graph, args.out_dir / f"{args.scene_name}_scene_graph.json", args.scene_name
    )
    print(f"Wrote scene graph → {graph_out}")
    if not args.no_object_list:
      list_path = write_object_list_txt(rows, args.out_dir / "object_list.txt")
      print(f"Wrote object list → {list_path}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
