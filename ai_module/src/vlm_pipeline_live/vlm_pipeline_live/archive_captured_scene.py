"""Archive a live scene graph into ``data/captured/<scene_name>/<run_id>/``.

Each archive lands in a new timestamped folder so prior runs are kept.

Example::

  ros2 run vlm_pipeline_live archive_captured_scene -- \\
    --graph /tmp/vlm_live_captures/office_2/latest_scene_graph.json \\
    --scene-name office_2 \\
    --out-root $PWD/src/vlm_pipeline_live/data/captured
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from vlm_pipeline_live.capture_paths import unique_capture_dir, utc_run_id
from vlm_pipeline_live.write_object_list_from_scene_graph import (
  export_scene_folder,
  load_scene_graph,
  objects_from_scene_graph,
)


def archive_scene(
  graph_path: Path,
  out_root: Path,
  scene_name: str,
  *,
  notes: str = "",
  world_name: str = "",
  run_id: str | None = None,
) -> Path:
  graph = load_scene_graph(graph_path)
  rows = objects_from_scene_graph(graph)
  stamp = run_id or utc_run_id()
  out_dir = unique_capture_dir(out_root, scene_name, run_id=stamp)
  written = export_scene_folder(
    graph_path,
    out_dir,
    scene_name,
    write_object_list=True,
  )

  n_rel = 0
  for region in graph.get("regions", {}).values():
    relationships = region.get("relationships", {}) or {}
    for mapping in relationships.values():
      if isinstance(mapping, dict):
        for dsts in mapping.values():
          if isinstance(dsts, list):
            n_rel += len(dsts)

  meta = {
    "scene_name": scene_name,
    "world_name": world_name or scene_name,
    "run_id": stamp,
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_graph": str(graph_path),
    "num_objects": len(rows),
    "num_relation_entries": n_rel,
    "notes": notes,
    "files": {key: str(path) for key, path in written.items()},
    "mode": "manual",
  }
  meta_path = out_dir / "capture_meta.json"
  with meta_path.open("w", encoding="utf-8") as handle:
    json.dump(meta, handle, indent=2)

  # Scene-level pointer to newest archive (does not delete older runs).
  latest_meta = out_root / scene_name / "latest.json"
  latest_meta.parent.mkdir(parents=True, exist_ok=True)
  latest_meta.write_text(
    json.dumps({"run_id": stamp, "path": str(out_dir)}, indent=2),
    encoding="utf-8",
  )

  return out_dir


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Archive a live scene graph under data/captured/<scene>/<run_id>/",
  )
  parser.add_argument(
    "--graph",
    type=Path,
    default=Path("/tmp/vlm_live_captures/office_2/latest_scene_graph.json"),
    help="Input live scene graph JSON (default: latest office_2 capture pointer)",
  )
  parser.add_argument(
    "--scene-name",
    required=True,
    help="Scene / world name (e.g. office_2, chinese_room)",
  )
  parser.add_argument(
    "--out-root",
    type=Path,
    default=Path("data/captured"),
    help="Root folder for captured scenes (default: data/captured)",
  )
  parser.add_argument(
    "--run-id",
    default="",
    help="Optional run id (default: UTC timestamp)",
  )
  parser.add_argument(
    "--world-name",
    default="",
    help="Optional sim world_name if different from scene-name",
  )
  parser.add_argument("--notes", default="", help="Optional free-text notes")
  args = parser.parse_args(argv)

  if not args.graph.is_file():
    raise SystemExit(f"Scene graph not found: {args.graph}")

  out_dir = archive_scene(
    args.graph,
    args.out_root,
    args.scene_name,
    notes=args.notes,
    world_name=args.world_name,
    run_id=args.run_id or None,
  )
  print(f"Archived capture → {out_dir}")
  for path in sorted(out_dir.iterdir()):
    print(f"  {path.name} ({path.stat().st_size} bytes)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
