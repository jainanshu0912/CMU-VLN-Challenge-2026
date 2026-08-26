"""Helpers for unique capture output paths (avoid overwriting prior runs)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def utc_run_id() -> str:
  """Filesystem-safe UTC timestamp, e.g. 20260820_012530."""
  return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def unique_capture_dir(
  root: str | Path,
  scene_name: str,
  *,
  run_id: str | None = None,
) -> Path:
  """Return ``<root>/<scene_name>/<run_id>/`` and create it."""
  stamp = run_id or utc_run_id()
  out = Path(root).expanduser() / scene_name / stamp
  out.mkdir(parents=True, exist_ok=True)
  return out


def resolve_unique_graph_path(
  *,
  scene_name: str,
  graph_output_path: str = "",
  graph_output_dir: str = "/tmp/vlm_live_captures",
  run_id: str | None = None,
) -> Path:
  """Choose a non-colliding scene-graph JSON path.

  - If ``graph_output_path`` is a concrete ``*.json`` file, insert a run id
    before the suffix: ``foo.json`` → ``foo_20260820_012530.json``.
  - If ``graph_output_path`` is empty, write under
    ``<graph_output_dir>/<scene_name>/<run_id>/scene_graph.json``.
  """
  stamp = run_id or utc_run_id()
  raw = (graph_output_path or "").strip()
  if raw:
    path = Path(raw).expanduser()
    if path.suffix.lower() == ".json":
      stamped = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    else:
      # Treat as a directory.
      stamped = path / scene_name / stamp / "scene_graph.json"
    stamped.parent.mkdir(parents=True, exist_ok=True)
    return stamped

  out_dir = unique_capture_dir(graph_output_dir, scene_name, run_id=stamp)
  return out_dir / "scene_graph.json"
