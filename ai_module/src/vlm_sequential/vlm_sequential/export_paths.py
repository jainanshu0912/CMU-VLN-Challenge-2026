"""Paths where Pipeline B writes a Pipeline A–compatible scene folder."""

from __future__ import annotations

from pathlib import Path


DEFAULT_EXPORT_ROOT = "/tmp/vla3d_live"
DEFAULT_SCENE_NAME = "live_scene"


def scene_export_dir(export_root: str, scene_name: str) -> Path:
  return Path(export_root).expanduser() / scene_name


def exported_scene_paths(export_root: str, scene_name: str) -> tuple[Path, Path]:
  scene_dir = scene_export_dir(export_root, scene_name)
  return (
    scene_dir / f"{scene_name}_object_result.csv",
    scene_dir / f"{scene_name}_scene_graph.json",
  )


def is_scene_exported(export_root: str, scene_name: str) -> bool:
  csv_path, graph_path = exported_scene_paths(export_root, scene_name)
  return csv_path.is_file() and graph_path.is_file()
