"""Unit tests for Pipeline A export path helpers (no ROS)."""

from __future__ import annotations

import unittest
from pathlib import Path

from vlm_sequential.export_paths import (
  exported_scene_paths,
  is_scene_exported,
  scene_export_dir,
)


class ExportPathTests(unittest.TestCase):
  def test_paths_use_scene_folder_and_filenames(self) -> None:
    csv_path, graph_path = exported_scene_paths("/tmp/vla3d_live", "live_scene")
    self.assertEqual(
      csv_path,
      Path("/tmp/vla3d_live/live_scene/live_scene_object_result.csv"),
    )
    self.assertEqual(
      graph_path,
      Path("/tmp/vla3d_live/live_scene/live_scene_scene_graph.json"),
    )
    self.assertEqual(
      scene_export_dir("/tmp/vla3d_live", "live_scene"),
      Path("/tmp/vla3d_live/live_scene"),
    )

  def test_is_scene_exported_requires_both_files(self) -> None:
    root = self._tmp()
    scene = "live_scene"
    self.assertFalse(is_scene_exported(str(root), scene))
    scene_dir = root / scene
    scene_dir.mkdir(parents=True)
    (scene_dir / f"{scene}_object_result.csv").write_text("object_id\n", encoding="utf-8")
    self.assertFalse(is_scene_exported(str(root), scene))
    (scene_dir / f"{scene}_scene_graph.json").write_text("{}", encoding="utf-8")
    self.assertTrue(is_scene_exported(str(root), scene))

  def _tmp(self) -> Path:
    import tempfile

    return Path(tempfile.mkdtemp())


if __name__ == "__main__":
  unittest.main()
