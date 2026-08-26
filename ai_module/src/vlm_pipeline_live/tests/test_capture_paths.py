"""Tests for unique capture path helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vlm_pipeline_live.capture_paths import resolve_unique_graph_path, unique_capture_dir


class CapturePathsTests(unittest.TestCase):
  def test_unique_dirs_differ(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      a = unique_capture_dir(tmp, "office_2", run_id="runA")
      b = unique_capture_dir(tmp, "office_2", run_id="runB")
      self.assertNotEqual(a, b)
      self.assertTrue(a.is_dir())
      self.assertTrue(b.is_dir())

  def test_stamped_json_path(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = resolve_unique_graph_path(
        scene_name="office_2",
        graph_output_path=str(Path(tmp) / "graph.json"),
        run_id="20260101_000000",
      )
      self.assertEqual(path.name, "graph_20260101_000000.json")
      self.assertTrue(path.parent.is_dir())

  def test_default_dir_layout(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = resolve_unique_graph_path(
        scene_name="office_2",
        graph_output_path="",
        graph_output_dir=tmp,
        run_id="20260101_000001",
      )
      self.assertEqual(path, Path(tmp) / "office_2" / "20260101_000001" / "scene_graph.json")


if __name__ == "__main__":
  unittest.main()
