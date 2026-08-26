"""Unit tests for scene-graph comparison helpers."""

from __future__ import annotations

import unittest

from vlm_pipeline_live.compare_scene_graphs import (
  compare_graphs,
  match_objects,
  iter_objects,
  _normalize_label,
)


def _obj(oid: str, label: str, x: float, y: float, z: float = 0.5):
  return {
    "object_id": oid,
    "raw_label": label,
    "bbox_center": [x, y, z],
    "bbox_size": [0.2, 0.2, 0.4],
  }


def _graph(objects):
  return {
    "scene_name": "test",
    "regions": {
      "0": {
        "region_id": "0",
        "objects": objects,
        "relationships": {
          "near": {"0": ["1"]},
        },
      }
    },
  }


class CompareSceneGraphsTest(unittest.TestCase):
  def test_normalize_aliases(self) -> None:
    self.assertEqual(_normalize_label("potted plant"), "plant")
    self.assertEqual(_normalize_label("TV Monitor"), "monitor")
    self.assertEqual(_normalize_label("focus light"), "lamp")

  def test_match_same_label_near(self) -> None:
    pred = iter_objects(_graph([_obj("0", "chair", 0.0, 0.0), _obj("1", "plant", 2.0, 0.0)]))
    gt = iter_objects(_graph([_obj("a", "chair", 0.1, 0.0), _obj("b", "potted plant", 2.05, 0.0)]))
    matches, extra, missing = match_objects(pred, gt, max_dist_m=1.0)
    self.assertEqual(len(matches), 2)
    self.assertEqual(extra, [])
    self.assertEqual(missing, [])

  def test_compare_report_fields(self) -> None:
    pred = _graph([_obj("0", "chair", 0.0, 0.0)])
    gt = _graph([_obj("9", "chair", 0.2, 0.0), _obj("8", "table", 5.0, 5.0)])
    report = compare_graphs(pred, gt, pred_path="p", gt_path="g", max_match_dist_m=1.0)
    self.assertEqual(report.matched, 1)
    self.assertEqual(report.unmatched_gt, 1)
    self.assertEqual(report.pred_objects, 1)
    self.assertEqual(report.gt_objects, 2)
    self.assertIsNotNone(report.label_recall)


if __name__ == "__main__":
  unittest.main()
