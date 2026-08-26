"""Tests for multi-backend compare helpers."""

from __future__ import annotations

from vlm_pipeline_live.compare_backend_runs import _parse_run, format_table


def test_parse_run():
  name, path = _parse_run("yoloe:/tmp/foo/latest_scene_graph.json")
  assert name == "yoloe"
  assert str(path) == "/tmp/foo/latest_scene_graph.json"


def test_format_table_smoke():
  text = format_table([{
    "name": "yoloe",
    "pred_objects": 40,
    "matched": 20,
    "extra_pred": 20,
    "missing_gt": 30,
    "label_precision": 0.5,
    "label_recall": 0.4,
    "mean_match_dist_m": 0.8,
    "relation_precision": 0.1,
    "relation_recall": 0.05,
  }])
  assert "yoloe" in text
  assert "matched" in text
