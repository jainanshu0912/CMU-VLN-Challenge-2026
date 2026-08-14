"""Unit tests for live scene graph builder (no ROS / GroundingDINO)."""

from __future__ import annotations

import unittest

from vlm_pipeline.scene_loader import SceneLoader
from vlm_pipeline_live.lidar_camera_fusion import DetectedObject3D
from vlm_pipeline_live.live_scene_graph import (
  LiveSceneGraphBuilder,
  detections_to_scene_objects,
  scene_data_to_vla3d_json,
)


class LiveSceneGraphTests(unittest.TestCase):
  def test_detections_to_objects(self):
    dets = [
      DetectedObject3D("Chair", 0.9, 1.0, 2.0, 0.5, 0.4, 0.4, 0.9, 0.1, 12, 0.0),
    ]
    objects = detections_to_scene_objects(dets)
    self.assertEqual(len(objects), 1)
    self.assertEqual(objects[0].object_id, "0")
    self.assertEqual(objects[0].raw_label, "chair")
    self.assertEqual(objects[0].nyu_label, "chair")

  def test_book_on_table(self):
    dets = [
      DetectedObject3D("table", 0.95, 0.0, 0.0, 0.4, 1.2, 0.8, 0.5, 0.0, 30, 0.0),
      DetectedObject3D("book", 0.9, 0.05, 0.0, 0.68, 0.25, 0.2, 0.04, 0.0, 10, 0.0),
    ]
    result = LiveSceneGraphBuilder().build_from_detections(dets)
    on_table = result.scene.regions["0"].relationships["on"]["0"]
    self.assertIn("1", on_table)

  def test_near_and_closest(self):
    dets = [
      DetectedObject3D("sofa", 0.9, 0.0, 0.0, 0.4, 2.0, 0.8, 0.8, 0.0, 20, 0.0),
      DetectedObject3D("pillow", 0.8, 0.5, 0.1, 0.5, 0.4, 0.3, 0.2, 0.0, 8, 0.0),
      DetectedObject3D("pillow", 0.8, 3.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.0, 8, 0.0),
    ]
    result = LiveSceneGraphBuilder(near_distance_m=1.5).build_from_detections(dets)
    near_sofa = result.scene.regions["0"].relationships["near"]["0"]
    self.assertIn("1", near_sofa)
    self.assertNotIn("2", near_sofa)

    closest_to_sofa = result.scene.regions["0"].relationships["closest"]["0"]
    self.assertIn("1", closest_to_sofa)

  def test_between(self):
    dets = [
      DetectedObject3D("plant", 0.9, 1.0, 0.0, 0.5, 0.3, 0.3, 0.8, 0.0, 10, 0.0),
      DetectedObject3D("chair", 0.9, 0.0, 0.0, 0.45, 0.5, 0.5, 0.9, 0.0, 10, 0.0),
      DetectedObject3D("chair", 0.9, 2.0, 0.0, 0.45, 0.5, 0.5, 0.9, 0.0, 10, 0.0),
    ]
    result = LiveSceneGraphBuilder().build_from_detections(dets)
    between = result.scene.regions["0"].relationships["between"]["0"]
    self.assertTrue(any(set(pair) == {"1", "2"} for pair in between))

  def test_json_roundtrip_via_scene_loader_parser(self):
    dets = [
      DetectedObject3D("table", 0.9, 0.0, 0.0, 0.4, 1.0, 1.0, 0.5, 0.0, 10, 0.0),
      DetectedObject3D("lamp", 0.8, 1.2, 0.0, 0.7, 0.2, 0.2, 0.5, 0.0, 8, 0.0),
    ]
    result = LiveSceneGraphBuilder(scene_name="unit_test").build_from_detections(dets)
    graph = scene_data_to_vla3d_json(result.scene)
    self.assertEqual(graph["scene_name"], "unit_test")
    self.assertIn("0", graph["regions"])
    self.assertIn("relationships", graph["regions"]["0"])
    self.assertEqual(len(graph["regions"]["0"]["objects"]), 2)

    # Ensure SceneLoader can parse the relationships block shape.
    loader = SceneLoader(".")
    parsed = loader._parse_relationships(graph["regions"]["0"]["relationships"])
    self.assertIn("near", parsed)
    self.assertIn("closest", parsed)

  def test_empty_detections(self):
    result = LiveSceneGraphBuilder().build_from_detections([])
    self.assertEqual(result.num_objects, 0)
    self.assertEqual(result.num_relations, 0)


if __name__ == "__main__":
  unittest.main()
