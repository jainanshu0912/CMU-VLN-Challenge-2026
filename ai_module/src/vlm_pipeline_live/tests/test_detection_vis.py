"""Tests for 2D detection overlay helpers."""

from __future__ import annotations

import unittest

import numpy as np

from vlm_pipeline_live.detection_vis import draw_detections_on_image, draw_projected_points
from vlm_pipeline_live.detection_backend import Detection2D


class DetectionVisTests(unittest.TestCase):
  def test_draw_boxes_changes_pixels(self) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    det = Detection2D(
      label="chair",
      confidence=0.91,
      x1=10,
      y1=12,
      x2=40,
      y2=50,
      heading_deg=0.0,
      crop_width=64,
      crop_height=64,
    )
    out = draw_detections_on_image(image, [det])
    self.assertEqual(out.shape, image.shape)
    self.assertGreater(int(out.sum()), 0)

  def test_draw_projected_points(self) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    px = np.array([20.0, 50.0])
    py = np.array([20.0, 50.0])
    vis = np.array([True, True])
    out = draw_projected_points(image, px, py, vis, labels=["a", "b"])
    self.assertEqual(out.shape, image.shape)
    self.assertGreater(int(out.sum()), 0)


if __name__ == "__main__":
  unittest.main()
