"""Unit tests for LiDAR-camera fusion without GroundingDINO."""

from __future__ import annotations

import unittest

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header

from vlm_pipeline_live.detection_backend import Detection2D
from vlm_pipeline_live.lidar_camera_fusion import (
  LidarCameraFusion,
  map_points_to_camera,
  nms_3d,
  odometry_to_map_sensor_transform,
  refine_matched_points,
  sit_box_on_floor,
)
from vlm_pipeline_live.lidar_camera_fusion import DetectedObject3D


def _make_cloud(points: np.ndarray, frame_id: str = "map") -> PointCloud2:
  msg = PointCloud2()
  msg.header = Header(frame_id=frame_id)
  msg.height = 1
  msg.width = len(points)
  msg.is_bigendian = False
  msg.is_dense = True
  msg.point_step = 12
  msg.row_step = msg.point_step * msg.width
  msg.fields = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
  ]
  msg.data = points.astype(np.float32).tobytes()
  return msg


def _identity_odom() -> Odometry:
  msg = Odometry()
  msg.pose.pose.orientation.w = 1.0
  return msg


class LidarFusionTests(unittest.TestCase):
  def test_map_sensor_roundtrip(self):
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    odom = _identity_odom()
    rot, trans = odometry_to_map_sensor_transform(odom)
    restored = (rot.T @ (points - trans).T).T
    self.assertTrue(np.allclose(restored, points))

  def test_fuse_assigns_points_in_box(self):
    fusion = LidarCameraFusion(min_lidar_points=3, nms_distance_m=0.2)
    odom = _identity_odom()

    cluster = np.array([
      [2.0, 0.0, 0.0],
      [2.1, 0.05, 0.0],
      [2.0, -0.05, 0.1],
      [2.05, 0.02, -0.05],
    ], dtype=np.float64)
    cloud = _make_cloud(cluster)

    points_cam = map_points_to_camera(cluster, odom)
    px, py, visible = fusion.projector.project_camera_points_to_crop(points_cam, 0.0)
    self.assertTrue(bool(np.all(visible)))

    x1 = float(np.min(px) - 5.0)
    x2 = float(np.max(px) + 5.0)
    y1 = float(np.min(py) - 5.0)
    y2 = float(np.max(py) + 5.0)

    det = Detection2D(
      label="chair",
      confidence=0.9,
      x1=x1,
      y1=y1,
      x2=x2,
      y2=y2,
      heading_deg=0.0,
      crop_width=640,
      crop_height=640,
    )
    objects = fusion.fuse([det], cloud, odom)
    self.assertEqual(len(objects), 1)
    self.assertEqual(objects[0].label, "chair")
    self.assertAlmostEqual(objects[0].cx, 2.05, delta=0.15)

  def test_nms_3d_merges_duplicates(self):
    first = DetectedObject3D(
      label="book",
      confidence=0.9,
      cx=1.0,
      cy=1.0,
      cz=1.0,
      x_length=0.2,
      y_length=0.2,
      z_length=0.2,
      heading=0.0,
      num_lidar_points=10,
      source_heading_deg=0.0,
    )
    second = DetectedObject3D(
      label="book",
      confidence=0.5,
      cx=1.1,
      cy=1.0,
      cz=1.0,
      x_length=0.2,
      y_length=0.2,
      z_length=0.2,
      heading=0.0,
      num_lidar_points=8,
      source_heading_deg=90.0,
    )
    kept = nms_3d([second, first], distance_m=0.5)
    self.assertEqual(len(kept), 1)
    self.assertAlmostEqual(kept[0].confidence, 0.9)

  def test_nms_3d_merges_aliased_labels(self):
    trash_can = DetectedObject3D(
      label="trash can",
      confidence=0.8,
      cx=1.0,
      cy=1.0,
      cz=0.4,
      x_length=0.3,
      y_length=0.3,
      z_length=0.5,
      heading=0.0,
      num_lidar_points=20,
      source_heading_deg=0.0,
    )
    trash_bin = DetectedObject3D(
      label="trash bin",
      confidence=0.6,
      cx=1.15,
      cy=1.0,
      cz=0.4,
      x_length=0.3,
      y_length=0.3,
      z_length=0.5,
      heading=0.0,
      num_lidar_points=12,
      source_heading_deg=90.0,
    )
    kept = nms_3d([trash_bin, trash_can], distance_m=0.5)
    self.assertEqual(len(kept), 1)
    self.assertEqual(kept[0].label, "trash can")

  def test_nms_3d_keeps_different_classes(self):
    chair = DetectedObject3D(
      label="chair",
      confidence=0.9,
      cx=1.0,
      cy=1.0,
      cz=0.5,
      x_length=0.5,
      y_length=0.5,
      z_length=0.9,
      heading=0.0,
      num_lidar_points=30,
      source_heading_deg=0.0,
    )
    desk = DetectedObject3D(
      label="desk",
      confidence=0.9,
      cx=1.1,
      cy=1.0,
      cz=0.5,
      x_length=1.2,
      y_length=0.7,
      z_length=0.8,
      heading=0.0,
      num_lidar_points=40,
      source_heading_deg=0.0,
    )
    kept = nms_3d([chair, desk], distance_m=0.5)
    self.assertEqual(len(kept), 2)

  def test_refine_drops_ceiling_points_for_furniture(self) -> None:
    robot_xy = np.array([0.0, 0.0])
    furniture = np.array([
      [2.0, 0.0, 0.15],
      [2.05, 0.04, 0.20],
      [1.95, -0.03, 0.18],
      [2.02, 0.02, 0.40],
      [2.0, 0.0, 0.55],
    ], dtype=np.float64)
    ceiling = np.array([
      [2.0, 0.0, 2.6],
      [2.1, 0.1, 2.7],
      [1.9, -0.1, 2.5],
      [2.0, 0.05, 2.55],
    ], dtype=np.float64)
    mixed = np.vstack([furniture, ceiling])
    refined = refine_matched_points(mixed, robot_xy, robot_z=0.75, label="sofa", min_points=3)
    self.assertGreaterEqual(len(refined), 3)
    self.assertLess(float(np.median(refined[:, 2])), 1.0)

  def test_sit_box_on_floor_lowers_floating_pillar(self) -> None:
    center = np.array([2.0, 0.0, 1.8])
    size = np.array([0.8, 0.6, 2.4])
    c2, s2 = sit_box_on_floor(center, size, robot_z=0.75, label="table")
    bottom = c2[2] - s2[2] / 2.0
    self.assertLess(s2[2], 1.05)
    self.assertLess(bottom, 0.25)


if __name__ == "__main__":
  unittest.main()
