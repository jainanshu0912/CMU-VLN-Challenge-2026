"""Fuse GroundingDINO 2D boxes with /registered_scan for 3D object boxes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from vlm_pipeline_live.equirect_to_perspective import EquirectPerspectiveProjector
from vlm_pipeline_live.grounding_dino_backend import Detection2D

try:
  from sensor_msgs_py import point_cloud2 as pc2
except ImportError:  # pragma: no cover
  pc2 = None


# ROS sensor frame (x forward, y left, z up) → camera optical (x right, y down, z forward).
DEFAULT_SENSOR_TO_CAMERA = np.array([
  [0.0, -1.0, 0.0],
  [0.0, 0.0, -1.0],
  [1.0, 0.0, 0.0],
], dtype=np.float64)


@dataclass
class DetectedObject3D:
  label: str
  confidence: float
  cx: float
  cy: float
  cz: float
  x_length: float
  y_length: float
  z_length: float
  heading: float
  num_lidar_points: int
  source_heading_deg: float

  def to_dict(self) -> dict:
    return asdict(self)


def odometry_to_map_sensor_transform(msg: Odometry) -> tuple[np.ndarray, np.ndarray]:
  """Return R_map_sensor and t_map_sensor where p_map = R @ p_sensor + t."""
  q = msg.pose.pose.orientation
  x, y, z, w = q.x, q.y, q.z, q.w

  rot = np.array([
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ], dtype=np.float64)

  trans = np.array([
    msg.pose.pose.position.x,
    msg.pose.pose.position.y,
    msg.pose.pose.position.z,
  ], dtype=np.float64)
  return rot, trans


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
  if pc2 is None:
    raise RuntimeError("sensor_msgs_py is required to parse PointCloud2 messages")

  # Prefer the numpy helper when available (avoids structured-dtype cast issues).
  if hasattr(pc2, "read_points_numpy"):
    points = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    if points.size == 0:
      return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)

  structured = np.array(
    list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
  )
  if structured.size == 0:
    return np.empty((0, 3), dtype=np.float64)
  # PointCloud2 often returns a padded structured dtype (itemsize 16 for xyz float32).
  if structured.dtype.names:
    return np.column_stack(
      [structured["x"], structured["y"], structured["z"]]
    ).astype(np.float64, copy=False)
  return np.asarray(structured, dtype=np.float64).reshape(-1, 3)


def map_points_to_camera(
  points_map: np.ndarray,
  odometry: Odometry,
  sensor_to_camera: np.ndarray = DEFAULT_SENSOR_TO_CAMERA,
) -> np.ndarray:
  """Transform map-frame LiDAR points into the camera optical frame."""
  rot_map_sensor, trans_map_sensor = odometry_to_map_sensor_transform(odometry)
  rot_sensor_map = rot_map_sensor.T

  points_sensor = (rot_sensor_map @ (points_map - trans_map_sensor).T).T
  return (sensor_to_camera @ points_sensor.T).T


def bbox_from_points(points_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  center = np.median(points_map, axis=0)
  low = np.percentile(points_map, 5, axis=0)
  high = np.percentile(points_map, 95, axis=0)
  size = np.maximum(high - low, 0.08)
  return center, size


def heading_from_points_xy(points_map: np.ndarray) -> float:
  if len(points_map) < 3:
    return 0.0
  xy = points_map[:, :2] - np.mean(points_map[:, :2], axis=0)
  if xy.shape[0] < 2:
    return 0.0
  cov = np.cov(xy.T)
  if not np.all(np.isfinite(cov)):
    return 0.0
  eigvals, eigvecs = np.linalg.eigh(cov)
  major = eigvecs[:, int(np.argmax(eigvals))]
  return math.atan2(float(major[1]), float(major[0]))


class LidarCameraFusion:
  """Assign registered-scan points to 2D detections and estimate 3D boxes."""

  def __init__(
    self,
    projector: EquirectPerspectiveProjector | None = None,
    min_lidar_points: int = 8,
    nms_distance_m: float = 0.5,
    max_robot_distance_m: float = 12.0,
    sensor_to_camera: np.ndarray | None = None,
  ) -> None:
    self.projector = projector or EquirectPerspectiveProjector()
    self.min_lidar_points = int(min_lidar_points)
    self.nms_distance_m = float(nms_distance_m)
    self.max_robot_distance_m = float(max_robot_distance_m)
    self.sensor_to_camera = (
      sensor_to_camera.copy()
      if sensor_to_camera is not None
      else DEFAULT_SENSOR_TO_CAMERA.copy()
    )

  def fuse(
    self,
    detections: list[Detection2D],
    registered_scan: PointCloud2,
    odometry: Odometry,
  ) -> list[DetectedObject3D]:
    points_map = pointcloud2_to_xyz(registered_scan)
    if len(points_map) == 0:
      return []

    robot_xy = np.array([
      odometry.pose.pose.position.x,
      odometry.pose.pose.position.y,
    ], dtype=np.float64)
    xy_dist = np.linalg.norm(points_map[:, :2] - robot_xy, axis=1)
    points_map = points_map[xy_dist <= self.max_robot_distance_m]
    if len(points_map) == 0:
      return []

    points_cam = map_points_to_camera(points_map, odometry, self.sensor_to_camera)
    objects: list[DetectedObject3D] = []

    for det in detections:
      px, py, visible = self.projector.project_camera_points_to_crop(
        points_cam,
        det.heading_deg,
      )
      in_box = (
        visible
        & (px >= det.x1)
        & (px <= det.x2)
        & (py >= det.y1)
        & (py <= det.y2)
      )
      matched = points_map[in_box]
      if len(matched) < self.min_lidar_points:
        continue

      center, size = bbox_from_points(matched)
      objects.append(
        DetectedObject3D(
          label=det.label,
          confidence=det.confidence,
          cx=float(center[0]),
          cy=float(center[1]),
          cz=float(center[2]),
          x_length=float(size[0]),
          y_length=float(size[1]),
          z_length=float(size[2]),
          heading=heading_from_points_xy(matched),
          num_lidar_points=int(len(matched)),
          source_heading_deg=det.heading_deg,
        )
      )

    return nms_3d(objects, self.nms_distance_m)


def nms_3d(objects: list[DetectedObject3D], distance_m: float) -> list[DetectedObject3D]:
  if not objects:
    return []

  kept: list[DetectedObject3D] = []
  for candidate in sorted(objects, key=lambda obj: obj.confidence, reverse=True):
    duplicate = False
    for existing in kept:
      if candidate.label != existing.label:
        continue
      dist = math.hypot(
        candidate.cx - existing.cx,
        candidate.cy - existing.cy,
        candidate.cz - existing.cz,
      )
      if dist < distance_m:
        duplicate = True
        break
    if not duplicate:
      kept.append(candidate)
  return kept
