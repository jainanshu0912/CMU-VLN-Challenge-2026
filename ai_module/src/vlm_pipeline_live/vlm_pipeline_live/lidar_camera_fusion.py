"""Fuse GroundingDINO 2D boxes with /registered_scan for 3D object boxes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

from vlm_pipeline_live.equirect_to_perspective import EquirectPerspectiveProjector
from vlm_pipeline_live.detection_backend import Detection2D
from vlm_pipeline_live.label_utils import canonicalize_label

try:
  from sensor_msgs_py import point_cloud2 as pc2
except ImportError:  # pragma: no cover
  pc2 = None


CEILING_LABELS = {
  "lamp",
  "lantern",
  "ceiling",
  "focus light",
  "wall lamp",
  "ceiling lamp",
}

# Typical max height so wall/ceiling points cannot inflate a chair into a 3 m pillar.
MAX_Z_LENGTH_M = {
  "pillow": 0.35,
  "book": 0.25,
  "shoes": 0.25,
  "tray": 0.2,
  "stool": 0.7,
  "chair": 1.1,
  "table": 1.0,
  "sofa": 1.2,
  "plant": 1.6,
  "potted plant": 1.6,
  "carpet": 0.15,
  "vase": 0.8,
  "arabic jar": 0.9,
}
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
  low = np.percentile(points_map, 8, axis=0)
  high = np.percentile(points_map, 92, axis=0)
  size = np.maximum(high - low, 0.08)
  return center, size


def refine_matched_points(
  matched: np.ndarray,
  robot_xy: np.ndarray,
  robot_z: float,
  label: str,
  min_points: int,
) -> np.ndarray:
  """Drop wall/ceiling outliers so the 3D box sits on the object, not in mid-air."""
  pts = matched
  if len(pts) == 0:
    return pts

  dist = np.linalg.norm(pts[:, :2] - robot_xy.reshape(1, 2), axis=1)
  if len(pts) >= max(min_points, 12):
    med_d = float(np.median(dist))
    keep = np.abs(dist - med_d) <= 0.8
    if int(np.count_nonzero(keep)) >= min_points:
      pts = pts[keep]
      dist = dist[keep]

  canon = canonicalize_label(label)
  if canon in CEILING_LABELS:
    zc = float(np.percentile(pts[:, 2], 70))
    keep_z = np.abs(pts[:, 2] - zc) <= 0.45
    if int(np.count_nonzero(keep_z)) >= min_points:
      pts = pts[keep_z]
    return pts

  zmin = robot_z - 1.05
  zmax = robot_z + 1.35
  keep_z = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
  if int(np.count_nonzero(keep_z)) >= min_points:
    pts = pts[keep_z]
  zc = float(np.percentile(pts[:, 2], 30))
  keep_band = np.abs(pts[:, 2] - zc) <= 0.55
  if int(np.count_nonzero(keep_band)) >= min_points:
    pts = pts[keep_band]
  return pts


def sit_box_on_floor(
  center: np.ndarray,
  size: np.ndarray,
  robot_z: float,
  label: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Shift/clamp a furniture box so it is not a floating pillar."""
  center = np.asarray(center, dtype=np.float64).copy()
  size = np.asarray(size, dtype=np.float64).copy()
  canon = canonicalize_label(label)
  if canon in CEILING_LABELS:
    size[2] = min(float(size[2]), 0.6)
    return center, size

  max_h = MAX_Z_LENGTH_M.get(canon, 1.7)
  size[2] = min(float(size[2]), max_h)
  size[0] = min(float(size[0]), 3.5)
  size[1] = min(float(size[1]), 3.5)

  floor_z = robot_z - 0.72
  bottom = center[2] - size[2] / 2.0
  # Only pull down boxes that are clearly airborne (not test clouds at z≈0).
  if bottom > max(floor_z + 0.18, 0.28):
    center[2] -= bottom - (floor_z + 0.04)
  bottom = center[2] - size[2] / 2.0
  if bottom < floor_z - 0.15 and robot_z > 0.3:
    lift = (floor_z - 0.02) - bottom
    center[2] += lift
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

    robot_z = float(odometry.pose.pose.position.z)
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
      matched = refine_matched_points(
        matched,
        robot_xy,
        robot_z,
        det.label,
        self.min_lidar_points,
      )
      if len(matched) < self.min_lidar_points:
        continue

      center, size = bbox_from_points(matched)
      center, size = sit_box_on_floor(center, size, robot_z, det.label)
      objects.append(
        DetectedObject3D(
          label=canonicalize_label(det.label),
          confidence=float(det.confidence),
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


def _xy_footprint(obj: DetectedObject3D) -> float:
  return math.hypot(max(obj.x_length, 0.05), max(obj.y_length, 0.05))


def _class_aware_merge_radius_m(
  a: DetectedObject3D,
  b: DetectedObject3D,
  base_distance_m: float,
) -> float:
  """Larger furniture gets a larger duplicate-merge radius."""
  scale = 0.55 * max(_xy_footprint(a), _xy_footprint(b))
  return max(base_distance_m, min(2.5, scale))


def _nms_rank(obj: DetectedObject3D) -> tuple[float, int]:
  # Prefer higher confidence, then denser LiDAR support.
  return (float(obj.confidence), int(obj.num_lidar_points))


def nms_3d(
  objects: list[DetectedObject3D],
  distance_m: float,
  *,
  class_aware: bool = True,
) -> list[DetectedObject3D]:
  """Suppress duplicate 3D boxes.

  When ``class_aware`` is True (default), only boxes with the same canonical
  label can suppress each other. Merge radius grows with object footprint so
  large desks/cabinets merge across views more aggressively than cups.
  """
  if not objects:
    return []

  # Canonicalize again so accumulated maps from older runs still merge.
  normalized: list[DetectedObject3D] = []
  for obj in objects:
    canon = canonicalize_label(obj.label)
    if canon != obj.label:
      obj = DetectedObject3D(
        label=canon,
        confidence=obj.confidence,
        cx=obj.cx,
        cy=obj.cy,
        cz=obj.cz,
        x_length=obj.x_length,
        y_length=obj.y_length,
        z_length=obj.z_length,
        heading=obj.heading,
        num_lidar_points=obj.num_lidar_points,
        source_heading_deg=obj.source_heading_deg,
      )
    normalized.append(obj)

  kept: list[DetectedObject3D] = []
  for candidate in sorted(normalized, key=_nms_rank, reverse=True):
    duplicate = False
    for existing in kept:
      if class_aware and candidate.label != existing.label:
        continue
      dist = math.hypot(
        candidate.cx - existing.cx,
        candidate.cy - existing.cy,
        candidate.cz - existing.cz,
      )
      radius = _class_aware_merge_radius_m(candidate, existing, distance_m)
      if dist < radius:
        duplicate = True
        break
    if not duplicate:
      kept.append(candidate)
  return kept
