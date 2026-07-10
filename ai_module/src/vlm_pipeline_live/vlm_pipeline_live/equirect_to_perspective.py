"""Equirectangular 360 camera → perspective crops for Pipeline B detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
  import cv2
except ImportError:  # pragma: no cover - optional acceleration
  cv2 = None

# Challenge system defaults (README / VLN plan).
DEFAULT_EQUIRECT_WIDTH = 1920
DEFAULT_EQUIRECT_HEIGHT = 640
DEFAULT_EQUIRECT_HFOV_DEG = 360.0
DEFAULT_EQUIRECT_VFOV_DEG = 120.0
DEFAULT_CROP_SIZE = 640
DEFAULT_CROP_HFOV_DEG = 90.0
DEFAULT_HEADINGS_DEG = (0.0, 90.0, 180.0, 270.0)


def normalize_vector(x: float, y: float, z: float) -> tuple[float, float, float]:
  norm = math.sqrt(x * x + y * y + z * z)
  if norm < 1e-9:
    return 0.0, 0.0, 1.0
  return x / norm, y / norm, z / norm


def equirect_pixel_to_ray(
  u: float,
  v: float,
  width: int = DEFAULT_EQUIRECT_WIDTH,
  height: int = DEFAULT_EQUIRECT_HEIGHT,
  hfov_deg: float = DEFAULT_EQUIRECT_HFOV_DEG,
  vfov_deg: float = DEFAULT_EQUIRECT_VFOV_DEG,
) -> tuple[float, float, float]:
  """Map an equirectangular pixel to a unit ray in the camera optical frame."""
  hfov_rad = math.radians(hfov_deg)
  vfov_rad = math.radians(vfov_deg)

  azimuth = (u / width - 0.5) * hfov_rad
  elevation = (0.5 - v / height) * vfov_rad

  cos_elev = math.cos(elevation)
  x = cos_elev * math.sin(azimuth)
  y = -math.sin(elevation)
  z = cos_elev * math.cos(azimuth)
  return normalize_vector(x, y, z)


def ray_to_equirect_pixel(
  ray: np.ndarray | tuple[float, float, float],
  width: int = DEFAULT_EQUIRECT_WIDTH,
  height: int = DEFAULT_EQUIRECT_HEIGHT,
  hfov_deg: float = DEFAULT_EQUIRECT_HFOV_DEG,
  vfov_deg: float = DEFAULT_EQUIRECT_VFOV_DEG,
) -> tuple[float, float]:
  """Map a unit ray in the camera optical frame to equirectangular pixel coords."""
  x, y, z = normalize_vector(float(ray[0]), float(ray[1]), float(ray[2]))
  hfov_rad = math.radians(hfov_deg)
  vfov_rad = math.radians(vfov_deg)

  azimuth = math.atan2(x, z)
  elevation = math.atan2(-y, math.hypot(x, z))

  u = (azimuth / hfov_rad + 0.5) * width
  v = (0.5 - elevation / vfov_rad) * height
  return u, v


@dataclass(frozen=True)
class PerspectiveCrop:
  """One perspective view extracted from a 360 equirectangular frame."""

  heading_deg: float
  image: np.ndarray
  ray_map: np.ndarray

  @property
  def height(self) -> int:
    return int(self.image.shape[0])

  @property
  def width(self) -> int:
    return int(self.image.shape[1])

  def ray_at(self, px: int, py: int) -> np.ndarray:
    """Return the unit ray (x, y, z) in the camera frame for a crop pixel."""
    return self.ray_map[py, px]


class EquirectPerspectiveProjector:
  """Project `/camera/image` equirect frames into fixed-heading perspective crops."""

  def __init__(
    self,
    equirect_width: int = DEFAULT_EQUIRECT_WIDTH,
    equirect_height: int = DEFAULT_EQUIRECT_HEIGHT,
    equirect_hfov_deg: float = DEFAULT_EQUIRECT_HFOV_DEG,
    equirect_vfov_deg: float = DEFAULT_EQUIRECT_VFOV_DEG,
    crop_size: int = DEFAULT_CROP_SIZE,
    crop_hfov_deg: float = DEFAULT_CROP_HFOV_DEG,
    headings_deg: tuple[float, ...] | list[float] = DEFAULT_HEADINGS_DEG,
  ) -> None:
    self.equirect_width = int(equirect_width)
    self.equirect_height = int(equirect_height)
    self.equirect_hfov_deg = float(equirect_hfov_deg)
    self.equirect_vfov_deg = float(equirect_vfov_deg)
    self.crop_size = int(crop_size)
    self.crop_hfov_deg = float(crop_hfov_deg)
    self.headings_deg = tuple(float(h) for h in headings_deg)

    half_crop = self.crop_size * 0.5
    focal = half_crop / math.tan(math.radians(self.crop_hfov_deg) * 0.5)
    self._focal = focal
    self._cx = half_crop
    self._cy = half_crop

    self._ray_maps: dict[float, np.ndarray] = {}
    self._sample_u: dict[float, np.ndarray] = {}
    self._sample_v: dict[float, np.ndarray] = {}
    for heading in self.headings_deg:
      ray_map, map_u, map_v = self._build_maps_for_heading(heading)
      self._ray_maps[heading] = ray_map
      self._sample_u[heading] = map_u
      self._sample_v[heading] = map_v

  def _build_maps_for_heading(
    self,
    heading_deg: float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    px = np.arange(self.crop_size, dtype=np.float32)
    py = np.arange(self.crop_size, dtype=np.float32)
    px_grid, py_grid = np.meshgrid(px, py)

    x_crop = (px_grid - self._cx) / self._focal
    y_crop = (py_grid - self._cy) / self._focal
    z_crop = np.ones_like(x_crop)

    norm = np.sqrt(x_crop * x_crop + y_crop * y_crop + z_crop * z_crop)
    x_crop /= norm
    y_crop /= norm
    z_crop /= norm

    heading_rad = math.radians(heading_deg)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    x_cam = cos_h * x_crop + sin_h * z_crop
    y_cam = y_crop
    z_cam = -sin_h * x_crop + cos_h * z_crop

    ray_map = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)

    azimuth = np.arctan2(x_cam, z_cam)
    elevation = np.arctan2(-y_cam, np.hypot(x_cam, z_cam))

    hfov_rad = math.radians(self.equirect_hfov_deg)
    vfov_rad = math.radians(self.equirect_vfov_deg)

    map_u = (azimuth / hfov_rad + 0.5) * self.equirect_width
    map_v = (0.5 - elevation / vfov_rad) * self.equirect_height
    map_u = np.mod(map_u, self.equirect_width)

    return ray_map, map_u.astype(np.float32), map_v.astype(np.float32)

  def ray_map_for_heading(self, heading_deg: float) -> np.ndarray:
    """Precomputed (H, W, 3) unit-ray map for a supported heading."""
    key = self._resolve_heading(heading_deg)
    return self._ray_maps[key]

  def _resolve_heading(self, heading_deg: float) -> float:
    rounded = round(float(heading_deg), 6)
    for heading in self.headings_deg:
      if abs(heading - rounded) < 1e-3 or abs(heading - rounded - 360.0) < 1e-3:
        return heading
    raise ValueError(
      f"Unsupported heading {heading_deg}. Expected one of {self.headings_deg}."
    )

  def crop_at_heading(self, equirect: np.ndarray, heading_deg: float) -> PerspectiveCrop:
    """Extract one perspective crop and its pixel→ray map."""
    equirect = _ensure_equirect_shape(
      equirect,
      self.equirect_width,
      self.equirect_height,
    )
    heading = self._resolve_heading(heading_deg)
    crop = _sample_equirect(
      equirect,
      self._sample_u[heading],
      self._sample_v[heading],
    )
    return PerspectiveCrop(
      heading_deg=heading,
      image=crop,
      ray_map=self._ray_maps[heading],
    )

  def crop_all(self, equirect: np.ndarray) -> list[PerspectiveCrop]:
    """Extract all configured heading crops from one equirectangular frame."""
    equirect = _ensure_equirect_shape(
      equirect,
      self.equirect_width,
      self.equirect_height,
    )
    return [self.crop_at_heading(equirect, heading) for heading in self.headings_deg]

  def project_rays_to_crop_pixels(
    self,
    rays: np.ndarray,
    heading_deg: float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame rays into crop pixel coordinates for one heading.

    Returns (px, py, valid_mask). Used later for LiDAR→image projection.
    """
    heading = self._resolve_heading(heading_deg)
    heading_rad = math.radians(heading)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    rays = np.asarray(rays, dtype=np.float64)
    x_cam = rays[..., 0]
    y_cam = rays[..., 1]
    z_cam = rays[..., 2]

    x_crop = cos_h * x_cam - sin_h * z_cam
    y_crop = y_cam
    z_crop = sin_h * x_cam + cos_h * z_cam

    valid = z_crop > 1e-6
    px = np.full(x_cam.shape, np.nan, dtype=np.float64)
    py = np.full(x_cam.shape, np.nan, dtype=np.float64)

    px[valid] = x_crop[valid] / z_crop[valid] * self._focal + self._cx
    py[valid] = y_crop[valid] / z_crop[valid] * self._focal + self._cy

    in_bounds = (
      valid
      & (px >= 0.0)
      & (px < self.crop_size)
      & (py >= 0.0)
      & (py < self.crop_size)
    )
    return px, py, in_bounds

  def project_camera_points_to_crop(
    self,
    points_cam: np.ndarray,
    heading_deg: float,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project N×3 camera-frame points to crop pixel coordinates."""
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
      raise ValueError(f"Expected N×3 points, got shape {points_cam.shape}")

    heading = self._resolve_heading(heading_deg)
    heading_rad = math.radians(heading)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]

    x_crop = cos_h * x_cam - sin_h * z_cam
    y_crop = y_cam
    z_crop = sin_h * x_cam + cos_h * z_cam

    valid = z_crop > 1e-6
    px = np.full(len(points_cam), np.nan, dtype=np.float64)
    py = np.full(len(points_cam), np.nan, dtype=np.float64)

    px[valid] = x_crop[valid] / z_crop[valid] * self._focal + self._cx
    py[valid] = y_crop[valid] / z_crop[valid] * self._focal + self._cy

    in_bounds = (
      valid
      & (px >= 0.0)
      & (px < self.crop_size)
      & (py >= 0.0)
      & (py < self.crop_size)
    )
    return px, py, in_bounds


def ros_image_to_numpy(msg) -> np.ndarray:
  """Convert a sensor_msgs/Image to an H×W×3 RGB uint8 array."""
  encoding = msg.encoding.lower()
  if encoding not in ("rgb8", "bgr8"):
    raise ValueError(f"Unsupported image encoding '{msg.encoding}' (expected rgb8 or bgr8)")

  channels = 3
  expected = msg.height * msg.width * channels
  if len(msg.data) != expected:
    raise ValueError(
      f"Unexpected image buffer length {len(msg.data)} (expected {expected})"
    )

  image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
  if encoding == "bgr8":
    image = image[..., ::-1].copy()
  return image


def _ensure_equirect_shape(
  image: np.ndarray,
  width: int,
  height: int,
) -> np.ndarray:
  if image.ndim != 3 or image.shape[2] not in (3, 4):
    raise ValueError(f"Expected H×W×3 image, got shape {image.shape}")

  if image.shape[0] != height or image.shape[1] != width:
    raise ValueError(
      f"Expected equirect size {width}×{height}, got {image.shape[1]}×{image.shape[0]}"
    )

  if image.shape[2] == 4:
    return image[..., :3]
  return image


def _sample_equirect(
  image: np.ndarray,
  map_u: np.ndarray,
  map_v: np.ndarray,
) -> np.ndarray:
  if cv2 is not None:
    sampled = cv2.remap(
      image,
      map_u,
      map_v,
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_CONSTANT,
      borderValue=0,
    )
    return sampled

  height, width = image.shape[:2]
  u0 = np.floor(map_u).astype(np.int32)
  v0 = np.floor(map_v).astype(np.int32)
  u1 = (u0 + 1) % width
  v1 = np.clip(v0 + 1, 0, height - 1)

  fu = map_u - u0
  fv = map_v - v0

  valid = (map_v >= 0.0) & (map_v < height)
  u0 = np.mod(u0, width)
  v0 = np.clip(v0, 0, height - 1)

  def gather(u_idx: np.ndarray, v_idx: np.ndarray) -> np.ndarray:
    return image[v_idx, u_idx]

  top = gather(u0, v0) * (1.0 - fu)[..., None] + gather(u1, v0) * fu[..., None]
  bottom = gather(u0, v1) * (1.0 - fu)[..., None] + gather(u1, v1) * fu[..., None]
  sampled = top * (1.0 - fv)[..., None] + bottom * fv[..., None]
  sampled = np.where(valid[..., None], sampled, 0).astype(np.uint8)
  return sampled


def _self_test() -> None:
  projector = EquirectPerspectiveProjector()
  u_center = projector.equirect_width * 0.5
  v_center = projector.equirect_height * 0.5
  ray = equirect_pixel_to_ray(u_center, v_center)
  assert abs(ray[2] - 1.0) < 1e-3, ray

  synthetic = np.zeros(
    (projector.equirect_height, projector.equirect_width, 3),
    dtype=np.uint8,
  )
  synthetic[:, :, 0] = np.linspace(0, 255, projector.equirect_width, dtype=np.uint8)
  synthetic[:, :, 1] = np.linspace(0, 255, projector.equirect_height, dtype=np.uint8)[:, None]

  crops = projector.crop_all(synthetic)
  assert len(crops) == 4
  for crop in crops:
    assert crop.image.shape == (projector.crop_size, projector.crop_size, 3)
    assert crop.ray_map.shape == (projector.crop_size, projector.crop_size, 3)
    center_ray = crop.ray_at(projector.crop_size // 2, projector.crop_size // 2)
    assert np.linalg.norm(center_ray) > 0.99

  print("equirect_to_perspective self-test passed")


if __name__ == "__main__":
  _self_test()
