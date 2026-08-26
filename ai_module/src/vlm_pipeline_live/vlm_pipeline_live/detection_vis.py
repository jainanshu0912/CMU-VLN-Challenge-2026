"""Draw GroundingDINO boxes / labels on perspective crops for debug."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from vlm_pipeline_live.detection_backend import Detection2D


# Distinct-ish colors cycling by detection index (RGB).
_BOX_COLORS = [
  (0, 220, 80),
  (255, 180, 0),
  (0, 180, 255),
  (255, 80, 80),
  (180, 80, 255),
  (255, 255, 0),
  (0, 255, 200),
  (255, 120, 200),
]


def _clamp_box(
  x1: float,
  y1: float,
  x2: float,
  y2: float,
  width: int,
  height: int,
) -> tuple[int, int, int, int]:
  ix1 = int(max(0, min(width - 1, round(x1))))
  iy1 = int(max(0, min(height - 1, round(y1))))
  ix2 = int(max(0, min(width - 1, round(x2))))
  iy2 = int(max(0, min(height - 1, round(y2))))
  if ix2 < ix1:
    ix1, ix2 = ix2, ix1
  if iy2 < iy1:
    iy1, iy2 = iy2, iy1
  return ix1, iy1, ix2, iy2


def draw_detections_on_image(
  image_rgb: np.ndarray,
  detections: Sequence[Detection2D],
  *,
  line_width: int = 3,
) -> np.ndarray:
  """Return a copy of ``image_rgb`` with boxes + labels drawn."""
  try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
  except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PIL is required to draw detection overlays") from exc

  if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
    raise ValueError(f"Expected HxWx3 RGB image, got shape {image_rgb.shape}")

  canvas = PILImage.fromarray(np.asarray(image_rgb, dtype=np.uint8).copy())
  draw = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.load_default()
  except Exception:  # pragma: no cover
    font = None

  height, width = image_rgb.shape[:2]
  for index, det in enumerate(detections):
    color = _BOX_COLORS[index % len(_BOX_COLORS)]
    x1, y1, x2, y2 = _clamp_box(det.x1, det.y1, det.x2, det.y2, width, height)
    for t in range(max(1, line_width)):
      draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

    label = f"{det.label} {det.confidence:.2f}"
    # Text background for readability.
    if font is not None:
      try:
        text_bbox = draw.textbbox((x1, max(0, y1 - 14)), label, font=font)
      except Exception:
        text_bbox = (x1, max(0, y1 - 14), x1 + 8 * len(label), y1)
    else:
      text_bbox = (x1, max(0, y1 - 14), x1 + 8 * len(label), y1)
    draw.rectangle(text_bbox, fill=(0, 0, 0))
    draw.text((text_bbox[0] + 1, text_bbox[1]), label, fill=color, font=font)

  return np.asarray(canvas, dtype=np.uint8)


def draw_projected_points(
  image_rgb: np.ndarray,
  px: np.ndarray,
  py: np.ndarray,
  visible: np.ndarray,
  *,
  color: tuple[int, int, int] = (0, 255, 255),
  radius: int = 4,
  labels: Sequence[str] | None = None,
) -> np.ndarray:
  """Draw projected map/camera points (e.g. fused 3D centers) on a crop."""
  try:
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont
  except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PIL is required to draw projected points") from exc

  canvas = PILImage.fromarray(np.asarray(image_rgb, dtype=np.uint8).copy())
  draw = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.load_default()
  except Exception:  # pragma: no cover
    font = None

  height, width = image_rgb.shape[:2]
  for i, (x, y, ok) in enumerate(zip(px, py, visible)):
    if not bool(ok):
      continue
    ix, iy = int(round(float(x))), int(round(float(y)))
    if ix < 0 or iy < 0 or ix >= width or iy >= height:
      continue
    draw.ellipse(
      [ix - radius, iy - radius, ix + radius, iy + radius],
      outline=color,
      fill=color,
    )
    # Cross-hair for visibility.
    draw.line([ix - radius - 2, iy, ix + radius + 2, iy], fill=color, width=2)
    draw.line([ix, iy - radius - 2, ix, iy + radius + 2], fill=color, width=2)
    if labels is not None and i < len(labels):
      text = str(labels[i])
      ty = max(0, iy - radius - 12)
      draw.text((ix + radius + 2, ty), text, fill=color, font=font)

  return np.asarray(canvas, dtype=np.uint8)


def filter_detections_for_heading(
  detections: Iterable[Detection2D],
  heading_deg: float,
  *,
  tol_deg: float = 0.5,
) -> list[Detection2D]:
  return [
    det for det in detections
    if abs(float(det.heading_deg) - float(heading_deg)) <= tol_deg
  ]
