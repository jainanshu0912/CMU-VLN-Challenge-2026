"""YOLO-World open-vocabulary 2D detection backend (Ultralytics).

Uses ``ultralytics.YOLOWorld`` (open-vocab World models). Default weight is
``yolov8s-worldv2.pt`` (Ultralytics' current World line). Override with
``yolo_model:=yolov8m-worldv2.pt`` / larger for accuracy.

API::

  backend = YoloWorldBackend(device="cuda")
  dets = backend.detect(crop_rgb, "chair . desk . lamp", heading_deg=0.0)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vlm_pipeline_live.detection_backend import Detection2D, prompt_to_class_list

DEFAULT_YOLO_WORLD_MODEL = os.environ.get(
  "YOLO_WORLD_MODEL",
  "yolov8s-worldv2.pt",
)


class YoloWorldBackend:
  """Lazy-loaded Ultralytics YOLO-World wrapper."""

  def __init__(
    self,
    model_name: str = DEFAULT_YOLO_WORLD_MODEL,
    *,
    device: str = "cuda",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
  ) -> None:
    self.model_name = (model_name or DEFAULT_YOLO_WORLD_MODEL).strip()
    self.device = (device or "cuda").strip() or "cuda"
    self.conf_threshold = float(conf_threshold)
    self.iou_threshold = float(iou_threshold)
    self._model: Any = None
    self._last_classes: list[str] | None = None

  @property
  def name(self) -> str:
    return "yolo_world"

  @property
  def is_available(self) -> bool:
    try:
      import ultralytics  # noqa: F401
      return True
    except ImportError:
      return False

  def _require_cuda(self) -> None:
    try:
      import torch
    except ImportError as exc:
      raise RuntimeError("PyTorch is required for YOLO-World.") from exc
    if not self.device.startswith("cuda"):
      raise RuntimeError(
        f"Pipeline B requires a CUDA device (got device={self.device!r})."
      )
    if not torch.cuda.is_available():
      raise RuntimeError(
        "CUDA is not available. Install CUDA torch and run inside the GPU container."
      )

  def _load_model(self) -> Any:
    if self._model is not None:
      return self._model

    if not self.is_available:
      raise RuntimeError(
        "ultralytics is not installed. Inside the AI container run:\n"
        "  pip install -U ultralytics"
      )

    self._require_cuda()
    from ultralytics import YOLOWorld

    print(
      f"[YOLO-World] Loading {self.model_name} on {self.device}...",
      flush=True,
    )
    model = YOLOWorld(self.model_name)
    # Move / bind device for predict calls.
    try:
      model.to(self.device)
    except Exception:
      pass
    self._model = model
    print(f"[YOLO-World] Model ready on {self.device}.", flush=True)
    return self._model

  def _set_classes(self, classes: list[str]) -> None:
    if not classes:
      raise ValueError("YOLO-World prompt produced an empty class list.")
    if self._last_classes == classes:
      return
    model = self._load_model()
    model.set_classes(classes)
    self._last_classes = list(classes)
    print(
      f"[YOLO-World] set_classes ({len(classes)}): {', '.join(classes[:12])}"
      f"{'...' if len(classes) > 12 else ''}",
      flush=True,
    )

  def detect(
    self,
    image_rgb: np.ndarray,
    prompt: str,
    heading_deg: float,
  ) -> list[Detection2D]:
    """Run YOLO-World on one RGB crop."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError(f"Expected H×W×3 RGB image, got {image_rgb.shape}")

    classes = prompt_to_class_list(prompt)
    self._set_classes(classes)
    model = self._load_model()

    # Ultralytics expects BGR ndarray for numpy inputs.
    image_bgr = image_rgb[:, :, ::-1]
    height, width = image_rgb.shape[:2]
    print(
      f"[YOLO-World] Inference heading={heading_deg:.0f}° on {self.device} "
      f"(conf={self.conf_threshold:.2f})...",
      flush=True,
    )

    results = model.predict(
      source=image_bgr,
      conf=self.conf_threshold,
      iou=self.iou_threshold,
      device=self.device,
      verbose=False,
    )
    if not results:
      return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
      return []

    names = result.names if isinstance(result.names, dict) else {}
    detections: list[Detection2D] = []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)

    for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
      label = str(names.get(int(cls_id), classes[int(cls_id)] if 0 <= int(cls_id) < len(classes) else "object"))
      label = label.strip().lower()
      x1f = float(max(0.0, min(width - 1.0, x1)))
      y1f = float(max(0.0, min(height - 1.0, y1)))
      x2f = float(max(0.0, min(float(width), x2)))
      y2f = float(max(0.0, min(float(height), y2)))
      if x2f <= x1f or y2f <= y1f:
        continue
      detections.append(
        Detection2D(
          label=label,
          confidence=float(score),
          x1=x1f,
          y1=y1f,
          x2=x2f,
          y2=y2f,
          heading_deg=float(heading_deg),
          crop_width=width,
          crop_height=height,
        )
      )
    return detections
