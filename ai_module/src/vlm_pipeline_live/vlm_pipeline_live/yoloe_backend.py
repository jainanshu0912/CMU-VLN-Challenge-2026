"""YOLOE open-vocabulary 2D detection backend (Ultralytics YOLO11-family).

YOLOE is Ultralytics' successor to YOLO-World and ships YOLO11-scale
checkpoints (e.g. ``yoloe-11s-seg.pt``). API matches YOLO-World::

  backend = YoloEBackend(model_name="yoloe-11s-seg.pt", device="cuda")
  dets = backend.detect(crop_rgb, "chair . desk . lamp", heading_deg=0.0)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vlm_pipeline_live.detection_backend import Detection2D, prompt_to_class_list

DEFAULT_YOLOE_MODEL = os.environ.get(
  "YOLOE_MODEL",
  "yoloe-11s-seg.pt",
)


class YoloEBackend:
  """Lazy-loaded Ultralytics YOLOE wrapper (text-prompt / open-vocab)."""

  def __init__(
    self,
    model_name: str = DEFAULT_YOLOE_MODEL,
    *,
    device: str = "cuda",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
  ) -> None:
    self.model_name = (model_name or DEFAULT_YOLOE_MODEL).strip()
    self.device = (device or "cuda").strip() or "cuda"
    self.conf_threshold = float(conf_threshold)
    self.iou_threshold = float(iou_threshold)
    self._model: Any = None
    self._last_classes: list[str] | None = None

  @property
  def name(self) -> str:
    return "yoloe"

  @property
  def is_available(self) -> bool:
    try:
      from ultralytics import YOLOE  # noqa: F401
      return True
    except ImportError:
      return False

  def _require_cuda(self) -> None:
    try:
      import torch
    except ImportError as exc:
      raise RuntimeError("PyTorch is required for YOLOE.") from exc
    if not self.device.startswith("cuda"):
      raise RuntimeError(
        f"Pipeline B requires a CUDA device (got device={self.device!r})."
      )
    if not torch.cuda.is_available():
      raise RuntimeError(
        "CUDA is not available. Install CUDA torch and run inside the GPU container."
      )

  def _ensure_clip(self) -> None:
    """YOLOE text prompts need ultralytics CLIP (PEP 668 blocks auto-pip)."""
    # So Ultralytics check_requirements can succeed if it retries install later.
    os.environ.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")
    try:
      import clip  # noqa: F401
    except ImportError as exc:
      raise RuntimeError(
        "YOLOE needs Ultralytics CLIP for set_classes(), but it is not installed.\n"
        "Inside the AI container run:\n"
        "  pip3 install -U --break-system-packages "
        "git+https://github.com/ultralytics/CLIP.git\n"
        "Then restart the launch."
      ) from exc

  def _load_model(self) -> Any:
    if self._model is not None:
      return self._model

    if not self.is_available:
      raise RuntimeError(
        "ultralytics YOLOE is not available. Inside the AI container run:\n"
        "  pip3 install -U --break-system-packages ultralytics\n"
        "Then use a text-prompt checkpoint such as yoloe-11s-seg.pt "
        "(not *-seg-pf.pt prompt-free weights)."
      )

    self._require_cuda()
    self._ensure_clip()
    from ultralytics import YOLOE

    print(
      f"[YOLOE] Loading {self.model_name} on {self.device}...",
      flush=True,
    )
    model = YOLOE(self.model_name)
    try:
      model.to(self.device)
    except Exception:
      pass
    self._model = model
    print(f"[YOLOE] Model ready on {self.device}.", flush=True)
    return self._model

  def _set_classes(self, classes: list[str]) -> None:
    if not classes:
      raise ValueError("YOLOE prompt produced an empty class list.")
    if self._last_classes == classes:
      return
    model = self._load_model()
    model.set_classes(classes)
    self._last_classes = list(classes)
    print(
      f"[YOLOE] set_classes ({len(classes)}): {', '.join(classes[:12])}"
      f"{'...' if len(classes) > 12 else ''}",
      flush=True,
    )

  def detect(
    self,
    image_rgb: np.ndarray,
    prompt: str,
    heading_deg: float,
  ) -> list[Detection2D]:
    """Run YOLOE on one RGB crop (boxes only; masks ignored for fusion)."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError(f"Expected H×W×3 RGB image, got {image_rgb.shape}")

    classes = prompt_to_class_list(prompt)
    self._set_classes(classes)
    model = self._load_model()

    image_bgr = image_rgb[:, :, ::-1]
    height, width = image_rgb.shape[:2]
    print(
      f"[YOLOE] Inference heading={heading_deg:.0f}° on {self.device} "
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
      label = str(
        names.get(
          int(cls_id),
          classes[int(cls_id)] if 0 <= int(cls_id) < len(classes) else "object",
        )
      )
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
