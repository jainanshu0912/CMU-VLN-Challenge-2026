"""Pluggable open-vocabulary 2D detection backends for Pipeline B.

Backends share one API so we can A/B GroundingDINO, YOLO-World, Gemini, etc.::

  detector = create_detection_backend("yolo_world", device="cuda")
  boxes = detector.detect(crop_rgb, prompt, heading_deg=0.0)

``prompt`` is the same dotted caption used today::

  "chair . desk . lamp . monitor"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection2D:
  label: str
  confidence: float
  x1: float
  y1: float
  x2: float
  y2: float
  heading_deg: float
  crop_width: int
  crop_height: int


@runtime_checkable
class OpenVocabDetector(Protocol):
  """Minimal contract every 2D backend must implement."""

  @property
  def name(self) -> str: ...

  @property
  def device(self) -> str: ...

  @property
  def is_available(self) -> bool: ...

  def detect(
    self,
    image_rgb: np.ndarray,
    prompt: str,
    heading_deg: float,
  ) -> list[Detection2D]: ...


def prompt_to_class_list(prompt: str) -> list[str]:
  """Split a GroundingDINO-style caption into YOLO/Gemini class names."""
  classes: list[str] = []
  for raw in (prompt or "").split("."):
    label = " ".join(raw.strip().lower().split())
    if label and label not in classes:
      classes.append(label)
  return classes


def create_detection_backend(
  backend: str,
  *,
  device: str = "cuda",
  box_threshold: float = 0.35,
  text_threshold: float = 0.25,
  model_config_path: str = "",
  model_checkpoint_path: str = "",
  yolo_model: str = "",
) -> OpenVocabDetector:
  """Factory for open-vocab detectors.

  Supported ``backend`` values:
    - ``grounding_dino`` / ``dino``
    - ``yolo_world`` / ``yolo`` / ``yolov8_world``  (Ultralytics YOLO-World v2)
    - ``yoloe`` / ``yolov11`` / ``yolov11_world``     (Ultralytics YOLOE on YOLO11)
    - ``owlvit`` / ``owl_vit`` / ``owlv2``            (Hugging Face OWL-ViT / v2)
    - ``gemini`` (stub — not implemented yet)
  """
  key = (backend or "grounding_dino").strip().lower().replace("-", "_")

  if key in {"grounding_dino", "dino", "groundingdino"}:
    from vlm_pipeline_live.grounding_dino_backend import GroundingDinoBackend

    return GroundingDinoBackend(
      config_path=model_config_path,
      checkpoint_path=model_checkpoint_path,
      box_threshold=box_threshold,
      text_threshold=text_threshold,
      device=device,
    )

  if key in {"yolo_world", "yolo", "yolov8_world", "yolo_world_v2"}:
    from vlm_pipeline_live.yolo_world_backend import (
      DEFAULT_YOLO_WORLD_MODEL,
      YoloWorldBackend,
    )

    return YoloWorldBackend(
      model_name=yolo_model or DEFAULT_YOLO_WORLD_MODEL,
      device=device,
      conf_threshold=box_threshold,
    )

  if key in {"yoloe", "yolov11", "yolo_v11", "yolov11_world", "yolo11_world"}:
    from vlm_pipeline_live.yoloe_backend import DEFAULT_YOLOE_MODEL, YoloEBackend

    return YoloEBackend(
      model_name=yolo_model or DEFAULT_YOLOE_MODEL,
      device=device,
      conf_threshold=box_threshold,
    )

  if key in {"owlvit", "owl_vit", "owlv2", "owl", "google_owlvit"}:
    from vlm_pipeline_live.owlvit_backend import DEFAULT_OWLVIT_MODEL, OwlVitBackend

    return OwlVitBackend(
      model_name=yolo_model or DEFAULT_OWLVIT_MODEL,
      device=device,
      conf_threshold=box_threshold,
    )

  if key in {"gemini", "google_gemini"}:
    raise NotImplementedError(
      "Gemini is not a standalone 2D detector in Pipeline B. "
      "Use detector_backend:=grounding_dino|yoloe|yolo_world|owlvit and enable "
      "label verification with gemini_verify:=true (Gemini Flash)."
    )

  raise ValueError(
    f"Unknown detector_backend={backend!r}. "
    "Choose grounding_dino | yolo_world | yoloe | owlvit "
    "(use gemini_verify:=true for Gemini label verification)."
  )
