"""OWL-ViT / OWL-ViT v2 zero-shot 2D detection backend (Hugging Face).

Default checkpoint is OWL-ViT v2 base (better open-vocab than v1)::

  backend = OwlVitBackend(model_name="google/owlv2-base-patch16", device="cuda")
  dets = backend.detect(crop_rgb, "chair . desk . lamp", heading_deg=0.0)

Requires ``transformers`` + ``torch`` (already in the AI GPU image) and a
one-time Hugging Face download of the weights.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vlm_pipeline_live.detection_backend import Detection2D, prompt_to_class_list

DEFAULT_OWLVIT_MODEL = os.environ.get(
  "OWLVIT_MODEL",
  "google/owlv2-base-patch16",
)


class OwlVitBackend:
  """Lazy-loaded Hugging Face OWL-ViT / OWL-ViT v2 wrapper."""

  def __init__(
    self,
    model_name: str = DEFAULT_OWLVIT_MODEL,
    *,
    device: str = "cuda",
    conf_threshold: float = 0.2,
    use_photo_prefix: bool = True,
  ) -> None:
    self.model_name = (model_name or DEFAULT_OWLVIT_MODEL).strip()
    self.device = (device or "cuda").strip() or "cuda"
    self.conf_threshold = float(conf_threshold)
    self.use_photo_prefix = bool(use_photo_prefix)
    if self.conf_threshold >= 0.3:
      print(
        f"[OWL-ViT] conf_threshold={self.conf_threshold:.2f} is high for OWL scores; "
        "try box_threshold:=0.2 if recall is low.",
        flush=True,
      )
    self._model: Any = None
    self._processor: Any = None
    self._is_v2 = "owlv2" in self.model_name.lower()

  @property
  def name(self) -> str:
    return "owlvit"

  @property
  def is_available(self) -> bool:
    try:
      import transformers  # noqa: F401
      import torch  # noqa: F401
      from PIL import Image  # noqa: F401
      return True
    except ImportError:
      return False

  def _require_cuda(self) -> None:
    try:
      import torch
    except ImportError as exc:
      raise RuntimeError("PyTorch is required for OWL-ViT.") from exc
    if not self.device.startswith("cuda"):
      raise RuntimeError(
        f"Pipeline B requires a CUDA device (got device={self.device!r})."
      )
    if not torch.cuda.is_available():
      raise RuntimeError(
        "CUDA is not available. Install CUDA torch and run inside the GPU container."
      )

  def _load(self) -> tuple[Any, Any]:
    if self._model is not None and self._processor is not None:
      return self._model, self._processor

    if not self.is_available:
      raise RuntimeError(
        "OWL-ViT needs transformers, torch, and pillow. Inside the AI container:\n"
        "  pip3 install -U --break-system-packages transformers pillow"
      )

    self._require_cuda()
    import torch

    print(
      f"[OWL-ViT] Loading {self.model_name} on {self.device}...",
      flush=True,
    )
    if self._is_v2:
      from transformers import Owlv2ForObjectDetection, Owlv2Processor

      processor = Owlv2Processor.from_pretrained(self.model_name)
      model = Owlv2ForObjectDetection.from_pretrained(self.model_name)
    else:
      from transformers import OwlViTForObjectDetection, OwlViTProcessor

      processor = OwlViTProcessor.from_pretrained(self.model_name)
      model = OwlViTForObjectDetection.from_pretrained(self.model_name)

    model.to(self.device)
    model.eval()
    self._processor = processor
    self._model = model
    print(
      f"[OWL-ViT] Ready on {self.device} "
      f"(cuda={torch.cuda.is_available()}).",
      flush=True,
    )
    return self._model, self._processor

  def _queries(self, classes: list[str]) -> list[str]:
    if not self.use_photo_prefix:
      return list(classes)
    queries: list[str] = []
    for label in classes:
      if label.startswith("a photo of"):
        queries.append(label)
      else:
        article = "an" if label[:1] in "aeiou" else "a"
        queries.append(f"a photo of {article} {label}")
    return queries

  def detect(
    self,
    image_rgb: np.ndarray,
    prompt: str,
    heading_deg: float,
  ) -> list[Detection2D]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError(f"Expected H×W×3 RGB image, got {image_rgb.shape}")

    classes = prompt_to_class_list(prompt)
    if not classes:
      return []

    from PIL import Image
    import torch

    model, processor = self._load()
    height, width = image_rgb.shape[:2]
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
    queries = self._queries(classes)

    print(
      f"[OWL-ViT] Inference heading={heading_deg:.0f}° on {self.device} "
      f"(conf={self.conf_threshold:.2f}, queries={len(queries)})...",
      flush=True,
    )

    inputs = processor(text=[queries], images=image, return_tensors="pt")
    inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
      outputs = model(**inputs)

    target_sizes = torch.tensor([(height, width)], device=self.device)
    post = getattr(processor, "post_process_object_detection", None)
    if post is None:
      post = getattr(processor, "post_process_grounded_object_detection", None)
    if post is None:
      raise RuntimeError("Processor has no OWL-ViT post_process_* helper")
    try:
      results = post(
        outputs=outputs,
        threshold=self.conf_threshold,
        target_sizes=target_sizes,
      )
    except TypeError:
      results = post(outputs, target_sizes, self.conf_threshold)
    if not results:
      return []

    result = results[0]
    boxes = result.get("boxes")
    scores = result.get("scores")
    labels = result.get("labels")
    if boxes is None or len(boxes) == 0:
      return []

    detections: list[Detection2D] = []
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)
    for (x1, y1, x2, y2), score, cls_id in zip(boxes_np, scores_np, labels_np):
      if 0 <= int(cls_id) < len(classes):
        label = classes[int(cls_id)]
      else:
        label = "object"
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
