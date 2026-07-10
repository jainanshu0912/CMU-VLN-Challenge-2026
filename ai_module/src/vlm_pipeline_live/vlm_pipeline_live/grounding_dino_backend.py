"""GroundingDINO open-vocabulary 2D detection backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
  import torch
except ImportError:  # pragma: no cover
  torch = None


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


DEFAULT_INDOOR_PROMPT = (
  "chair . sofa . table . pillow . book . lamp . tv . monitor . plant . stool . "
  "bed . desk . cabinet . shelf . bottle . cup . bowl . window . door . picture . "
  "clock . keyboard . mouse . laptop . trash can . box . basket . curtain . mirror . "
  "rug . cushion . vase . candle . remote . phone . plate . jar . bin . ottoman"
)

# Shorter caption for CPU dev runs (4 crops × fewer classes = faster).
CPU_TEST_PROMPT = "chair . sofa . table . pillow . book . lamp . plant . tv . stool"


def prompt_from_question(question: str) -> str:
  """Build a GroundingDINO caption from a challenge question."""
  stopwords = {
    "find", "the", "a", "an", "how", "many", "count", "go", "to", "near", "and",
    "or", "on", "in", "at", "of", "with", "from", "that", "this", "is", "are",
    "stop", "closest", "farthest", "between", "below", "above", "beside", "under",
    "over", "by", "into", "through", "for", "your", "left", "right", "front", "back",
  }
  tokens: list[str] = []
  for raw in question.replace(",", " ").replace(".", " ").split():
    word = raw.strip().lower()
    if len(word) < 2 or word in stopwords or word.isdigit():
      continue
    if word not in tokens:
      tokens.append(word)
  if not tokens:
    return DEFAULT_INDOOR_PROMPT
  return " . ".join(tokens)


def _patch_groundingdino_transformers_compat() -> None:
  """Patch GroundingDINO BertModelWarper for transformers 5.x API changes.

  Transformers 4.x has get_head_mask natively — skip entirely.
  Transformers 5.x removed it — patch the installed bertwarper.py on disk.
  """
  import groundingdino.models.GroundingDINO.bertwarper as bertwarper

  if getattr(bertwarper, "_vlm_transformers_patched", False):
    return

  from transformers import BertModel as _BertModel
  if hasattr(_BertModel, "get_head_mask"):
    # transformers 4.x: original GroundingDINO code works fine, nothing to do.
    bertwarper._vlm_transformers_patched = True
    return

  # transformers 5.x: patch the installed bertwarper.py in place.
  import inspect
  import os

  bertwarper_path = inspect.getfile(bertwarper)
  print(f"[GroundingDINO] Patching {bertwarper_path} for transformers 5.x ...", flush=True)

  with open(bertwarper_path, encoding="utf-8") as fh:
    src = fh.read()

  if "_vlm_patched_sentinel" in src:
    bertwarper._vlm_transformers_patched = True
    return

  # 1. Replace the bare attribute access that crashes on transformers 5.x.
  src = src.replace(
    "self.get_head_mask = bert_model.get_head_mask",
    "self.get_head_mask = getattr(bert_model, 'get_head_mask', None)  # _vlm_patched_sentinel",
  )

  # 2. Make the get_extended_attention_mask call accept or drop the `device` arg.
  src = src.replace(
    "extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(\n"
    "            attention_mask, input_shape, device\n"
    "        )",
    (
      "try:\n"
      "            extended_attention_mask: torch.Tensor = "
      "self.get_extended_attention_mask(attention_mask, input_shape, device)\n"
      "        except TypeError:\n"
      "            extended_attention_mask = "
      "self.get_extended_attention_mask(attention_mask, input_shape)"
    ),
  )

  # 3. Guard the get_head_mask call so None is handled.
  src = src.replace(
    "head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)",
    (
      "if self.get_head_mask is not None:\n"
      "            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)\n"
      "        else:\n"
      "            head_mask = [None] * self.config.num_hidden_layers"
    ),
  )

  with open(bertwarper_path, "w", encoding="utf-8") as fh:
    fh.write(src)

  # Reload the module so the running process picks up the changes.
  import importlib
  importlib.reload(bertwarper)

  bertwarper._vlm_transformers_patched = True
  print("[GroundingDINO] bertwarper.py patched for transformers 5.x.", flush=True)


class GroundingDinoBackend:
  """Lazy-loaded GroundingDINO wrapper."""

  def __init__(
    self,
    config_path: str,
    checkpoint_path: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: str = "",
    force_cpu: bool = False,
  ) -> None:
    self.config_path = config_path
    self.checkpoint_path = checkpoint_path
    self.box_threshold = box_threshold
    self.text_threshold = text_threshold
    self.force_cpu = force_cpu
    self.device = device or self._default_device(force_cpu)
    self._model = None
    if self.is_available:
      _patch_groundingdino_transformers_compat()

  @staticmethod
  def _default_device(force_cpu: bool = False) -> str:
    if force_cpu:
      return "cpu"
    if torch is not None and torch.cuda.is_available():
      return "cuda"
    return "cpu"

  @property
  def is_available(self) -> bool:
    try:
      import groundingdino  # noqa: F401
      return torch is not None
    except ImportError:
      return False

  def _load_model(self):
    if self._model is not None:
      return self._model

    if not self.is_available:
      raise RuntimeError(
        "GroundingDINO is not installed. Install torch and GroundingDINO, then set "
        "model_config_path and model_checkpoint_path parameters."
      )

    import sys
    print(
      f"[GroundingDINO] Loading weights from {self.checkpoint_path} on {self.device} "
      "(first run may take 1–3 min)...",
      flush=True,
    )

    _patch_groundingdino_transformers_compat()

    from groundingdino.util.inference import load_model

    model = load_model(self.config_path, self.checkpoint_path)
    if self.device != "cpu":
      model = model.to(self.device)
    self._model = model
    print("[GroundingDINO] Model loaded.", flush=True)
    return self._model

  def detect(
    self,
    image_rgb: np.ndarray,
    prompt: str,
    heading_deg: float,
  ) -> list[Detection2D]:
    """Run GroundingDINO on one RGB crop."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError(f"Expected H×W×3 RGB image, got {image_rgb.shape}")

    _patch_groundingdino_transformers_compat()
    from groundingdino.util.inference import predict

    model = self._load_model()
    image_tensor = self._preprocess(image_rgb).to(self.device)
    print(
      f"[GroundingDINO] Running inference heading={heading_deg:.0f}° on {self.device} "
      f"(CPU: often 2–8 min per crop)...",
      flush=True,
    )
    boxes, logits, phrases = predict(
      model=model,
      image=image_tensor,
      caption=prompt,
      box_threshold=self.box_threshold,
      text_threshold=self.text_threshold,
    )

    height, width = image_rgb.shape[:2]
    detections: list[Detection2D] = []
    for box, score, phrase in zip(boxes, logits, phrases):
      cx, cy, bw, bh = box.tolist()
      x1 = max(0.0, (cx - bw / 2.0) * width)
      y1 = max(0.0, (cy - bh / 2.0) * height)
      x2 = min(float(width), (cx + bw / 2.0) * width)
      y2 = min(float(height), (cy + bh / 2.0) * height)
      if x2 <= x1 or y2 <= y1:
        continue

      label = phrase.strip().lower()
      detections.append(
        Detection2D(
          label=label,
          confidence=float(score),
          x1=x1,
          y1=y1,
          x2=x2,
          y2=y2,
          heading_deg=heading_deg,
          crop_width=width,
          crop_height=height,
        )
      )
    return detections

  @staticmethod
  def _preprocess(image_rgb: np.ndarray):
    from PIL import Image
    import groundingdino.datasets.transforms as T

    transform = T.Compose([
      T.RandomResize([800], max_size=1333),
      T.ToTensor(),
      T.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = transform(image_pil, None)
    return image_tensor
