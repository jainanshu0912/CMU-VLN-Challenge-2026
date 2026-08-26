"""GroundingDINO open-vocabulary 2D detection backend."""

from __future__ import annotations

import threading

import numpy as np

try:
  import torch
except ImportError:  # pragma: no cover
  torch = None

from vlm_pipeline_live.detection_backend import Detection2D
from vlm_pipeline_live.label_utils import (
  DEFAULT_INDOOR_PROMPT,
  HOTEL_PROMPT,
  OFFICE_PROMPT,
  canonicalize_label,
  prompt_for_scene_type,
)

# Re-export for existing imports.
__all__ = [
  "DEFAULT_INDOOR_PROMPT",
  "HOTEL_PROMPT",
  "OFFICE_PROMPT",
  "Detection2D",
  "GroundingDinoBackend",
  "canonicalize_label",
  "prompt_for_scene_type",
  "prompt_from_question",
]


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


def _sync_bert_model_warper_import(bertwarper) -> None:
  """Keep groundingdino.py's BertModelWarper name in sync after reload.

  ``from .bertwarper import BertModelWarper`` binds a class object. Reloading
  bertwarper alone leaves that stale binding, so GroundingDINO can still run
  the pre-patch ``__init__`` while the on-disk source already looks fixed.
  """
  try:
    import groundingdino.models.GroundingDINO.groundingdino as gdino
  except ImportError:
    return
  gdino.BertModelWarper = bertwarper.BertModelWarper


def _patch_groundingdino_transformers_compat() -> None:
  """Patch GroundingDINO BertModelWarper for transformers 5.x API changes.

  Transformers 4.x has get_head_mask natively — skip entirely.
  Transformers 5.x removed it — patch the installed bertwarper.py on disk.
  """
  import groundingdino.models.GroundingDINO.bertwarper as bertwarper

  if getattr(bertwarper, "_vlm_transformers_patched_v2", False):
    _sync_bert_model_warper_import(bertwarper)
    return

  from transformers import BertModel as _BertModel
  if hasattr(_BertModel, "get_head_mask"):
    # transformers 4.x: original GroundingDINO code works fine, nothing to do.
    bertwarper._vlm_transformers_patched_v2 = True
    return

  # transformers 5.x: patch the installed bertwarper.py in place.
  import importlib
  import inspect

  bertwarper_path = inspect.getfile(bertwarper)
  print(f"[GroundingDINO] Patching {bertwarper_path} for transformers 5.x ...", flush=True)

  with open(bertwarper_path, encoding="utf-8") as fh:
    src = fh.read()

  # Upgrade any previous getattr-only / bare patch to an explicit try/except.
  # Only rewrite once (_vlm_patched_v2 marker).
  if "_vlm_patched_v2" not in src:
    head_mask_try = (
      "try:  # _vlm_patched_v2\n"
      "            self.get_head_mask = bert_model.get_head_mask\n"
      "        except AttributeError:\n"
      "            self.get_head_mask = None"
    )
    if "getattr(bert_model, 'get_head_mask'" in src:
      src = src.replace(
        "self.get_head_mask = getattr(bert_model, 'get_head_mask', None)  # _vlm_patched_sentinel",
        head_mask_try,
      )
    else:
      src = src.replace(
        "self.get_head_mask = bert_model.get_head_mask",
        head_mask_try,
      )

    # Make the get_extended_attention_mask call accept or drop the `device` arg.
    if "except TypeError:\n            extended_attention_mask = " not in src:
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

    # Guard the get_head_mask call so None is handled.
    if "if self.get_head_mask is not None:" not in src:
      src = src.replace(
        "head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)",
        (
          "if self.get_head_mask is not None:\n"
          "            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)\n"
          "        else:\n"
          "            head_mask = [None] * self.config.num_hidden_layers"
        ),
      )

    try:
      with open(bertwarper_path, "w", encoding="utf-8") as fh:
        fh.write(src)
    except OSError as exc:
      # Container user `docker` cannot write site-packages. Shim in memory instead.
      print(
        f"[GroundingDINO] Cannot write {bertwarper_path} ({exc}); "
        "applying transformers 5.x shim in memory.",
        flush=True,
      )
      _patch_bertwarper_in_memory(bertwarper)
      _sync_bert_model_warper_import(bertwarper)
      bertwarper._vlm_transformers_patched_v2 = True
      return

  # Reload + rebind so already-imported GroundingDINO sees the new class.
  importlib.reload(bertwarper)
  _sync_bert_model_warper_import(bertwarper)

  bertwarper._vlm_transformers_patched_v2 = True
  print("[GroundingDINO] bertwarper.py patched for transformers 5.x.", flush=True)


def _patch_bertwarper_in_memory(bertwarper) -> None:
  """Apply transformers 5.x BertModelWarper shims without rewriting site-packages."""
  cls = bertwarper.BertModelWarper
  orig_init = cls.__init__
  orig_forward = cls.forward

  def init_compat(self, bert_model, *args, **kwargs):
    if not hasattr(bert_model, "get_head_mask"):
      bert_model.get_head_mask = None
    orig_init(self, bert_model, *args, **kwargs)
    self.get_head_mask = getattr(bert_model, "get_head_mask", None)

  def forward_compat(self, *args, **kwargs):
    get_ext = self.get_extended_attention_mask
    get_head = self.get_head_mask

    def get_ext_compat(*a, **k):
      try:
        return get_ext(*a, **k)
      except TypeError:
        if len(a) >= 3:
          a = a[:2]
        k.pop("device", None)
        return get_ext(*a, **k)

    def get_head_compat(head_mask, num_layers):
      if get_head is None:
        return [None] * num_layers
      return get_head(head_mask, num_layers)

    self.get_extended_attention_mask = get_ext_compat
    self.get_head_mask = get_head_compat
    try:
      return orig_forward(self, *args, **kwargs)
    finally:
      self.get_extended_attention_mask = get_ext
      self.get_head_mask = get_head

  cls.__init__ = init_compat
  cls.forward = forward_compat


def _patch_groundingdino_ms_deform_attn_fallback() -> None:
  """Use PyTorch MSDeformAttn when GroundingDINO CUDA extension ``_C`` is missing.

  Pip installs often ship without compiled ops. Upstream then still takes the
  CUDA branch and crashes with ``NameError: _C is not defined``. The container
  user cannot rewrite ``/usr/local/lib/.../ms_deform_attn.py``, so this is an
  in-memory monkey-patch only.
  """
  import groundingdino.models.GroundingDINO.ms_deform_attn as msda

  if getattr(msda, "_vlm_msda_fallback_patched", False):
    return

  if getattr(msda, "_C", None) is not None:
    msda._vlm_msda_fallback_patched = True
    return

  pytorch_attn = msda.multi_scale_deformable_attn_pytorch
  orig_apply = msda.MultiScaleDeformableAttnFunction.apply

  def _apply_with_pytorch_fallback(
    value,
    spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
  ):
    if getattr(msda, "_C", None) is None:
      return pytorch_attn(
        value, spatial_shapes, sampling_locations, attention_weights
      )
    return orig_apply(
      value,
      spatial_shapes,
      level_start_index,
      sampling_locations,
      attention_weights,
      im2col_step,
    )

  msda.MultiScaleDeformableAttnFunction.apply = _apply_with_pytorch_fallback
  msda._vlm_msda_fallback_patched = True
  print(
    "[GroundingDINO] CUDA extension _C is missing; using PyTorch MSDeformAttn "
    "fallback in memory (no site-packages write).",
    flush=True,
  )


def _patch_groundingdino_runtime() -> None:
  """Apply all GroundingDINO install/runtime shims used by Pipeline B."""
  _patch_groundingdino_transformers_compat()
  _patch_groundingdino_ms_deform_attn_fallback()


class GroundingDinoBackend:
  """Lazy-loaded GroundingDINO wrapper."""

  def __init__(
    self,
    config_path: str,
    checkpoint_path: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: str = "cuda",
  ) -> None:
    self.config_path = config_path
    self.checkpoint_path = checkpoint_path
    self.box_threshold = box_threshold
    self.text_threshold = text_threshold
    self.device = (device or "cuda").strip() or "cuda"
    self._model = None
    self._load_lock = threading.Lock()
    if self.is_available:
      _patch_groundingdino_runtime()

  @property
  def name(self) -> str:
    return "grounding_dino"

  @staticmethod
  def require_cuda(device: str) -> None:
    if torch is None:
      raise RuntimeError("PyTorch is required for GroundingDINO (GPU pipeline).")
    if not device.startswith("cuda"):
      raise RuntimeError(
        f"Pipeline B requires a CUDA device (got device={device!r})."
      )
    if not torch.cuda.is_available():
      raise RuntimeError(
        "CUDA is not available. Install CUDA torch and run inside the GPU container."
      )

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

    with self._load_lock:
      if self._model is not None:
        return self._model

      if not self.is_available:
        raise RuntimeError(
          "GroundingDINO is not installed. Install torch and GroundingDINO, then set "
          "model_config_path and model_checkpoint_path parameters."
        )

      self.require_cuda(self.device)

      print(
        f"[GroundingDINO] Loading weights from {self.checkpoint_path} on {self.device} "
        f"(cuda_available={torch.cuda.is_available()})...",
        flush=True,
      )
      print(
        f"[GroundingDINO] GPU: {torch.cuda.get_device_name(0)} "
        f"| {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        flush=True,
      )

      _patch_groundingdino_runtime()

      from groundingdino.util.inference import load_model

      # Official API defaults to device="cuda"; pass our resolved device explicitly.
      try:
        model = load_model(self.config_path, self.checkpoint_path, device=self.device)
      except TypeError:
        model = load_model(self.config_path, self.checkpoint_path)
      model = model.to(self.device)
      model.eval()
      self._model = model
      print(f"[GroundingDINO] Model loaded on {self.device}.", flush=True)
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

    _patch_groundingdino_runtime()
    from groundingdino.util.inference import predict

    model = self._load_model()
    image_tensor = self._preprocess(image_rgb).to(self.device)
    print(
      f"[GroundingDINO] Inference heading={heading_deg:.0f}° on {self.device}...",
      flush=True,
    )

    with torch.inference_mode():
      try:
        boxes, logits, phrases = predict(
          model=model,
          image=image_tensor,
          caption=prompt,
          box_threshold=self.box_threshold,
          text_threshold=self.text_threshold,
          device=self.device,
        )
      except TypeError:
        # Older GroundingDINO builds without a device= argument.
        model = model.to(self.device)
        boxes, logits, phrases = predict(
          model=model,
          image=image_tensor,
          caption=prompt,
          box_threshold=self.box_threshold,
          text_threshold=self.text_threshold,
        )

    if self.device.startswith("cuda") and torch.cuda.is_available():
      torch.cuda.empty_cache()

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
      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.fromarray(image_rgb)
    image_tensor, _ = transform(image_pil, None)
    return image_tensor
