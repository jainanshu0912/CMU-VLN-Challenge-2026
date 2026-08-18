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

    with open(bertwarper_path, "w", encoding="utf-8") as fh:
      fh.write(src)

  # Reload + rebind so already-imported GroundingDINO sees the new class.
  importlib.reload(bertwarper)
  _sync_bert_model_warper_import(bertwarper)

  bertwarper._vlm_transformers_patched_v2 = True
  print("[GroundingDINO] bertwarper.py patched for transformers 5.x.", flush=True)


def _patch_groundingdino_ms_deform_attn_fallback() -> None:
  """Use PyTorch MSDeformAttn when GroundingDINO CUDA extension ``_C`` is missing.

  Pip installs often ship without the compiled ops. Upstream then still takes the
  CUDA branch and crashes with ``NameError: _C is not defined``.
  """
  import importlib
  import inspect

  import groundingdino.models.GroundingDINO.ms_deform_attn as msda

  if getattr(msda, "_vlm_msda_fallback_patched", False):
    return

  path = inspect.getfile(msda)
  with open(path, encoding="utf-8") as fh:
    src = fh.read()

  if "_vlm_msda_fallback" not in src:
    old = "if torch.cuda.is_available() and value.is_cuda:"
    new = (
      "if torch.cuda.is_available() and value.is_cuda "
      "and globals().get('_C') is not None:  # _vlm_msda_fallback"
    )
    if old not in src:
      print(
        "[GroundingDINO] ms_deform_attn.py CUDA-branch pattern not found; "
        "skipping _C fallback patch.",
        flush=True,
      )
      msda._vlm_msda_fallback_patched = True
      return
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(src.replace(old, new, 1))
    print(f"[GroundingDINO] Patched {path} to fall back without _C ops.", flush=True)

  importlib.reload(msda)
  # Keep transformer module's imported MSDA class in sync if already loaded.
  try:
    import groundingdino.models.GroundingDINO.transformer as transformer
    transformer.MSDeformAttn = msda.MultiScaleDeformableAttention
  except Exception:
    pass
  msda._vlm_msda_fallback_patched = True


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
      _patch_groundingdino_runtime()

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

    print(
      f"[GroundingDINO] Loading weights from {self.checkpoint_path} on {self.device} "
      f"(cuda_available={torch.cuda.is_available()})...",
      flush=True,
    )
    if self.device.startswith("cuda") and torch.cuda.is_available():
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
    timing_hint = (
      "GPU: typically a few seconds per crop"
      if self.device.startswith("cuda")
      else "CPU: often 2–8 min per crop"
    )
    print(
      f"[GroundingDINO] Inference heading={heading_deg:.0f}° on {self.device} "
      f"({timing_hint})...",
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
