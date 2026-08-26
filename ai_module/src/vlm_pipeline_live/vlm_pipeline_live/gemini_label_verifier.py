"""Gemini free-tier label verifier for Pipeline B 2D detections.

Flow (per crop)::

  detector boxes  →  Gemini keep / relabel / drop  →  filtered Detection2D list

Uses the Google AI Studio free API (``gemini-3.6-flash`` by default).
Set ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.

This is a *verifier*, not a standalone detector — boxes still come from
GroundingDINO / YOLOE / YOLO-World.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from vlm_pipeline_live.detection_backend import Detection2D, prompt_to_class_list
from vlm_pipeline_live.label_utils import canonicalize_label

LogFn = Callable[[str], None]

DEFAULT_GEMINI_VERIFY_MODEL = os.environ.get(
  "GEMINI_VERIFY_MODEL",
  "gemini-3.6-flash",
)

# Keep each Gemini call small so free-tier responses don't truncate mid-JSON.
DEFAULT_MAX_DETS_PER_CALL = int(os.environ.get("GEMINI_VERIFY_BATCH", "12"))
DEFAULT_MAX_RETRIES = int(os.environ.get("GEMINI_VERIFY_RETRIES", "3"))

_VERIFY_RESPONSE_SCHEMA: dict = {
  "type": "OBJECT",
  "properties": {
    "decisions": {
      "type": "ARRAY",
      "items": {
        "type": "OBJECT",
        "properties": {
          "id": {"type": "INTEGER"},
          "action": {"type": "STRING"},
          "label": {"type": "STRING"},
          "confidence": {"type": "NUMBER"},
        },
        "required": ["id", "action"],
      },
    }
  },
  "required": ["decisions"],
}


@dataclass(frozen=True)
class VerificationDecision:
  det_id: int
  action: str  # keep | relabel | drop
  label: str = ""
  confidence: float | None = None


def _extract_json_blob(text: str) -> str:
  raw = (text or "").strip()
  if not raw:
    raise ValueError("Empty Gemini response")
  fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
  if fenced:
    raw = fenced.group(1).strip()
  start_obj = raw.find("{")
  start_arr = raw.find("[")
  if start_obj < 0 and start_arr < 0:
    raise ValueError(f"No JSON object/array in Gemini response: {raw[:200]!r}")
  if start_obj < 0 or (0 <= start_arr < start_obj):
    start, end_char = start_arr, "]"
  else:
    start, end_char = start_obj, "}"
  end = raw.rfind(end_char)
  if end <= start:
    raise ValueError(f"Truncated JSON in Gemini response: {raw[:200]!r}")
  return raw[start : end + 1]


def _repair_json(blob: str) -> str:
  """Best-effort fixes for common Gemini JSON glitches."""
  fixed = blob.strip()
  # Trailing commas before } or ]
  fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
  # Smart quotes → normal quotes
  fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
  # Single-quoted keys/strings (very rough)
  if "'" in fixed and '"' not in fixed:
    fixed = fixed.replace("'", '"')
  # Odd quote count → close an open string
  if fixed.count('"') % 2 == 1:
    fixed += '"'
  return fixed


def _load_json_lenient(text: str) -> object:
  blob = _extract_json_blob(text)
  try:
    return json.loads(blob)
  except json.JSONDecodeError:
    repaired = _repair_json(blob)
    try:
      return json.loads(repaired)
    except json.JSONDecodeError:
      pass
    # Truncation: close open braces/brackets.
    open_curly = repaired.count("{") - repaired.count("}")
    open_square = repaired.count("[") - repaired.count("]")
    trimmed = repaired.rstrip().rstrip(",")
    trimmed += "]" * max(0, open_square) + "}" * max(0, open_curly)
    try:
      return json.loads(trimmed)
    except json.JSONDecodeError as exc:
      raise ValueError(
        f"Could not parse Gemini JSON ({exc}): {blob[:240]!r}"
      ) from exc


def parse_verification_response(text: str) -> list[VerificationDecision]:
  """Parse Gemini JSON into verification decisions (lenient)."""
  data = _load_json_lenient(text)
  if isinstance(data, dict):
    items = data.get("decisions") or data.get("results") or data.get("items")
    if items is None and "id" in data:
      items = [data]
  elif isinstance(data, list):
    items = data
  else:
    raise ValueError(f"Unexpected JSON type: {type(data)}")

  if not isinstance(items, list):
    raise ValueError("Gemini JSON missing decisions list")

  decisions: list[VerificationDecision] = []
  for item in items:
    if not isinstance(item, dict):
      continue
    try:
      det_id = int(item.get("id", item.get("det_id", -1)))
    except (TypeError, ValueError):
      continue
    action = str(item.get("action", "keep")).strip().lower()
    if action in {"reject", "remove", "delete", "false"}:
      action = "drop"
    if action in {"true", "ok", "accept", "valid"}:
      action = "keep"
    if action in {"rename", "fix", "correct"}:
      action = "relabel"
    if action not in {"keep", "relabel", "drop"}:
      action = "keep"
    label = str(item.get("label", item.get("new_label", "")) or "").strip().lower()
    conf_raw = item.get("confidence", item.get("score"))
    conf: float | None
    try:
      conf = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
      conf = None
    decisions.append(
      VerificationDecision(
        det_id=det_id,
        action=action,
        label=label,
        confidence=conf,
      )
    )
  return decisions


def apply_verification(
  detections: Sequence[Detection2D],
  decisions: Sequence[VerificationDecision],
  *,
  allowed_labels: Sequence[str] | None = None,
) -> list[Detection2D]:
  """Apply keep/relabel/drop decisions. Missing ids default to keep."""
  by_id = {d.det_id: d for d in decisions}
  _ = allowed_labels
  out: list[Detection2D] = []

  for index, det in enumerate(detections):
    decision = by_id.get(index)
    if decision is None:
      out.append(det)
      continue

    if decision.action == "drop":
      continue

    label = det.label
    conf = det.confidence
    if decision.action == "relabel":
      if not decision.label:
        continue
      label = canonicalize_label(decision.label)
      if decision.confidence is not None:
        conf = float(decision.confidence)
    elif decision.action == "keep":
      if decision.label:
        label = canonicalize_label(decision.label)
      if decision.confidence is not None:
        conf = max(float(det.confidence), float(decision.confidence))

    out.append(
      Detection2D(
        label=canonicalize_label(label),
        confidence=float(conf),
        x1=det.x1,
        y1=det.y1,
        x2=det.x2,
        y2=det.y2,
        heading_deg=det.heading_deg,
        crop_width=det.crop_width,
        crop_height=det.crop_height,
      )
    )
  return out


def draw_numbered_detections(
  image_rgb: np.ndarray,
  detections: Sequence[Detection2D],
  *,
  id_offset: int = 0,
) -> np.ndarray:
  """Draw boxes with integer ids so Gemini can reference them."""
  try:
    import cv2
  except ImportError as exc:
    raise RuntimeError("OpenCV (cv2) is required to annotate crops for Gemini.") from exc

  canvas = np.asarray(image_rgb, dtype=np.uint8).copy()
  if canvas.ndim != 3 or canvas.shape[2] != 3:
    raise ValueError(f"Expected H×W×3 RGB image, got {canvas.shape}")
  height, width = canvas.shape[:2]

  for index, det in enumerate(detections):
    det_id = id_offset + index
    x1 = int(max(0, min(width - 1, round(det.x1))))
    y1 = int(max(0, min(height - 1, round(det.y1))))
    x2 = int(max(0, min(width - 1, round(det.x2))))
    y2 = int(max(0, min(height - 1, round(det.y2))))
    color = (0, 220, 80)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    tag = f"{det_id}:{det.label}"
    cv2.putText(
      canvas,
      tag,
      (x1, max(16, y1 - 6)),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      color,
      1,
      cv2.LINE_AA,
    )
  return canvas


def _build_verify_prompt(
  detections: Sequence[Detection2D],
  allowed_labels: Sequence[str],
  *,
  id_offset: int = 0,
) -> str:
  lines = [
    "Verify indoor object detections. Image boxes are tagged id:label.",
    "For EVERY listed id return one decision.",
    "Actions: keep | relabel | drop.",
    "On relabel, set label to a short noun from the vocabulary when possible.",
    "",
    "Vocabulary:",
    ", ".join(allowed_labels[:40]) if allowed_labels else "(common indoor nouns)",
    "",
    "Detections:",
  ]
  for index, det in enumerate(detections):
    det_id = id_offset + index
    lines.append(
      f"- id={det_id} label={det.label!r} conf={det.confidence:.2f} "
      f"box=[{det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f}]"
    )
  lines.append(
    'Respond with JSON only: '
    '{"decisions":[{"id":0,"action":"keep","label":"","confidence":0.9}]}'
  )
  return "\n".join(lines)


def _is_daily_quota(err: str) -> bool:
  return any(
    token in err
    for token in (
      "GenerateRequestsPerDay",
      "PerDayPerProjectPerModel-FreeTier",
    )
  )


def _is_retryable(err: str) -> bool:
  # Daily free-tier exhaustion is NOT worth short retries — disable instead.
  if _is_daily_quota(err):
    return False
  return any(
    token in err
    for token in (
      "503",
      "UNAVAILABLE",
      "high demand",
      "RESOURCE_EXHAUSTED",
      "429",
      "rate limit",
      "Quota",
      "temporarily",
      "DeadlineExceeded",
      "timed out",
      "Timeout",
    )
  )


def _retry_delay_seconds(err: str, fallback: float) -> float:
  match = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", err)
  if match:
    return min(90.0, max(1.0, float(match.group(1)) + 0.25))
  match = re.search(r"'retryDelay':\s*'([0-9]+)s'", err)
  if match:
    return min(90.0, max(1.0, float(match.group(1)) + 0.25))
  return fallback


class GeminiLabelVerifier:
  """Call Gemini Flash to verify / relabel / drop 2D detections on a crop."""

  def __init__(
    self,
    *,
    model: str = DEFAULT_GEMINI_VERIFY_MODEL,
    api_key: str | None = None,
    enabled: bool = True,
    fail_open: bool = True,
    max_dets_per_call: int = DEFAULT_MAX_DETS_PER_CALL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    log_fn: LogFn | None = None,
  ) -> None:
    self.model = (model or DEFAULT_GEMINI_VERIFY_MODEL).strip()
    self._api_key = (
      (api_key or "").strip()
      or os.environ.get("GEMINI_API_KEY", "").strip()
      or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    self.enabled = bool(enabled)
    self.fail_open = bool(fail_open)
    self.max_dets_per_call = max(1, int(max_dets_per_call))
    self.max_retries = max(1, int(max_retries))
    self._log = log_fn or (lambda _msg: None)
    self._disabled_reason: str | None = None
    self._client = None

  @property
  def name(self) -> str:
    return "gemini_verify"

  @property
  def api_key_fingerprint(self) -> str:
    key = self._api_key or ""
    if not key:
      return "(missing)"
    if len(key) <= 8:
      return "(set, short)"
    return f"...{key[-4:]} (len={len(key)})"

  @property
  def is_available(self) -> bool:
    if not self.enabled or not self._api_key or self._disabled_reason:
      return False
    try:
      import google.genai  # noqa: F401
      return True
    except ImportError:
      pass
    try:
      import google.generativeai  # noqa: F401
      return True
    except ImportError:
      return False

  def verify_crop(
    self,
    image_rgb: np.ndarray,
    detections: Sequence[Detection2D],
    prompt: str,
  ) -> list[Detection2D]:
    """Verify detections for one crop. Fail-open returns originals on error."""
    if not detections:
      return []
    if not self.enabled:
      return list(detections)
    if self._disabled_reason:
      return list(detections)
    if not self.is_available:
      msg = (
        "Gemini verifier unavailable (set a valid GEMINI_API_KEY / GOOGLE_API_KEY "
        "from https://aistudio.google.com/apikey and "
        "pip3 install -U --break-system-packages google-genai pillow)."
      )
      if self.fail_open:
        self._log(f"[gemini_verify] {msg} Keeping original boxes.")
        self._disabled_reason = msg
        return list(detections)
      raise RuntimeError(msg)

    allowed = prompt_to_class_list(prompt)
    all_decisions: list[VerificationDecision] = []

    try:
      for start in range(0, len(detections), self.max_dets_per_call):
        batch = list(detections[start : start + self.max_dets_per_call])
        annotated = draw_numbered_detections(image_rgb, batch, id_offset=start)
        text_prompt = _build_verify_prompt(batch, allowed, id_offset=start)
        response_text = self._call_gemini_with_retries(annotated, text_prompt)
        batch_decisions = parse_verification_response(response_text)
        all_decisions.extend(batch_decisions)

      verified = apply_verification(detections, all_decisions, allowed_labels=allowed)
      n_keep = sum(1 for d in all_decisions if d.action == "keep")
      n_relabel = sum(1 for d in all_decisions if d.action == "relabel")
      n_drop = sum(1 for d in all_decisions if d.action == "drop")
      self._log(
        f"[gemini_verify] {len(detections)} → {len(verified)} "
        f"(keep={n_keep} relabel={n_relabel} drop={n_drop}) model={self.model}"
      )
      return verified
    except Exception as exc:
      err = str(exc)
      auth_bad = any(
        token in err
        for token in (
          "API_KEY_INVALID",
          "API key not valid",
          "PERMISSION_DENIED",
        )
      )
      model_gone = any(
        token in err
        for token in (
          "NOT_FOUND",
          "no longer available",
          "is not found",
        )
      )
      quota_day = _is_daily_quota(err)
      if auth_bad or model_gone or quota_day:
        self._disabled_reason = err
        if quota_day:
          hint = (
            "Free-tier Gemini daily quota hit (often ~20 req/model/day). "
            "Options: wait for reset, gemini_verify:=false for mapping, "
            "try another model id with its own quota, or enable billing. "
            "See https://ai.google.dev/gemini-api/docs/rate-limits"
          )
        elif auth_bad:
          hint = "Fix API key: https://aistudio.google.com/apikey"
        else:
          hint = "Fix model: gemini_model:=gemini-3.6-flash"
        self._log(
          f"[gemini_verify] permanently disabling this run "
          f"({err.split(chr(10))[0][:160]}). {hint}"
        )
      elif self.fail_open:
        self._log(
          f"[gemini_verify] failed ({exc}); keeping original boxes."
        )
      if self.fail_open:
        return list(detections)
      raise

  def _call_gemini_with_retries(self, image_rgb: np.ndarray, text_prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, self.max_retries + 1):
      try:
        return self._call_gemini(image_rgb, text_prompt)
      except Exception as exc:
        last_exc = exc
        err = str(exc)
        if _is_daily_quota(err):
          raise
        if attempt < self.max_retries and _is_retryable(err):
          sleep_s = _retry_delay_seconds(err, fallback=min(20.0, 1.5 * (2 ** (attempt - 1))))
          self._log(
            f"[gemini_verify] retry {attempt}/{self.max_retries} after "
            f"{sleep_s:.1f}s ({err.split(chr(10))[0][:120]})"
          )
          time.sleep(sleep_s)
          continue
        raise
    assert last_exc is not None
    raise last_exc

  def _call_gemini(self, image_rgb: np.ndarray, text_prompt: str) -> str:
    from PIL import Image

    pil_image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))

    try:
      from google import genai
      from google.genai import types

      if self._client is None:
        self._client = genai.Client(api_key=self._api_key)

      try:
        config = types.GenerateContentConfig(
          temperature=0.0,
          max_output_tokens=4096,
          response_mime_type="application/json",
          response_schema=_VERIFY_RESPONSE_SCHEMA,
          automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True,
          ),
        )
      except TypeError:
        config = types.GenerateContentConfig(
          temperature=0.0,
          max_output_tokens=4096,
          response_mime_type="application/json",
        )

      response = self._client.models.generate_content(
        model=self.model,
        contents=[text_prompt, pil_image],
        config=config,
      )
      text = getattr(response, "text", None)
      if text:
        return str(text).strip()
      raise RuntimeError("Empty Gemini response (google.genai)")
    except ImportError:
      pass

    try:
      import google.generativeai as genai_old
      from google.generativeai.types import GenerationConfig
    except ImportError as exc:
      raise RuntimeError(
        "google-genai (preferred) or google-generativeai is required. "
        "pip3 install -U --break-system-packages google-genai pillow"
      ) from exc

    genai_old.configure(api_key=self._api_key)
    model = genai_old.GenerativeModel(model_name=self.model)
    response = model.generate_content(
      [text_prompt, pil_image],
      generation_config=GenerationConfig(
        temperature=0.0,
        max_output_tokens=4096,
        response_mime_type="application/json",
      ),
    )
    text = getattr(response, "text", None)
    if not text:
      try:
        text = response.candidates[0].content.parts[0].text
      except Exception as exc:
        raise RuntimeError(f"Empty Gemini response: {exc}") from exc
    return str(text).strip()
