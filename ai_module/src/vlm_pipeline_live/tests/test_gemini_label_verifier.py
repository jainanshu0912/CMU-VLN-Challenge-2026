"""Unit tests for Gemini label verification helpers (no API calls)."""

from __future__ import annotations

from vlm_pipeline_live.detection_backend import Detection2D
from vlm_pipeline_live.gemini_label_verifier import (
  VerificationDecision,
  apply_verification,
  parse_verification_response,
)


def _det(label: str, conf: float = 0.8) -> Detection2D:
  return Detection2D(
    label=label,
    confidence=conf,
    x1=10,
    y1=10,
    x2=100,
    y2=100,
    heading_deg=0.0,
    crop_width=640,
    crop_height=640,
  )


def test_parse_verification_response_object():
  text = """
  ```json
  {"decisions":[
    {"id":0,"action":"keep"},
    {"id":1,"action":"relabel","label":"monitor","confidence":0.9},
    {"id":2,"action":"drop"}
  ]}
  ```
  """
  decisions = parse_verification_response(text)
  assert [d.action for d in decisions] == ["keep", "relabel", "drop"]
  assert decisions[1].label == "monitor"


def test_parse_aliases():
  text = '{"decisions":[{"id":0,"action":"reject"},{"id":1,"action":"rename","label":"chair"}]}'
  decisions = parse_verification_response(text)
  assert decisions[0].action == "drop"
  assert decisions[1].action == "relabel"


def test_apply_verification_keep_relabel_drop():
  dets = [_det("table"), _det("tv"), _det("noise")]
  decisions = [
    VerificationDecision(0, "keep"),
    VerificationDecision(1, "relabel", label="monitor", confidence=0.95),
    VerificationDecision(2, "drop"),
  ]
  out = apply_verification(dets, decisions)
  assert len(out) == 2
  assert out[0].label == "table"
  assert out[1].label == "monitor"
  assert out[1].confidence == 0.95


def test_parse_trailing_comma_and_truncation():
  text = '{"decisions":[{"id":0,"action":"keep",},{"id":1,"action":"drop",}]}'
  decisions = parse_verification_response(text)
  assert [d.action for d in decisions] == ["keep", "drop"]


def test_parse_truncated_closes():
  # Missing closing braces — repair should still recover first decision if possible.
  text = '{"decisions":[{"id":0,"action":"keep","label":"chair"'
  decisions = parse_verification_response(text)
  assert decisions[0].action == "keep"


def test_apply_missing_id_keeps():
  dets = [_det("chair"), _det("desk")]
  out = apply_verification(dets, [VerificationDecision(0, "keep")])
  assert len(out) == 2
