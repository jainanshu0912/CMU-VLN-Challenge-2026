"""Unit tests for the FastDDS-fallback filesystem handshake."""

from __future__ import annotations

from pathlib import Path

from vlm_pipeline_live import process_handshake as hs


def test_bump_and_read(tmp_path, monkeypatch):
  monkeypatch.setattr(hs, "HANDSHAKE_DIR", Path(tmp_path))
  assert hs.read_token(hs.RUN_DETECTION) == 0
  assert hs.bump(hs.RUN_DETECTION) == 1
  assert hs.bump(hs.RUN_DETECTION) == 2
  assert hs.read_token(hs.RUN_DETECTION) == 2
  hs.write_text(hs.DETECTIONS_JSON, '{"objects":[]}')
  assert hs.read_text(hs.DETECTIONS_JSON) == '{"objects":[]}'
