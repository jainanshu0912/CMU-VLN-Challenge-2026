"""Filesystem handshake when FastDDS does not match RELIABLE /vlm_live/* topics.

Camera / scan / odom use BEST_EFFORT and arrive. Intra-AI RELIABLE Bool topics
often stay at subscribers=0 in the eval-style container (no ipc:host / UDP XML).
"""

from __future__ import annotations

from pathlib import Path

HANDSHAKE_DIR = Path("/tmp/vlm_live_handshake")
RUN_DETECTION = "run_detection"
DETECTION_COMPLETE = "detection_complete"
DETECTIONS_JSON = "detections.json"
SCENE_GRAPH_COMPLETE = "scene_graph_complete"
EXPLORATION_COMPLETE = "exploration_complete"


def _path(name: str) -> Path:
  HANDSHAKE_DIR.mkdir(parents=True, exist_ok=True)
  return HANDSHAKE_DIR / name


def write_token(name: str, token: int) -> None:
  _path(name).write_text(str(int(token)), encoding="utf-8")


def read_token(name: str) -> int:
  path = _path(name)
  if not path.exists():
    return 0
  try:
    return int(path.read_text(encoding="utf-8").strip() or "0")
  except ValueError:
    return 0


def bump(name: str) -> int:
  token = read_token(name) + 1
  write_token(name, token)
  return token


def write_text(name: str, text: str) -> None:
  _path(name).write_text(text, encoding="utf-8")


def read_text(name: str) -> str:
  path = _path(name)
  if not path.exists():
    return ""
  return path.read_text(encoding="utf-8")
