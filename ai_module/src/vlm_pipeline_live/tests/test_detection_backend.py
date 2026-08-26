"""Unit tests for pluggable open-vocab detection backends."""

from __future__ import annotations

import pytest

from vlm_pipeline_live.detection_backend import (
  create_detection_backend,
  prompt_to_class_list,
)


def test_prompt_to_class_list_splits_and_dedupes():
  classes = prompt_to_class_list("Chair . desk .  chair . lamp")
  assert classes == ["chair", "desk", "lamp"]


def test_prompt_to_class_list_empty():
  assert prompt_to_class_list("") == []
  assert prompt_to_class_list("  .  . ") == []


def test_factory_resolves_aliases():
  # Do not load weights — only construct the wrapper object.
  yolo = create_detection_backend("yolo_world", yolo_model="yolov8s-worldv2.pt")
  assert yolo.name == "yolo_world"

  yoloe = create_detection_backend("yolov11", yolo_model="yoloe-11s-seg.pt")
  assert yoloe.name == "yoloe"

  dino = create_detection_backend(
    "dino",
    model_config_path="/tmp/cfg.py",
    model_checkpoint_path="/tmp/ckpt.pth",
  )
  assert dino.name == "grounding_dino"

  owl = create_detection_backend("owlv2", yolo_model="google/owlv2-base-patch16")
  assert owl.name == "owlvit"


def test_factory_gemini_not_a_detector():
  with pytest.raises(NotImplementedError, match="verifier"):
    create_detection_backend("gemini")


def test_factory_unknown():
  with pytest.raises(ValueError):
    create_detection_backend("not_a_backend")
