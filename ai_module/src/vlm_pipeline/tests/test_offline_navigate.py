"""Offline test: instruction-following parse + waypoint resolve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT))

from vlm_pipeline.navigate_parser import BETWEEN_KIND, NavigateParser
from vlm_pipeline.navigate_pipeline import NavigatePipeline
from vlm_pipeline.question_classifier import QuestionClassifier, QuestionType
from vlm_pipeline.scene_loader import SceneLoader

QUESTIONS_PATH = Path(__file__).resolve().parents[4] / "questions" / "questions.json"


def run_navigate_pipeline_test() -> int:
  if not QUESTIONS_PATH.is_file():
    print(f"ERROR: missing {QUESTIONS_PATH}")
    return 1

  loader = SceneLoader()
  classifier = QuestionClassifier()
  parser = NavigateParser()
  pipeline = NavigatePipeline(standoff_m=0.8)

  with QUESTIONS_PATH.open(encoding="utf-8") as handle:
    scenes = json.load(handle)

  total = 0
  classified = 0
  parsed_ok = 0
  resolved_ok = 0
  failures: list[str] = []

  for entry in scenes:
    scene_name = entry["scene"]
    try:
      scene = loader.load(scene_name)
    except FileNotFoundError as exc:
      failures.append(f"{scene_name}: missing data ({exc})")
      continue

    for question in entry["questions"]["instruction_following"]:
      total += 1
      qtype = classifier.classify(question)
      if qtype != QuestionType.NAVIGATE:
        failures.append(f"{scene_name}: classified as {qtype.value}: {question}")
        continue
      classified += 1

      try:
        parsed = parser.parse(question)
      except Exception as exc:
        failures.append(f"{scene_name}: parse error ({exc}): {question}")
        continue

      if not parsed.legs:
        failures.append(f"{scene_name}: zero legs: {question}")
        continue
      parsed_ok += 1

      waypoints = pipeline.resolve(scene, parsed, robot_x=0.0, robot_y=0.0)
      if not waypoints:
        failures.append(
          f"{scene_name}: unresolved legs {parsed.to_dict()}: {question}"
        )
        continue
      if len(waypoints) < len(parsed.legs):
        failures.append(
          f"{scene_name}: resolved {len(waypoints)}/{len(parsed.legs)} "
          f"legs {parsed.to_dict()}: {question}"
        )
        continue
      resolved_ok += 1
      print(
        f"OK  [{scene_name}] legs={len(parsed.legs)} wps={len(waypoints)} | {question}"
      )

  print()
  print(f"Total instruction questions: {total}")
  print(f"Classified NAVIGATE:          {classified}")
  print(f"Parsed with >=1 leg:          {parsed_ok}")
  print(f"Resolved >=1 waypoint:        {resolved_ok}")
  if failures:
    print(f"\nFailures ({len(failures)}):")
    for line in failures:
      print(f"  - {line}")
    return 1

  # Spot-check a between leg exists for a known question.
  sample = parser.parse(
    "First, go to the potted plant furthest from the hookah, "
    "then take the path between the two columns, and stop at the tray on the table."
  )
  kinds = [leg.kind for leg in sample.legs]
  if BETWEEN_KIND not in kinds:
    print(f"ERROR: expected a between leg, got {sample.to_dict()}")
    return 1

  print("\nAll navigate offline checks passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(run_navigate_pipeline_test())
