"""Offline test: find pipeline (classify -> parse -> graph search)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Allow running without colcon install: python tests/test_offline_find.py
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT))

from vlm_pipeline.graph_search import GraphSearchMatcher
from vlm_pipeline.query_parser import QueryParser
from vlm_pipeline.question_classifier import QuestionClassifier, QuestionType
from vlm_pipeline.scene_loader import SceneLoader

QUESTIONS_PATH = Path(__file__).resolve().parents[4] / "questions" / "questions.json"


def _expected_target_word(question: str) -> str:
  text = question.strip()
  text = re.sub(r"^(find the|the)\s+", "", text, flags=re.IGNORECASE)
  first = text.split()[0].lower()
  if first in {"red", "blue", "black", "white", "small", "large"}:
    return text.split()[1].lower() if len(text.split()) > 1 else first
  return first


def run_find_pipeline_test() -> int:
  if not QUESTIONS_PATH.is_file():
    print(f"ERROR: missing {QUESTIONS_PATH}")
    return 1

  loader = SceneLoader()
  classifier = QuestionClassifier()
  parser = QueryParser(use_llm=False)
  matcher = GraphSearchMatcher()

  with QUESTIONS_PATH.open(encoding="utf-8") as handle:
    scenes = json.load(handle)

  total = 0
  classified_find = 0
  parsed_ok = 0
  matched = 0
  label_ok = 0
  failures: list[str] = []

  print("=" * 72)
  print("FIND PIPELINE OFFLINE TEST")
  print("Stages: QuestionClassifier -> QueryParser -> GraphSearchMatcher.find")
  print("=" * 72)

  for entry in scenes:
    scene_name = entry["scene"]
    scene = loader.load(scene_name)

    for question in entry["questions"]["object_reference"]:
      total += 1
      qtype = classifier.classify(question)
      if qtype != QuestionType.FIND:
        failures.append(f"[{scene_name}] misclassified as {qtype.value}: {question}")
        continue
      classified_find += 1

      parsed = parser.parse(question, QuestionType.FIND)
      if not parsed.target_class:
        failures.append(f"[{scene_name}] empty target_class: {question}")
        continue
      parsed_ok += 1

      result = matcher.find(scene, parsed)
      if result is None:
        failures.append(f"[{scene_name}] NO MATCH: {question}")
        print(f"FAIL {scene_name}: {question}")
        print(f"       parsed: {parsed.to_dict()}")
        continue

      matched += 1
      expected = _expected_target_word(question)
      label_hit = expected in result.raw_label.lower()
      if label_hit:
        label_ok += 1
      else:
        failures.append(
          f"[{scene_name}] label mismatch: expected '{expected}' in '{result.raw_label}' "
          f"(id={result.object_id}) for: {question}"
        )

      status = "OK" if label_hit else "LABEL?"
      print(
        f"{status:6} [{scene_name}] id={result.object_id:>3} "
        f"{result.raw_label!r} @ ({result.cx:.2f}, {result.cy:.2f}) | {question[:55]}"
      )

  print("=" * 72)
  print(f"Questions:              {total}")
  print(f"Classified as find:     {classified_find}/{total}")
  print(f"Parsed (target_class):  {parsed_ok}/{total}")
  print(f"Graph search matched:   {matched}/{total}")
  print(f"Label sanity check:     {label_ok}/{matched} (of matched)")
  print("=" * 72)

  if failures:
    print("\nIssues ({}):".format(len(failures)))
    for line in failures:
      print(f"  - {line}")

  # Pipeline "complete" for offline: all classified, all parsed, high match rate
  offline_ok = classified_find == total and parsed_ok == total and matched >= 25
  ros_note = (
    "ROS publish wired in main_node.py "
    "(/selected_object_marker, /way_point_with_heading, /numerical_response)."
  )
  print(f"\nOffline find pipeline: {'PASS (usable)' if offline_ok else 'INCOMPLETE'}")
  print(ros_note)

  return 0 if matched == total else 1


if __name__ == "__main__":
  raise SystemExit(run_find_pipeline_test())
