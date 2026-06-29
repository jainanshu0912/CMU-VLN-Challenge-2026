"""Offline test: count pipeline (classify -> parse -> CountPipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
  sys.path.insert(0, str(_PKG_ROOT))

from vlm_pipeline.count_pipeline import CountPipeline
from vlm_pipeline.query_parser import QueryParser
from vlm_pipeline.question_classifier import QuestionClassifier, QuestionType
from vlm_pipeline.scene_loader import SceneLoader

QUESTIONS_PATH = Path(__file__).resolve().parents[4] / "questions" / "questions.json"

# Expected counts from VLA-3D scene graph + object CSV (no official challenge GT).
EXPECTED_COUNTS: dict[str, int] = {
  "arabic_room": 2,
  "chinese_room": 6,
  "home_building_1": 6,
  "home_building_2": 3,
  "hotel_room_1": 4,
  "hotel_room_2": 3,
  "japanese_room": 3,
  "livingroom_1": 8,
  "livingroom_2": 1,
  "livingroom_3": 2,
  "livingroom_4": 6,
  "loft": 7,
  "office_1": 6,
  "office_2": 2,
  "studio": 1,
}


def run_count_pipeline_test() -> int:
  if not QUESTIONS_PATH.is_file():
    print(f"ERROR: missing {QUESTIONS_PATH}")
    return 1

  loader = SceneLoader()
  classifier = QuestionClassifier()
  parser = QueryParser(use_llm=False)
  pipeline = CountPipeline()

  with QUESTIONS_PATH.open(encoding="utf-8") as handle:
    scenes = json.load(handle)

  total = 0
  classified_count = 0
  parsed_ok = 0
  matched = 0
  failures: list[str] = []

  print("=" * 72)
  print("COUNT PIPELINE OFFLINE TEST")
  print("Stages: QuestionClassifier -> QueryParser -> CountPipeline")
  print("=" * 72)

  for entry in scenes:
    scene_name = entry["scene"]
    scene = loader.load(scene_name)
    question = entry["questions"]["numerical"][0]
    expected = EXPECTED_COUNTS.get(scene_name)

    total += 1
    qtype = classifier.classify(question)
    if qtype != QuestionType.COUNT:
      failures.append(f"[{scene_name}] misclassified as {qtype.value}: {question}")
      continue
    classified_count += 1

    parsed = parser.parse(question, QuestionType.COUNT)
    if not parsed.target_class:
      failures.append(f"[{scene_name}] empty target_class: {question}")
      continue
    parsed_ok += 1

    count = pipeline.count(scene, parsed)
    objects = pipeline.filter_objects(scene, parsed)
    ok = expected is not None and count == expected
    if ok:
      matched += 1
    else:
      failures.append(
        f"[{scene_name}] count={count} expected={expected}: {question}"
      )

    status = "OK" if ok else "FAIL"
    ids = ",".join(obj.object_id for obj in objects)
    print(
      f"{status:6} [{scene_name}] count={count:2} "
      f"(expected {expected}) ids={ids} | {question[:50]}"
    )

  print("=" * 72)
  print(f"Questions:              {total}")
  print(f"Classified as count:    {classified_count}/{total}")
  print(f"Parsed (target_class):  {parsed_ok}/{total}")
  print(f"Matched expected:       {matched}/{total}")
  print("=" * 72)

  if failures:
    print(f"\nIssues ({len(failures)}):")
    for line in failures:
      print(f"  - {line}")

  offline_ok = classified_count == total and parsed_ok == total and matched == total
  print(f"\nOffline count pipeline: {'PASS' if offline_ok else 'INCOMPLETE'}")
  return 0 if offline_ok else 1


if __name__ == "__main__":
  raise SystemExit(run_count_pipeline_test())
