"""Stage 1 — LLM object filter.

SORT3D first shrinks the scene inventory so the reasoner only sees objects
relevant to the question. That keeps the prompt short and reduces distractors.

Flow:
  1. Extract nouns / referring phrases from the question (LLM or lightweight NLP).
  2. Keep inventory objects whose labels/aliases fuzzy-match those nouns.
  3. Pass the filtered inventory to the toolbox reasoner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from vlm_pipeline_sort3d.scene_inventory import InventoryObject, SceneInventory, inventory_prompt_block

# Optional LLM callable: (system_prompt, user_prompt) -> response text
LlmCallable = Callable[[str, str], str]


FILTER_SYSTEM_PROMPT = """\
You extract object nouns from a robot spatial-query question.
Return ONLY valid JSON of the form:
{"nouns": ["pillow", "book", "stool"]}
Include target objects and landmark/anchor objects. Do not include verbs or relations.
"""


@dataclass
class FilterResult:
  nouns: List[str]
  filtered: SceneInventory
  raw_llm_response: str = ""


def _norm(text: str) -> str:
  return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _rule_based_nouns(question: str) -> List[str]:
  """Cheap fallback when no LLM is available."""
  stop = {
    "the", "a", "an", "find", "how", "many", "count", "that", "which", "is",
    "are", "of", "to", "and", "or", "on", "in", "at", "near", "closest",
    "farthest", "between", "above", "below", "left", "right", "with", "from",
  }
  tokens = re.findall(r"[a-zA-Z]+", question.lower())
  # Keep content words; merge simple bigrams later via fuzzy match on labels.
  return [t for t in tokens if t not in stop and len(t) > 2]


def _parse_nouns_json(text: str) -> List[str]:
  text = text.strip()
  # Tolerate fenced JSON.
  if "```" in text:
    parts = text.split("```")
    for part in parts:
      part = part.strip()
      if part.startswith("json"):
        part = part[4:].strip()
      if part.startswith("{"):
        text = part
        break
  try:
    data = json.loads(text)
  except json.JSONDecodeError:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
      return []
    try:
      data = json.loads(match.group(0))
    except json.JSONDecodeError:
      return []
  nouns = data.get("nouns") if isinstance(data, dict) else None
  if not isinstance(nouns, list):
    return []
  return [str(n).strip() for n in nouns if str(n).strip()]


def _object_matches_nouns(obj: InventoryObject, nouns: Sequence[str]) -> bool:
  labels = [_norm(obj.name)] + [_norm(a) for a in obj.aliases]
  labels = [l for l in labels if l]
  for noun in nouns:
    needle = _norm(noun)
    if not needle:
      continue
    for hay in labels:
      if needle == hay or needle in hay or hay in needle:
        return True
  return False


class ObjectFilter:
  """Stage 1 filter: question → relevant subset of the scene inventory."""

  def __init__(self, llm: Optional[LlmCallable] = None) -> None:
    self.llm = llm

  def extract_nouns(self, question: str) -> tuple[List[str], str]:
    if self.llm is None:
      return _rule_based_nouns(question), ""
    user = f"Question: {question}\nObjects available (for context only):\n(none — extract nouns from the question alone)"
    raw = self.llm(FILTER_SYSTEM_PROMPT, user)
    nouns = _parse_nouns_json(raw)
    if not nouns:
      nouns = _rule_based_nouns(question)
    return nouns, raw

  def filter(
    self,
    question: str,
    inventory: SceneInventory,
    *,
    keep_all_if_empty: bool = True,
  ) -> FilterResult:
    nouns, raw = self.extract_nouns(question)
    kept = [obj for obj in inventory.objects if _object_matches_nouns(obj, nouns)]
    if not kept and keep_all_if_empty:
      kept = list(inventory.objects)
    filtered = SceneInventory(scene_name=inventory.scene_name, objects=kept)
    return FilterResult(nouns=nouns, filtered=filtered, raw_llm_response=raw)

  def build_llm_filter_prompt(self, question: str, inventory: SceneInventory) -> str:
    """Optional richer prompt that also shows inventory names (for debugging)."""
    return (
      f"Question: {question}\n\n"
      f"Scene objects:\n{inventory_prompt_block(inventory)}\n\n"
      'Return JSON: {"nouns": [...]}'
    )
