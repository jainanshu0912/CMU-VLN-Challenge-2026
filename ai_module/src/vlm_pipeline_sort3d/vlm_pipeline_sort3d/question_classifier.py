"""Question type routing for Pipeline C (independent copy)."""

from __future__ import annotations

import re
from enum import Enum


class QuestionType(str, Enum):
  FIND = "find"
  COUNT = "count"
  NAVIGATE = "navigate"


_COUNT_RE = re.compile(r"^\s*(how\s+many|count)\b", re.IGNORECASE)
_FIND_RE = re.compile(r"^\s*(find\b|the\s+\w+)", re.IGNORECASE)


def classify_question(question: str) -> QuestionType:
  q = question.strip()
  if _COUNT_RE.search(q):
    return QuestionType.COUNT
  if _FIND_RE.search(q):
    return QuestionType.FIND
  # Article-first object references without "Find".
  if re.match(r"^\s*(a|an|the)\s+\w+", q, flags=re.IGNORECASE):
    return QuestionType.FIND
  return QuestionType.NAVIGATE
