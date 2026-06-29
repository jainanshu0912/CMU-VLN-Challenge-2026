"""Keyword-based routing for challenge question types."""

from __future__ import annotations

from enum import Enum


class QuestionType(Enum):
  FIND = "find"
  COUNT = "count"
  NAVIGATE = "navigate"


class QuestionClassifier:
  """Route questions to find, count, or navigate pipelines."""

  def classify(self, question: str) -> QuestionType:
    text = question.strip()
    if not text:
      raise ValueError("Question is empty")

    lower = text.lower()

    if lower.startswith("how many") or lower.startswith("count"):
      return QuestionType.COUNT

    if lower.startswith("find") or lower.startswith("the "):
      return QuestionType.FIND

    return QuestionType.NAVIGATE
