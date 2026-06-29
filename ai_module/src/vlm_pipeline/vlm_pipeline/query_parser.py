"""Extract structured find/count queries from natural-language questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vlm_pipeline.question_classifier import QuestionType
from vlm_pipeline.vlm_backends.base import VlmBackend, VlmBackendError

COLOR_WORDS = frozenset({
  "red", "black", "blue", "white", "green", "brown", "gray", "grey", "yellow",
})
SIZE_WORDS = frozenset({
  "small", "big", "large", "tiny", "huge",
})

RELATION_PATTERNS: List[tuple[str, str]] = [
  (r"\s+closest to the\s+", "closest"),
  (r"\s+closest to a\s+", "closest"),
  (r"\s+farthest from the\s+", "farthest"),
  (r"\s+furthest from the\s+", "farthest"),
  (r"\s+between\s+", "between"),
  (r"\s+above the\s+", "above"),
  (r"\s+below the\s+", "below"),
  (r"\s+below a\s+", "below"),
  (r"\s+near the\s+", "near"),
  (r"\s+on the\s+", "on"),
  (r"\s+on a\s+", "on"),
  (r"\s+under the\s+", "below"),
  (r"\s+under a\s+", "below"),
  (r"\s+with\s+", "with"),
]

FIND_PREFIX_RE = re.compile(r"^(find the|the)\s+", re.IGNORECASE)
COUNT_PREFIX_RE = re.compile(r"^(how many|count the number of)\s+", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
THAT_IS_RE = re.compile(r"\s+that is$", re.IGNORECASE)
THAT_HAS_RE = re.compile(r"\s+that has .+$", re.IGNORECASE)
ON_THEM_RE = re.compile(r"\s+on them\.?$", re.IGNORECASE)

LLM_SYSTEM_PROMPT = (
  "You extract structured JSON from indoor navigation questions. "
  "Respond with ONLY valid JSON, no markdown fences or commentary."
)

LLM_USER_TEMPLATE = """Question: {question}
Task type: {question_type}

If task type is find, output:
{{"question_type":"find","target_class":"","anchors":[{{"class":"","role":"anchor1"}}],"relation":"","attributes":{{"color":null,"size":null}}}}

If task type is count, output:
{{"question_type":"count","target_class":"","attribute_filters":{{"color":null,"size":null}},"spatial_filter":{{"relation":null,"anchor":null}}}}

Rules:
- Use short object class names (e.g. pillow, sofa, TV cabinet).
- relation is one of: above, below, closest, farthest, between, beside, near, in, on, hanging_on.
- For between, use two anchors with roles anchor1 and anchor2.
- Put color/size adjectives in attributes, not in target_class when possible.
"""


@dataclass
class Anchor:
  class_name: str
  role: str = "anchor"

  def to_dict(self) -> Dict[str, str]:
    return {"class": self.class_name, "role": self.role}


@dataclass
class AttributeFilters:
  color: Optional[str] = None
  size: Optional[str] = None

  def to_dict(self) -> Dict[str, Optional[str]]:
    return {"color": self.color, "size": self.size}


@dataclass
class SpatialFilter:
  relation: Optional[str] = None
  anchor: Optional[str] = None

  def to_dict(self) -> Dict[str, Optional[str]]:
    return {"relation": self.relation, "anchor": self.anchor}


@dataclass
class ParsedQuery:
  question_type: str
  target_class: str
  anchors: List[Anchor] = field(default_factory=list)
  relation: Optional[str] = None
  attributes: AttributeFilters = field(default_factory=AttributeFilters)
  attribute_filters: AttributeFilters = field(default_factory=AttributeFilters)
  spatial_filter: SpatialFilter = field(default_factory=SpatialFilter)
  source: str = "rule_based"

  def to_dict(self) -> Dict[str, Any]:
    result: Dict[str, Any] = {
      "question_type": self.question_type,
      "target_class": self.target_class,
      "source": self.source,
    }
    if self.question_type == "find":
      result["anchors"] = [anchor.to_dict() for anchor in self.anchors]
      result["relation"] = self.relation
      result["attributes"] = self.attributes.to_dict()
    else:
      result["attribute_filters"] = self.attribute_filters.to_dict()
      result["spatial_filter"] = self.spatial_filter.to_dict()
    return result


class QueryParser:
  """Rule-based parser by default; optional LLM backend when configured."""

  def __init__(
    self,
    backend: Optional[VlmBackend] = None,
    use_llm: bool = False,
  ) -> None:
    self.backend = backend
    self.use_llm = use_llm

  def parse(self, question: str, question_type: QuestionType) -> ParsedQuery:
    if question_type == QuestionType.NAVIGATE:
      raise ValueError("Navigate questions are not supported by QueryParser")

    if self.use_llm and self.backend is not None and self.backend.is_available():
      try:
        return self._parse_with_llm(question, question_type)
      except (VlmBackendError, json.JSONDecodeError, ValueError) as exc:
        return self._parse_rule_based(question, question_type, fallback_reason=str(exc))

    return self._parse_rule_based(question, question_type)

  def _parse_with_llm(self, question: str, question_type: QuestionType) -> ParsedQuery:
    prompt = LLM_USER_TEMPLATE.format(
      question=question,
      question_type=question_type.value,
    )
    response = self.backend.complete(
      prompt=prompt,
      system=LLM_SYSTEM_PROMPT,
      temperature=0.0,
      max_tokens=512,
    )
    data = _extract_json(response.text)
    return _parsed_query_from_dict(data, source=self.backend.provider)

  def _parse_rule_based(
    self,
    question: str,
    question_type: QuestionType,
    fallback_reason: Optional[str] = None,
  ) -> ParsedQuery:
    if question_type == QuestionType.COUNT:
      parsed = _parse_count_rule_based(question)
    else:
      parsed = _parse_find_rule_based(question)

    if fallback_reason:
      parsed.source = f"rule_based (llm fallback: {fallback_reason})"
    return parsed


def _parse_find_rule_based(question: str) -> ParsedQuery:
  text = question.strip()
  text = FIND_PREFIX_RE.sub("", text)
  text = THAT_HAS_RE.sub("", text)
  text = THAT_IS_RE.sub("", text).strip()

  target_part, relation, anchor_part = _split_by_relation(text)
  color, size, target_class = _split_attributes(target_part)

  anchors: List[Anchor] = []
  if relation == "between" and anchor_part:
    parts = re.split(r"\s+and\s+", anchor_part, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
      anchors = [
        Anchor(_clean_phrase(parts[0]), "anchor1"),
        Anchor(_clean_phrase(parts[1]), "anchor2"),
      ]
    else:
      anchors = [Anchor(_clean_phrase(anchor_part), "anchor1")]
  elif anchor_part:
    anchors = [Anchor(_clean_phrase(anchor_part), "anchor1")]

  return ParsedQuery(
    question_type="find",
    target_class=target_class,
    anchors=anchors,
    relation=relation,
    attributes=AttributeFilters(color=color, size=size),
    source="rule_based",
  )


def _parse_count_rule_based(question: str) -> ParsedQuery:
  text = question.strip()
  text = COUNT_PREFIX_RE.sub("", text).strip()
  text = ON_THEM_RE.sub("", text).strip()

  spatial_filter = SpatialFilter()
  attribute_filters = AttributeFilters()
  target_part = text

  if re.search(r"\s+are\s+", text, flags=re.IGNORECASE):
    target_part, spatial_text = re.split(r"\s+are\s+", text, maxsplit=1, flags=re.IGNORECASE)
    _, relation, anchor_part = _split_by_relation(spatial_text)
    spatial_filter = SpatialFilter(
      relation=relation,
      anchor=_clean_phrase(anchor_part) if anchor_part else None,
    )
  elif re.search(r"\s+with\s+", text, flags=re.IGNORECASE):
    target_part, with_part = re.split(r"\s+with\s+", text, maxsplit=1, flags=re.IGNORECASE)
    _, relation, anchor_part = _split_by_relation(with_part)
    spatial_filter = SpatialFilter(
      relation=relation or "with",
      anchor=_clean_phrase(anchor_part) if anchor_part else _clean_phrase(with_part),
    )

  color, size, target_class = _split_attributes(target_part)
  attribute_filters = AttributeFilters(color=color, size=size)

  return ParsedQuery(
    question_type="count",
    target_class=target_class,
    attribute_filters=attribute_filters,
    spatial_filter=spatial_filter,
    source="rule_based",
  )


def _split_by_relation(text: str) -> tuple[str, Optional[str], str]:
  padded = f" {text.strip()}"
  best_match: Optional[re.Match[str]] = None
  best_relation: Optional[str] = None

  for pattern, relation in RELATION_PATTERNS:
    match = re.search(pattern, padded, flags=re.IGNORECASE)
    if match and (best_match is None or match.start() < best_match.start()):
      best_match = match
      best_relation = relation

  if best_match is None or best_relation is None:
    return text.strip(), None, ""

  return (
    padded[:best_match.start()].strip(),
    best_relation,
    padded[best_match.end():].strip(),
  )


def _split_attributes(phrase: str) -> tuple[Optional[str], Optional[str], str]:
  words = phrase.strip().split()
  color: Optional[str] = None
  size: Optional[str] = None
  remaining: List[str] = []

  for word in words:
    lower = word.lower()
    if color is None and lower in COLOR_WORDS:
      color = lower
    elif size is None and lower in SIZE_WORDS:
      size = lower
    else:
      remaining.append(word)

  class_name = _clean_phrase(" ".join(remaining))
  if not class_name and phrase.strip():
    class_name = _clean_phrase(phrase)

  return color, size, class_name


PLURAL_LABEL_SUFFIXES = frozenset({
  "pictures", "photos", "records", "flowers", "paintings", "glasses",
})


def _singularize(label: str) -> str:
  if not label:
    return label
  words = label.split()
  if not words:
    return label
  last = words[-1].lower()
  if last in PLURAL_LABEL_SUFFIXES:
    return label
  if last.endswith("ies") and len(last) > 3:
    words[-1] = words[-1][: -3] + "y"
  elif last.endswith("s") and not last.endswith("ss") and last not in {"gas", "bus"}:
    words[-1] = words[-1][: -1]
  return " ".join(words)


def _clean_phrase(phrase: str) -> str:
  cleaned = phrase.strip().rstrip(".?")
  cleaned = ARTICLE_RE.sub("", cleaned)
  cleaned = THAT_IS_RE.sub("", cleaned)
  cleaned = re.sub(r"\s+", " ", cleaned)
  cleaned = cleaned.strip().rstrip(".?")
  return _singularize(cleaned)


def _extract_json(text: str) -> Dict[str, Any]:
  stripped = text.strip()
  if stripped.startswith("```"):
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

  start = stripped.find("{")
  end = stripped.rfind("}")
  if start == -1 or end == -1:
    raise ValueError("No JSON object found in LLM response")

  return json.loads(stripped[start:end + 1])


def _parsed_query_from_dict(data: Dict[str, Any], source: str) -> ParsedQuery:
  question_type = data.get("question_type", "find")
  anchors = [
    Anchor(
      class_name=anchor.get("class", ""),
      role=anchor.get("role", "anchor1"),
    )
    for anchor in data.get("anchors", [])
  ]

  attributes_data = data.get("attributes", {})
  attribute_filters_data = data.get("attribute_filters", attributes_data)
  spatial_data = data.get("spatial_filter", {})

  return ParsedQuery(
    question_type=question_type,
    target_class=data.get("target_class", ""),
    anchors=anchors,
    relation=data.get("relation"),
    attributes=AttributeFilters(
      color=attributes_data.get("color"),
      size=attributes_data.get("size"),
    ),
    attribute_filters=AttributeFilters(
      color=attribute_filters_data.get("color"),
      size=attribute_filters_data.get("size"),
    ),
    spatial_filter=SpatialFilter(
      relation=spatial_data.get("relation"),
      anchor=spatial_data.get("anchor"),
    ),
    source=source,
  )
