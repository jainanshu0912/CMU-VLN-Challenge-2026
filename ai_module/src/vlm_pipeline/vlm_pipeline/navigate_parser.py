"""Parse instruction-following questions into ordered navigation legs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

BETWEEN_KIND = "between"
OBJECT_KIND = "object"
NEAR_PATH_KIND = "near_path"

AVOID_RE = re.compile(
  r"(?:,\s*)?(?:avoiding|and avoid|,?\s*avoid)\s+(?:the\s+)?(?:path\s+)?(.+)$",
  re.IGNORECASE,
)
FIRST_RE = re.compile(r"^first,?\s*", re.IGNORECASE)
ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)

# "take the path between A and B" / "go between A and B"
BETWEEN_RE = re.compile(
  r"^(?:take the path between|go between|take the path in between)\s+(.+)$",
  re.IGNORECASE,
)
# "take the path near X" / "take the path near X to Y"
NEAR_PATH_RE = re.compile(
  r"^(?:take the path near)\s+(.+)$",
  re.IGNORECASE,
)
OBJECT_PREFIX_RE = re.compile(
  r"^(?:go\s+near|go\s+to|stop\s+at|stop\s+by|pass\s+by|to)\s+",
  re.IGNORECASE,
)
TO_OBJECT_RE = re.compile(r"^(.*?)\s+to\s+(?:the\s+)?(.+)$", re.IGNORECASE)
TWO_CLASS_RE = re.compile(r"^(?:the\s+)?two\s+(.+)$", re.IGNORECASE)

SPLIT_RE = re.compile(
  r"\s*,\s*then,?\s+"
  r"|\s+then,?\s+"
  r"|\s*,\s*and\s+finally,?\s+"
  r"|\s+and\s+finally,?\s+"
  r"|\s*,\s*and\s+(?=stop|go|take|pass|to\b)"
  r"|\s+and\s+(?=stop|go|take|pass)"
  r"|\s*,\s*(?=stop|go|take|pass)",
  re.IGNORECASE,
)


@dataclass
class NavLeg:
  kind: str
  phrase: str = ""
  anchor1: str = ""
  anchor2: str = ""

  def to_dict(self) -> dict:
    data = {"kind": self.kind}
    if self.kind == BETWEEN_KIND:
      data["anchor1"] = self.anchor1
      data["anchor2"] = self.anchor2
    else:
      data["phrase"] = self.phrase
    return data


@dataclass
class ParsedNavigate:
  legs: List[NavLeg] = field(default_factory=list)
  avoid_phrases: List[str] = field(default_factory=list)
  source: str = "rule_based"

  def to_dict(self) -> dict:
    return {
      "question_type": "navigate",
      "legs": [leg.to_dict() for leg in self.legs],
      "avoid_phrases": list(self.avoid_phrases),
      "source": self.source,
    }


class NavigateParser:
  """Rule-based parser for Type-3 instruction-following questions."""

  def parse(self, question: str) -> ParsedNavigate:
    text = question.strip().rstrip(".")
    if not text:
      raise ValueError("Question is empty")

    avoid_phrases: List[str] = []
    avoid_match = AVOID_RE.search(text)
    if avoid_match:
      avoid_phrases.append(_clean_phrase(avoid_match.group(1)))
      text = text[: avoid_match.start()].strip(" ,")

    text = FIRST_RE.sub("", text).strip()
    segments = _split_segments(text)
    legs: List[NavLeg] = []
    for segment in segments:
      legs.extend(_parse_segment(segment))

    if not legs:
      # Fallback: treat entire instruction as one object destination.
      cleaned = OBJECT_PREFIX_RE.sub("", text).strip()
      if cleaned:
        legs.append(NavLeg(kind=OBJECT_KIND, phrase=_clean_phrase(cleaned)))

    return ParsedNavigate(legs=legs, avoid_phrases=avoid_phrases)


def _split_segments(text: str) -> List[str]:
  protected = _protect_between_and(text)
  parts = SPLIT_RE.split(protected)
  return [_restore_and(part.strip(" ,")) for part in parts if part and part.strip(" ,")]


def _protect_between_and(text: str) -> str:
  """Keep 'between A and B' from being split on 'and'."""

  def _replace(match: re.Match[str]) -> str:
    return match.group(0).replace(" and ", " <<AND>> ", 1)

  return re.sub(
    r"(?:take the path between|go between|path between)\s+.+?\s+and\s+.+?(?=\s*(?:,|then|and\s+(?:go|stop|take|finally)|$|\s+to\s+))",
    _replace,
    text,
    flags=re.IGNORECASE,
  )


def _restore_and(text: str) -> str:
  return text.replace(" <<AND>> ", " and ")


def _parse_segment(segment: str) -> List[NavLeg]:
  text = segment.strip(" ,.")
  if not text:
    return []

  between_match = BETWEEN_RE.match(text)
  if between_match:
    rest = between_match.group(1).strip()
    # "A and B to C" / "A and B, ..."
    to_match = TO_OBJECT_RE.match(rest)
    if to_match:
      between_part, dest = to_match.groups()
      leg = _between_leg(between_part)
      legs = [leg] if leg else []
      if dest.strip():
        legs.append(NavLeg(kind=OBJECT_KIND, phrase=_clean_phrase(dest)))
      return legs
    leg = _between_leg(rest)
    return [leg] if leg else []

  near_path_match = NEAR_PATH_RE.match(text)
  if near_path_match:
    rest = near_path_match.group(1).strip()
    to_match = TO_OBJECT_RE.match(rest)
    if to_match:
      near_part, dest = to_match.groups()
      return [
        NavLeg(kind=NEAR_PATH_KIND, phrase=_clean_phrase(near_part)),
        NavLeg(kind=OBJECT_KIND, phrase=_clean_phrase(dest)),
      ]
    return [NavLeg(kind=NEAR_PATH_KIND, phrase=_clean_phrase(rest))]

  object_text = OBJECT_PREFIX_RE.sub("", text).strip()
  if not object_text:
    return []
  return [NavLeg(kind=OBJECT_KIND, phrase=_clean_phrase(object_text))]


def _between_leg(text: str) -> Optional[NavLeg]:
  text = text.strip()
  two_match = TWO_CLASS_RE.match(text)
  if two_match:
    cls = _clean_phrase(two_match.group(1))
    # Singularize lightly for matching ("columns" -> keep as-is; matcher synonyms handle).
    return NavLeg(kind=BETWEEN_KIND, anchor1=cls, anchor2=cls)

  parts = re.split(r"\s+and\s+", text, maxsplit=1, flags=re.IGNORECASE)
  if len(parts) != 2:
    return None
  return NavLeg(
    kind=BETWEEN_KIND,
    anchor1=_clean_phrase(parts[0]),
    anchor2=_clean_phrase(parts[1]),
  )


def _clean_phrase(phrase: str) -> str:
  text = phrase.strip().rstrip(".")
  text = ARTICLE_RE.sub("", text).strip()
  return text
