#!/usr/bin/env python3
"""Generate scene caption prompts from VLA-3D Unity scene graphs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/anshu/vla3d_data/Unity")
STRUCT = {
  "wall",
  "walls",
  "floor",
  "ceiling",
  "bathroom walls",
  "door frame",
  "window frame",
  "exterior walls",
  "partition wall",
  "glass wall",
  "air vent",
  "handle",
  "gate",
}
SYNONYMS = {
  "focus light": ["lamp", "focus light"],
  "ceiling light": ["lamp", "ceiling lamp"],
  "ceiling lamp": ["lamp", "ceiling lamp"],
  "wall lamp": ["lamp", "wall lamp"],
  "desk light": ["lamp"],
  "bedroom light": ["lamp"],
  "spot light": ["lamp", "spotlight"],
  "computer monitor": ["monitor", "computer monitor"],
  "computer mouse": ["mouse", "computer mouse"],
  "coffee cup": ["cup", "coffee cup"],
  "paper cup": ["cup"],
  "trash bin": ["trash can", "trash bin"],
  "curtains": ["curtain", "curtains"],
  "sofa pillows": ["pillow"],
  "pillows": ["pillow"],
  "couch": ["sofa", "couch"],
  "sofa cushion": ["pillow", "cushion"],
  "night stand": ["nightstand", "night stand", "bedside table"],
  "nightstand": ["nightstand", "night stand"],
  "bedside table": ["nightstand", "bedside table"],
  "dining chair": ["chair"],
  "deck chair": ["chair"],
  "potted plant": ["plant", "potted plant"],
  "potted cactus": ["plant"],
  "potted branch": ["plant"],
  "potted bamboo": ["plant"],
  "painting": ["picture", "painting"],
  "photo": ["picture", "photo"],
  "drawing": ["picture"],
  "books": ["book"],
  "drawer": ["drawers", "drawer"],
  "tv remote": ["remote"],
  "windows": ["window"],
  "wine bottle": ["bottle", "wine bottle"],
  "beer bottle": ["bottle"],
  "soap bottle": ["soap", "soap bottle"],
  "soap dish": ["soap"],
  "towel rail": ["towel rack", "towel rail"],
  "entrance door": ["door"],
  "balcony door": ["door"],
  "wardrobe door": ["door", "wardrobe"],
  "round table": ["table"],
  "tv cabinet": ["cabinet"],
  "desk cabinet": ["cabinet"],
  "sink cabinet": ["cabinet"],
  "bookcase": ["bookshelf", "shelf"],
  "calligraphy painting": ["picture", "painting"],
}
GROUPS = {
  "office": ["office_1", "office_2"],
  "hotel": ["hotel_room_1", "hotel_room_2"],
  "livingroom": ["livingroom_1", "livingroom_2", "livingroom_3", "livingroom_4"],
  "home": ["home_building_1", "home_building_2"],
  "cultural": ["arabic_room", "chinese_room", "japanese_room"],
}


def labels_for(scene_dir: Path) -> Counter:
  cands = list(scene_dir.glob("*_scene_graph.json"))
  if not cands:
    return Counter()
  data = json.loads(cands[0].read_text(encoding="utf-8"))
  labels: list[str] = []
  for region in data.get("regions", {}).values():
    for obj in region.get("objects", []):
      lab = (obj.get("raw_label") or obj.get("nyu_label") or "").strip().lower()
      lab = " ".join(lab.replace("_", " ").split())
      if lab and lab not in STRUCT:
        labels.append(lab)
  return Counter(labels)


def prompt_from_counter(counter: Counter, max_terms: int = 42) -> str:
  scored: dict[str, int] = {}
  for lab, cnt in counter.most_common():
    for token in SYNONYMS.get(lab, [lab]):
      scored[token] = max(scored.get(token, 0), cnt)
  ordered = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
  terms: list[str] = []
  for token, _ in ordered:
    if token not in terms:
      terms.append(token)
    if len(terms) >= max_terms:
      break
  return " . ".join(terms)


def fmt_prompt(prompt: str) -> str:
  parts = prompt.split(" . ")
  lines: list[str] = []
  cur: list[str] = []
  cur_len = 0
  for part in parts:
    add = (3 if cur else 0) + len(part)
    if cur and cur_len + add > 70:
      lines.append(" . ".join(cur))
      cur = [part]
      cur_len = len(part)
    else:
      cur.append(part)
      cur_len += add
  if cur:
    lines.append(" . ".join(cur))
  if len(lines) == 1:
    return f'  "{lines[0]}"'
  out = "  (\n"
  for index, line in enumerate(lines):
    sep = " . " if index < len(lines) - 1 else ""
    out += f'    "{line}{sep}"\n'
  out += "  )"
  return out


def main() -> None:
  scenes = sorted(
    p.name for p in ROOT.iterdir() if p.is_dir() and p.name != "live_scene"
  )
  print("SCENE_PROMPTS = {")
  for name in scenes:
    print(f'  "{name}":')
    print(fmt_prompt(prompt_from_counter(labels_for(ROOT / name))) + ",")
  print("}")
  print()
  print("TYPE_PROMPTS = {")
  for group, members in GROUPS.items():
    agg: Counter = Counter()
    for member in members:
      agg += labels_for(ROOT / member)
    print(f'  "{group}":')
    print(fmt_prompt(prompt_from_counter(agg, max_terms=48)) + ",")
  print("}")


if __name__ == "__main__":
  main()
