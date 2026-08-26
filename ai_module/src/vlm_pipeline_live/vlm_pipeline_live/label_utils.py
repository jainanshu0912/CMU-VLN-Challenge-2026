"""Label prompts and canonicalization for Pipeline B detections.

Scene captions are derived from VLA-3D Unity GT label frequencies under
``~/vla3d_data/Unity/<scene>/`` (structural wall/floor/ceiling omitted).

Use ``scene_type:=office_2`` / ``hotel_room_1`` / ``chinese_room`` / … for a
per-scene prompt, or coarse types: ``office`` | ``hotel`` | ``livingroom`` |
``home`` | ``cultural`` | ``indoor``.

Regenerate with::

  python3 scripts/generate_scene_prompts.py
"""

from __future__ import annotations

import re

SCENE_PROMPTS = {
  "arabic_room":
  (
    "focus light . lamp . pillow . plant . potted plant . door . wall lamp . "
    "window . glass . sofa . carpet . ceiling lamp . column . lantern . "
    "picture . stool . table . vase . arabic jar . book . coffee pot . "
    "hookah . hookah wire . shoes . tray"
  ),
  "chinese_room":
  (
    "focus light . lamp . pillow . chair . vase . painting . picture . "
    "plant . potted plant . book . stool . bowl . ceiling lamp . door . "
    "light switch . symbol decoraion . table . window . cabinet . carpet . "
    "circle decoration . elephant figurine . eye glasses . horse figurine . "
    "remote . side table . sofa . tablecloth . tea table . tower decoration . "
    "tv"
  ),
  "home_building_1":
  (
    "ceiling lamp . lamp . bottle . tree . pillow . chair . cup . spoon . "
    "drawer . drawers . spice jar . dish . bowl . curtain . plate . book . "
    "fork . plant . potted plant . speaker . box . carpet . bedside table . "
    "face cream . night stand . nightstand . photo . picture . sofa . "
    "air conditioner . cupboard . desk . door . dvd player . kitchen door . "
    "kitchen knife . lotion . trash can . tv . bed bench . circular light . "
    "clock"
  ),
  "home_building_2":
  (
    "ceiling lamp . lamp . bottle . wine bottle . pillow . plant . "
    "potted plant . window . door . chair . curtain . cushion . night stand . "
    "nightstand . speaker . wine glass . carpet . dish . bed . book . clock . "
    "dressing chair . dvd . kettle . magazine . photo . picture . quilt . "
    "ash tray . balcony railing . cabinet . coffee table . dining table . "
    "dressing table . kitchen cabinet . kitchen island . mattress . mirror . "
    "ottoman . painting . range hood . sculpture"
  ),
  "hotel_room_1":
  (
    "focus light . lamp . light switch . pillow . paper . towel . "
    "bedside table . chair . curtain . curtains . nightstand . picture . "
    "sink . soap . towel rack . vase . bed . bench . blanket . book . "
    "cabinet . ceiling lamp . eye glasses . magazine . mirror . "
    "paper holder . phone . shower . shower tap . suitcase . toilet . "
    "trash bin . trash can . tv . wall lamp . window"
  ),
  "hotel_room_2":
  (
    "focus light . lamp . pillow . picture . towel . bedside table . "
    "ceiling lamp . chair . curtain . light switch . map . mirror . "
    "night stand . nightstand . toilet paper . bag . bathtub . bed . "
    "bed frame . bench . blanket . cabinet . camera . carpet . eye glasses . "
    "fireplace . flowers . glass . paper holder . perfume bottle . phone . "
    "remote . shower tap . sink . soap . soap bottle . table . tap . toilet . "
    "towel rack . towel rail . trash bin"
  ),
  "japanese_room":
  (
    "door . column . glass . pillow . flowers . lantern . painting . "
    "picture . table . ceiling lamp . chopsticks . lamp . placemat . plant . "
    "potted plant . vase . wardrobe . dish . display ledge . fan decoration . "
    "jar . sake bottle . sauce bowl . tatami . teapot . "
    "zen stone decoration"
  ),
  "livingroom_1":
  (
    "focus light . lamp . chair . door . picture . plant . potted plant . "
    "ceiling lamp . vase . book . pillow . curtain . dish . remote . table . "
    "window . ball candle holder . bird decoration . bottle . bowl . "
    "cabinet . carpet . light switch . ottoman . pen . plant stand . "
    "pyramid candle holder . shelf . sofa . trophy decoration"
  ),
  "livingroom_2":
  (
    "candle . lamp . spotlight . book . couch . door . pillow . plant . "
    "potted plant . sofa . bottle . sliced bread . stool . tv . basket . "
    "bread . cabinet . can of coke . carpet . ceiling lamp . chair . clock . "
    "coffee cup . coffee table . crystal ball decoration . cup . "
    "dice decoration . dish . dvd . fridge . kitchen counter . "
    "kitchen island . light switch . microwave . painting . picture . "
    "range hood . remote . shelf . sink . soccer ball . table"
  ),
  "livingroom_3":
  (
    "chair . focus light . lamp . book . picture . plant . potted plant . "
    "door . pillow . vase . decorative ball . table . bowl . box . candle . "
    "column . light switch . photo . speaker . window . buddha decoration . "
    "cabinet . carpet . chess . coffee table . couch . dvd . "
    "elephant decoration . flower . magazine . mirror . newtons cradle . "
    "painting . shelf . sofa . stool . tablecloth . tray . tv . wall decal"
  ),
  "livingroom_4":
  (
    "book . window . picture . chair . pillow . focus light . lamp . plant . "
    "potted plant . candle . column . curtain . sofa . candlestick . door . "
    "drawer . drawers . fossil decoration . table . bookshelf . cabinet . "
    "carpet . clock . fireplace . firewood . flowers . horse figurine . "
    "light switch . mirror . phone . shelf . vase"
  ),
  "loft":
  (
    "chair . pillow . door . painting . picture . sofa . table . vase . "
    "book . focus light . lamp . light switch . plant . potted plant . boot . "
    "cup . fence . flower . mirror . window . blanket . bowl of apples . "
    "cabinet . carpet . clock . coffee cup . dvd . fireplace . glass . "
    "newspaper . phone . photo . remote . shoe rack . sphere decoration . "
    "stairs . tv . wall lamp . wardrobe"
  ),
  "office_1":
  (
    "chair . focus light . lamp . paper . computer monitor . computer mouse . "
    "keyboard . monitor . mouse . folder . marker . plant . potted plant . "
    "book . box . cup . door . paper box . phone . bench . bottle . file . "
    "shelf . table . clock . coffee machine . eye glasses . file cabinet . "
    "laptop . map wall decal . pen . projector screen . trash bin . "
    "trash can . water cooler . window"
  ),
  "office_2":
  (
    "window . ceiling lamp . coffee cup . column . cup . folder . lamp . "
    "poster . cabinet . chair . computer mouse . drawers . mouse . "
    "mouse pad . computer . computer monitor . keyboard . marker . monitor . "
    "plant . potted plant . box . door . file . notebook . phone . "
    "sticky notes . table . calendar . clock . coffee machine . eraser . "
    "exit sign . files . fire alarm . fire hose . headphones . laptop . "
    "light switch . notice board . paper . pen"
  ),
  "studio":
  (
    "focus light . lamp . book . vase . box . bottle . window . "
    "bird decoration . bookshelf . cabinet . chair . clothes . couch . door . "
    "easel . framed record . guitar . gymnastics decoration . hanger . hat . "
    "light switch . newspaper . phone . picture . pillow . plant . "
    "potted plant . round box . shelf . sofa . table . tv"
  ),
}

TYPE_PROMPTS = {
  "office":
  (
    "chair . computer mouse . folder . mouse . computer monitor . keyboard . "
    "monitor . window . focus light . lamp . marker . paper . plant . "
    "potted plant . box . ceiling lamp . coffee cup . column . cup . door . "
    "phone . poster . cabinet . drawers . file . mouse pad . table . book . "
    "computer . paper box . bench . bottle . clock . coffee machine . "
    "laptop . notebook . pen . shelf . sticky notes . calendar . eraser . "
    "exit sign . eye glasses . file cabinet . files . fire alarm . "
    "fire hose . headphones"
  ),
  "hotel":
  (
    "focus light . lamp . pillow . light switch . towel . picture . chair . "
    "ceiling lamp . mirror . paper . sink . towel rack . bed . "
    "bedside table . bench . blanket . curtain . curtains . eye glasses . "
    "map . night stand . nightstand . paper holder . phone . shower tap . "
    "soap . toilet . toilet paper . trash bin . trash can . tv . vase . "
    "window . bag . bathtub . bed frame . book . cabinet . camera . carpet . "
    "fireplace . flowers . glass . magazine . perfume bottle . remote . "
    "shower . soap bottle"
  ),
  "livingroom":
  (
    "book . chair . focus light . lamp . picture . plant . potted plant . "
    "door . pillow . window . candle . vase . table . column . curtain . "
    "spotlight . ceiling lamp . light switch . sofa . carpet . couch . bowl . "
    "cabinet . decorative ball . dish . remote . shelf . stool . tv . "
    "bottle . box . candlestick . clock . coffee table . drawer . drawers . "
    "dvd . fossil decoration . mirror . painting . photo . sliced bread . "
    "speaker . tray . ball candle holder . basket . bird decoration . "
    "bookshelf"
  ),
  "home":
  (
    "ceiling lamp . lamp . pillow . bottle . tree . wine bottle . chair . "
    "plant . potted plant . cup . curtain . dish . spoon . speaker . drawer . "
    "drawers . spice jar . book . carpet . door . window . bowl . plate . "
    "fork . photo . picture . box . sofa . bedside table . clock . cushion . "
    "face cream . kettle . night stand . nightstand . quilt . trash can . "
    "tv . wine glass . air conditioner . bed . cupboard . desk . dvd player . "
    "kitchen door . kitchen knife . lotion . mattress"
  ),
  "cultural":
  (
    "focus light . lamp . pillow . door . plant . potted plant . vase . "
    "column . glass . table . ceiling lamp . chair . window . lantern . "
    "painting . picture . stool . book . sofa . wall lamp . carpet . "
    "flowers . bowl . chopsticks . light switch . placemat . "
    "symbol decoraion . wardrobe . arabic jar . cabinet . circle decoration . "
    "coffee pot . dish . display ledge . elephant figurine . eye glasses . "
    "fan decoration . hookah . hookah wire . horse figurine . jar . remote . "
    "sake bottle . sauce bowl . shoes . side table . tablecloth . tatami"
  ),
}


# Backward-compatible aliases used across the package.
OFFICE_PROMPT = TYPE_PROMPTS["office"]
HOTEL_PROMPT = TYPE_PROMPTS["hotel"]
DEFAULT_INDOOR_PROMPT = TYPE_PROMPTS["livingroom"]

# Map noisy / multi-token detector phrases → a single canonical label for NMS + graph.
LABEL_CANONICAL: dict[str, str] = {
  "potted plant": "plant",
  "potted cactus": "plant",
  "potted bamboo": "plant",
  "potted branch": "plant",
  "plant": "plant",
  "focus light": "lamp",
  "ceiling lamp": "lamp",
  "ceiling light": "lamp",
  "wall lamp": "lamp",
  "desk light": "lamp",
  "bedroom light": "lamp",
  "spot light": "lamp",
  "spotlight": "lamp",
  "circular light": "lamp",
  "lamp": "lamp",
  "light": "lamp",
  "tv monitor": "monitor",
  "computer monitor": "monitor",
  "television": "monitor",
  "tv": "monitor",
  "monitor": "monitor",
  "computer mouse": "mouse",
  "mouse": "mouse",
  "mouse pad": "mouse pad",
  "coffee cup": "cup",
  "paper cup": "cup",
  "cup": "cup",
  "trash can bin": "trash can",
  "trash bin": "trash can",
  "trash can": "trash can",
  "bin": "bin",
  "cabinet shelf door": "cabinet",
  "cabinet shelf": "cabinet",
  "desk cabinet": "cabinet",
  "sink cabinet": "cabinet",
  "tv cabinet": "cabinet",
  "kitchen cabinet": "cabinet",
  "cabinet": "cabinet",
  "drawers": "drawers",
  "drawer": "drawers",
  "bookshelf": "shelf",
  "bookcase": "shelf",
  "shelves": "shelf",
  "shelf": "shelf",
  "cushion": "pillow",
  "sofa cushion": "pillow",
  "sofa pillows": "pillow",
  "pillows": "pillow",
  "pillow": "pillow",
  "couch": "sofa",
  "sofa": "sofa",
  "painting": "picture",
  "photo": "picture",
  "drawing": "picture",
  "picture": "picture",
  "poster": "poster",
  "window picture": "window",
  "windows": "window",
  "door way": "door",
  "doorway": "door",
  "door frame": "door",
  "entrance door": "door",
  "balcony door": "door",
  "kitchen door": "door",
  "wardrobe door": "wardrobe",
  "bottle jar": "bottle",
  "wine bottle": "bottle",
  "beer bottle": "bottle",
  "sake bottle": "bottle",
  "perfume bottle": "bottle",
  "curtains": "curtain",
  "curtain": "curtain",
  "night stand": "nightstand",
  "nightstand": "nightstand",
  "bedside table": "nightstand",
  "towel rail": "towel rack",
  "towel rack": "towel rack",
  "soap bottle": "soap",
  "soap dish": "soap",
  "soap": "soap",
  "toilet paper": "toilet paper",
  "paper holder": "paper holder",
  "shower tap": "tap",
  "tap": "tap",
  "eye glasses": "glasses",
  "eyeglasses": "glasses",
  "books": "book",
  "tv remote": "remote",
  "remote": "remote",
  "dining chair": "chair",
  "deck chair": "chair",
  "dressing chair": "chair",
  "round table": "table",
  "coffee table": "table",
  "dining table": "table",
  "tea table": "table",
  "side table": "table",
}


def canonicalize_label(label: str) -> str:
  """Normalize a detector phrase to a stable class name."""
  text = (label or "").strip().lower()
  text = re.sub(r"[_\-]+", " ", text)
  text = re.sub(r"\s+", " ", text)
  if not text:
    return "object"
  if text in LABEL_CANONICAL:
    return LABEL_CANONICAL[text]
  for raw, canon in sorted(LABEL_CANONICAL.items(), key=lambda kv: -len(kv[0])):
    if raw in text:
      return canon
  parts = text.split()
  if len(parts) >= 3:
    two = " ".join(parts[:2])
    if two in LABEL_CANONICAL:
      return LABEL_CANONICAL[two]
    if parts[0] in LABEL_CANONICAL:
      return LABEL_CANONICAL[parts[0]]
  return text


_TYPE_ALIASES: dict[str, str] = {
  "office": "office",
  "office_room": "office",
  "office_room2": "office",
  "hotel": "hotel",
  "hotel_room": "hotel",
  "guest_room": "hotel",
  "bedroom_hotel": "hotel",
  "living": "livingroom",
  "living_room": "livingroom",
  "livingroom": "livingroom",
  "indoor": "livingroom",
  "default": "livingroom",
  "home": "home",
  "home_building": "home",
  "cultural": "cultural",
  "theme_room": "cultural",
}


def prompt_for_scene_type(scene_type: str) -> str:
  """Return an open-vocab caption for a scene name or coarse type.

  Resolution order:
    1. Exact Unity scene folder name (``office_2``, ``hotel_room_1``, …)
    2. Coarse type alias (``office``, ``hotel``, ``livingroom``, …)
    3. Fallback: livingroom / indoor aggregate
  """
  key = (scene_type or "").strip().lower().replace("-", "_")
  if key in SCENE_PROMPTS:
    return SCENE_PROMPTS[key]
  type_key = _TYPE_ALIASES.get(key)
  if type_key and type_key in TYPE_PROMPTS:
    return TYPE_PROMPTS[type_key]
  return DEFAULT_INDOOR_PROMPT
