"""Tests for per-scene / type prompts and label canonicalization."""

from __future__ import annotations

import unittest

from vlm_pipeline_live.label_utils import (
  DEFAULT_INDOOR_PROMPT,
  HOTEL_PROMPT,
  OFFICE_PROMPT,
  SCENE_PROMPTS,
  TYPE_PROMPTS,
  canonicalize_label,
  prompt_for_scene_type,
)


class LabelUtilsTests(unittest.TestCase):
  def test_all_unity_scenes_have_prompts(self) -> None:
    expected = {
      "arabic_room",
      "chinese_room",
      "home_building_1",
      "home_building_2",
      "hotel_room_1",
      "hotel_room_2",
      "japanese_room",
      "livingroom_1",
      "livingroom_2",
      "livingroom_3",
      "livingroom_4",
      "loft",
      "office_1",
      "office_2",
      "studio",
    }
    self.assertEqual(set(SCENE_PROMPTS), expected)

  def test_scene_specific_terms(self) -> None:
    self.assertIn("keyboard", prompt_for_scene_type("office_1"))
    self.assertIn("mouse pad", prompt_for_scene_type("office_2"))
    self.assertIn("towel", prompt_for_scene_type("hotel_room_1"))
    self.assertIn("hookah", prompt_for_scene_type("arabic_room"))
    self.assertIn("tatami", prompt_for_scene_type("japanese_room"))
    self.assertIn("guitar", prompt_for_scene_type("studio"))

  def test_type_aliases(self) -> None:
    self.assertEqual(prompt_for_scene_type("office"), OFFICE_PROMPT)
    self.assertEqual(prompt_for_scene_type("hotel"), HOTEL_PROMPT)
    self.assertEqual(prompt_for_scene_type("indoor"), DEFAULT_INDOOR_PROMPT)
    self.assertIn("office", TYPE_PROMPTS)
    self.assertIn("hotel", TYPE_PROMPTS)

  def test_canonicalize(self) -> None:
    self.assertEqual(canonicalize_label("trash can bin"), "trash can")
    self.assertEqual(canonicalize_label("cabinet shelf"), "cabinet")
    self.assertEqual(canonicalize_label("ceiling lamp"), "lamp")
    self.assertEqual(canonicalize_label("computer mouse"), "mouse")
    self.assertEqual(canonicalize_label("towel rail"), "towel rack")
    self.assertEqual(canonicalize_label("bedside table"), "nightstand")
    self.assertEqual(canonicalize_label("soap bottle"), "soap")


if __name__ == "__main__":
  unittest.main()
