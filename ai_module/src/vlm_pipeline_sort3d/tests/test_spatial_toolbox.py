"""Unit tests for Pipeline C spatial toolbox (no LLM required)."""

from __future__ import annotations

from vlm_pipeline_sort3d.scene_inventory import InventoryObject, SceneInventory
from vlm_pipeline_sort3d.spatial_toolbox import SpatialToolbox, ToolboxConfig
from vlm_pipeline_sort3d.toolbox_reasoner import parse_tool_call
from vlm_pipeline_sort3d.question_classifier import QuestionType, classify_question


def _toy_inventory() -> SceneInventory:
  objs = [
    InventoryObject("3", "folding screen", 1.0, 2.0, 1.0, 0.1, 1.5, 2.0),
    InventoryObject("5", "table", 0.0, 0.0, 0.4, 1.0, 0.6, 0.7),
    InventoryObject("8", "table", 2.0, 0.5, 0.4, 1.2, 0.6, 0.7),
    InventoryObject("1", "bowl", 2.0, 0.5, 0.85, 0.2, 0.2, 0.1, caption="The bowl is white."),
    InventoryObject("4", "bowl", 0.0, 0.0, 0.85, 0.2, 0.2, 0.1, caption="The bowl is blue."),
  ]
  return SceneInventory("toy", objs)


def test_find_all_and_closest_on_chain():
  tb = SpatialToolbox(_toy_inventory(), ToolboxConfig())
  screens = tb.find_all("folding screen")
  assert screens == ["3"]
  tables = tb.find_all("table")
  assert set(tables) == {"5", "8"}
  closest_table = tb.find_closest(tables, screens)
  assert closest_table == ["8"]
  bowls = tb.find_all("bowl")
  on_t = tb.find_on(bowls, closest_table)
  assert on_t == ["1"]


def test_count():
  tb = SpatialToolbox(_toy_inventory())
  assert tb.count(tb.find_all("bowl")) == 2


def test_parse_tool_call():
  call = parse_tool_call('Thought...\nTOOL_CALL: find_closest(["5", "8"], ["3"])\n')
  assert call is not None
  assert call.name == "find_closest"
  assert call.args == [["5", "8"], ["3"]]


def test_classify_question():
  assert classify_question("How many bowls are on the table?") == QuestionType.COUNT
  assert classify_question("Find the bowl on the table.") == QuestionType.FIND
  assert classify_question("The lantern between the vase and the stone.") == QuestionType.FIND
