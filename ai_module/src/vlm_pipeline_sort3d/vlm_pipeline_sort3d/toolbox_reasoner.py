"""Stage 3 — LLM spatial toolbox reasoner.

The LLM receives the question + filtered object list + tool docs + one
in-context example. It emits chain-of-thought and sequential tool calls.
This module parses those calls, executes them on ``SpatialToolbox``, and
feeds results back until a final answer (object id or count) is produced.

This is the main behavioral difference from Pipeline A:
  A: parse question → rigid JSON → one-shot graph search
  C: LLM plans a multi-step tool program over geometry helpers
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from vlm_pipeline_sort3d.scene_inventory import SceneInventory, inventory_prompt_block
from vlm_pipeline_sort3d.spatial_toolbox import SpatialToolbox, Waypoint

LlmCallable = Callable[[str, str], str]

AnswerType = Union[str, int, Waypoint, None]


TOOLBOX_DOC = """\
Available tools (call ONE per step; arguments are Python literals):
  find_all(class_name: str) -> [ids]
  find_near(targets: [ids], anchors: [ids]) -> [ids]
  find_on(targets: [ids], anchors: [ids]) -> [ids]
  find_above(targets: [ids], anchors: [ids]) -> [ids]
  find_below(targets: [ids], anchors: [ids]) -> [ids]
  find_between(targets: [ids], anchor_a: [ids], anchor_b: [ids]) -> [ids]
  find_closest(targets: [ids], references: [ids]) -> [ids]
  find_farthest(targets: [ids], references: [ids]) -> [ids]
  find_left(targets: [ids], anchors: [ids]) -> [ids]
  find_right(targets: [ids], anchors: [ids]) -> [ids]
  order_bottom_to_top(targets: [ids]) -> [ids]
  order_smallest_to_largest(targets: [ids]) -> [ids]
  order_closest_to_farthest(targets: [ids], references: [ids]) -> [ids]
  count(object_ids: [ids]) -> int
  go_near(object_id: str) -> waypoint
  go_between(id1: str, id2: str) -> waypoint
  finish_find(object_id: str)   # final answer for FIND questions
  finish_count(n: int)          # final answer for COUNT questions
"""


IN_CONTEXT_EXAMPLE = """\
Example
Question: Find the bowl on the table closest to the folding screen.
Objects:
- id=3 name='folding screen' center=(1.0,2.0,1.0) size=(0.1,1.5,2.0) caption='The folding screen.'
- id=5 name='table' center=(0.0,0.0,0.4) size=(1.0,0.6,0.7) caption='The table.'
- id=8 name='table' center=(2.0,0.5,0.4) size=(1.2,0.6,0.7) caption='The table.'
- id=1 name='bowl' center=(2.0,0.5,0.8) size=(0.2,0.2,0.1) caption='The bowl is white.'
- id=4 name='bowl' center=(0.0,0.0,0.8) size=(0.2,0.2,0.1) caption='The bowl is blue.'

Reasoning:
screens = find_all("folding screen")
TOOL_CALL: find_all("folding screen")
TOOL_RESULT: ["3"]
tables = find_all("table")
TOOL_CALL: find_all("table")
TOOL_RESULT: ["5", "8"]
t = find_closest(["5", "8"], ["3"])
TOOL_CALL: find_closest(["5", "8"], ["3"])
TOOL_RESULT: ["8"]
bowls = find_all("bowl")
TOOL_CALL: find_all("bowl")
TOOL_RESULT: ["1", "4"]
on_t = find_on(["1", "4"], ["8"])
TOOL_CALL: find_on(["1", "4"], ["8"])
TOOL_RESULT: ["1"]
TOOL_CALL: finish_find("1")
"""


REASONER_SYSTEM_PROMPT = f"""\
You are a spatial reasoning agent for a robot. You solve FIND and COUNT questions
by calling a spatial toolbox. Geometry is handled by the tools — you only plan
which tools to call and in what order.

{TOOLBOX_DOC}

Respond with exactly one line starting with TOOL_CALL: each turn.
Do not invent object IDs that are not in the provided object list.
When ready, call finish_find or finish_count.

{IN_CONTEXT_EXAMPLE}
"""


@dataclass
class ToolCall:
  name: str
  args: List[Any] = field(default_factory=list)
  kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonerResult:
  success: bool
  answer: AnswerType
  answer_kind: str  # "find" | "count" | "waypoint" | "none"
  object_id: Optional[str] = None
  count: Optional[int] = None
  waypoint: Optional[Waypoint] = None
  trace: List[str] = field(default_factory=list)
  error: str = ""


_TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def parse_tool_call(text: str) -> Optional[ToolCall]:
  """Parse the first TOOL_CALL: name(args...) line from an LLM response."""
  match = _TOOL_CALL_RE.search(text.strip())
  if not match:
    # Tolerate a bare function call on its own line.
    for line in text.strip().splitlines():
      line = line.strip()
      if re.match(r"^[a-z_]+\(.*\)$", line):
        match_line = line
        break
    else:
      return None
  else:
    match_line = match.group(1).strip()

  try:
    tree = ast.parse(match_line, mode="eval")
  except SyntaxError:
    return None
  if not isinstance(tree.body, ast.Call) or not isinstance(tree.body.func, ast.Name):
    return None

  def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
      return node.value
    if isinstance(node, ast.List):
      return [_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
      return [_literal(elt) for elt in node.elts]
    if isinstance(node, ast.Name) and node.id in ("True", "False", "None"):
      return {"True": True, "False": False, "None": None}[node.id]
    # Allow bare identifiers as strings (object ids sometimes unquoted).
    if isinstance(node, ast.Name):
      return node.id
    raise ValueError(f"Unsupported argument node: {ast.dump(node)}")

  args = [_literal(a) for a in tree.body.args]
  kwargs = {kw.arg: _literal(kw.value) for kw in tree.body.keywords if kw.arg}
  return ToolCall(name=tree.body.func.id, args=args, kwargs=kwargs)


class ToolboxReasoner:
  """Multi-step LLM ↔ toolbox execution loop."""

  def __init__(
    self,
    llm: Optional[LlmCallable] = None,
    max_steps: int = 12,
  ) -> None:
    self.llm = llm
    self.max_steps = max_steps

  def reason(
    self,
    question: str,
    inventory: SceneInventory,
    toolbox: SpatialToolbox,
    *,
    question_type: str = "find",
  ) -> ReasonerResult:
    if self.llm is None:
      return ReasonerResult(
        success=False,
        answer=None,
        answer_kind="none",
        error="No LLM backend configured for ToolboxReasoner",
      )

    history: List[str] = [
      f"Question: {question}",
      f"Question type: {question_type.upper()}",
      "Objects:",
      inventory_prompt_block(inventory),
      "",
      "Begin. Output one TOOL_CALL per response.",
    ]
    trace: List[str] = []

    for step in range(self.max_steps):
      user_prompt = "\n".join(history)
      raw = self.llm(REASONER_SYSTEM_PROMPT, user_prompt)
      trace.append(f"STEP {step + 1} LLM:\n{raw}")
      call = parse_tool_call(raw)
      if call is None:
        history.append(f"Assistant: {raw}")
        history.append("System: Could not parse a TOOL_CALL. Reply with TOOL_CALL: name(...).")
        continue

      trace.append(f"STEP {step + 1} CALL: {call.name}{call.args} {call.kwargs}")

      if call.name == "finish_find":
        oid = str(call.args[0]) if call.args else call.kwargs.get("object_id")
        if oid is None:
          return ReasonerResult(False, None, "none", error="finish_find missing id", trace=trace)
        wp = toolbox.go_near(str(oid))
        return ReasonerResult(
          success=True,
          answer=str(oid),
          answer_kind="find",
          object_id=str(oid),
          waypoint=wp,
          trace=trace,
        )

      if call.name == "finish_count":
        n = call.args[0] if call.args else call.kwargs.get("n")
        try:
          n_int = int(n)
        except (TypeError, ValueError):
          return ReasonerResult(False, None, "none", error="finish_count bad int", trace=trace)
        return ReasonerResult(
          success=True,
          answer=n_int,
          answer_kind="count",
          count=n_int,
          trace=trace,
        )

      try:
        result = toolbox.call(call.name, *call.args, **call.kwargs)
      except Exception as exc:  # noqa: BLE001 — surface tool errors to the LLM
        result = f"ERROR: {exc}"

      result_text = _format_tool_result(result)
      trace.append(f"STEP {step + 1} RESULT: {result_text}")
      history.append(f"Assistant: TOOL_CALL: {call.name}({_fmt_args(call)})")
      history.append(f"TOOL_RESULT: {result_text}")

    return ReasonerResult(
      success=False,
      answer=None,
      answer_kind="none",
      error=f"Exceeded max_steps={self.max_steps}",
      trace=trace,
    )


def _fmt_args(call: ToolCall) -> str:
  parts = [repr(a) for a in call.args]
  parts += [f"{k}={v!r}" for k, v in call.kwargs.items()]
  return ", ".join(parts)


def _format_tool_result(result: Any) -> str:
  if isinstance(result, Waypoint):
    return json.dumps({"x": result.x, "y": result.y, "yaw": result.yaw})
  if isinstance(result, (list, tuple, int, float, str, bool)) or result is None:
    return json.dumps(result)
  return repr(result)
