"""Stage 2 — object captioner (Track B / real robot).

SORT3D uses rich 2D VLM captions so the LLM can reason about color, material,
shape, and affordances — not just class labels.

For Track A offline eval on VLA-3D, captions are already filled from
``raw_label`` + ``color_scheme`` in ``scene_inventory.load_vla3d_inventory``.
This module is the hook for Qwen2-VL crop captioning when live crops exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from vlm_pipeline_sort3d.scene_inventory import InventoryObject, SceneInventory

# (object_name, image_bytes_or_path) -> caption text
CaptionCallable = Callable[[str, object], str]

DEFAULT_CAPTION_PROMPT = (
  "Describe the {object} in this image using color, material, shape, "
  "affordances. Format: 'The {object} is <color>, <material>, <shape>'"
)


@dataclass
class CaptionConfig:
  enabled: bool = False
  model_name: str = "Qwen/Qwen2.5-VL-Instruct-3B"


class ObjectCaptioner:
  """Attach VLM captions to inventory objects.

  Currently a stub: without a caption backend it leaves captions unchanged.
  """

  def __init__(
    self,
    caption_fn: Optional[CaptionCallable] = None,
    config: Optional[CaptionConfig] = None,
  ) -> None:
    self.caption_fn = caption_fn
    self.config = config or CaptionConfig()

  def caption_objects(
    self,
    inventory: SceneInventory,
    crops_by_id: Optional[dict] = None,
  ) -> SceneInventory:
    if not self.config.enabled or self.caption_fn is None:
      return inventory

    crops_by_id = crops_by_id or {}
    updated: list[InventoryObject] = []
    for obj in inventory.objects:
      crop = crops_by_id.get(obj.object_id)
      if crop is None:
        updated.append(obj)
        continue
      try:
        caption = self.caption_fn(obj.name, crop)
      except Exception:
        caption = obj.caption
      updated.append(
        InventoryObject(
          object_id=obj.object_id,
          name=obj.name,
          cx=obj.cx,
          cy=obj.cy,
          cz=obj.cz,
          x_length=obj.x_length,
          y_length=obj.y_length,
          z_length=obj.z_length,
          caption=caption or obj.caption,
          aliases=obj.aliases,
        )
      )
    return SceneInventory(scene_name=inventory.scene_name, objects=updated)

  @staticmethod
  def static_caption(obj: InventoryObject) -> str:
    """Fallback used for VLA-3D / no-VLM path."""
    return obj.caption or f"The {obj.name}."
