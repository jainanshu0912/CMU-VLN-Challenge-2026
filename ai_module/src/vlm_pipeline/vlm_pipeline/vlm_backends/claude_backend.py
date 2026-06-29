"""Anthropic Claude backend."""

from __future__ import annotations

import os
from typing import Optional

from vlm_pipeline.vlm_backends.base import VlmBackend, VlmBackendError, VlmResponse


class ClaudeBackend(VlmBackend):
  """Claude models via the Anthropic SDK."""

  def __init__(
    self,
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
  ) -> None:
    self._model = model
    self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

  @property
  def provider(self) -> str:
    return "anthropic"

  @property
  def model(self) -> str:
    return self._model

  def _check_availability(self) -> None:
    if not self._api_key:
      raise VlmBackendError("ANTHROPIC_API_KEY is not set")
    try:
      import anthropic  # noqa: F401
    except ImportError as exc:
      raise ImportError(
        "anthropic package is required for ClaudeBackend. Install with: pip install anthropic"
      ) from exc

  def complete(
    self,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
  ) -> VlmResponse:
    self._check_availability()

    try:
      from anthropic import Anthropic
    except ImportError as exc:
      raise ImportError(
        "anthropic package is required for ClaudeBackend. Install with: pip install anthropic"
      ) from exc

    client = Anthropic(api_key=self._api_key)
    request_kwargs = {
      "model": self._model,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "messages": [{"role": "user", "content": prompt}],
    }
    if system:
      request_kwargs["system"] = system

    try:
      response = client.messages.create(**request_kwargs)
    except Exception as exc:
      raise VlmBackendError(f"Anthropic request failed: {exc}") from exc

    text_blocks = [
      block.text
      for block in response.content
      if hasattr(block, "text") and block.text
    ]
    text = "\n".join(text_blocks).strip()
    return VlmResponse(
      text=text,
      model=self._model,
      provider=self.provider,
      raw=response,
    )
