"""OpenAI GPT backend."""

from __future__ import annotations

import os
from typing import Optional

from vlm_pipeline.vlm_backends.base import VlmBackend, VlmBackendError, VlmResponse


class OpenAIBackend(VlmBackend):
  """GPT-4o (and other OpenAI chat models) via the official SDK."""

  def __init__(
    self,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
  ) -> None:
    self._model = model
    self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

  @property
  def provider(self) -> str:
    return "openai"

  @property
  def model(self) -> str:
    return self._model

  def _check_availability(self) -> None:
    if not self._api_key:
      raise VlmBackendError("OPENAI_API_KEY is not set")
    try:
      import openai  # noqa: F401
    except ImportError as exc:
      raise ImportError(
        "openai package is required for OpenAIBackend. Install with: pip install openai"
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
      from openai import OpenAI
    except ImportError as exc:
      raise ImportError(
        "openai package is required for OpenAIBackend. Install with: pip install openai"
      ) from exc

    client = OpenAI(api_key=self._api_key)
    messages = []
    if system:
      messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
      response = client.chat.completions.create(
        model=self._model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
      )
    except Exception as exc:
      raise VlmBackendError(f"OpenAI request failed: {exc}") from exc

    text = response.choices[0].message.content or ""
    return VlmResponse(
      text=text.strip(),
      model=self._model,
      provider=self.provider,
      raw=response,
    )
