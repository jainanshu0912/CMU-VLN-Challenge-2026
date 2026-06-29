"""Google Gemini backend."""

from __future__ import annotations

import os
from typing import Optional

from vlm_pipeline.vlm_backends.base import VlmBackend, VlmBackendError, VlmResponse


class GeminiBackend(VlmBackend):
  """Gemini models via google-generativeai."""

  def __init__(
    self,
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
  ) -> None:
    self._model = model
    self._api_key = (
      api_key
      or os.environ.get("GOOGLE_API_KEY")
      or os.environ.get("GEMINI_API_KEY")
    )

  @property
  def provider(self) -> str:
    return "google"

  @property
  def model(self) -> str:
    return self._model

  def _check_availability(self) -> None:
    if not self._api_key:
      raise VlmBackendError("GOOGLE_API_KEY or GEMINI_API_KEY is not set")
    try:
      import google.generativeai  # noqa: F401
    except ImportError as exc:
      raise ImportError(
        "google-generativeai is required for GeminiBackend. "
        "Install with: pip install google-generativeai"
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
      import google.generativeai as genai
      from google.generativeai.types import GenerationConfig
    except ImportError as exc:
      raise ImportError(
        "google-generativeai is required for GeminiBackend. "
        "Install with: pip install google-generativeai"
      ) from exc

    genai.configure(api_key=self._api_key)
    gemini_model = genai.GenerativeModel(
      model_name=self._model,
      system_instruction=system,
    )

    try:
      response = gemini_model.generate_content(
        prompt,
        generation_config=GenerationConfig(
          temperature=temperature,
          max_output_tokens=max_tokens,
        ),
      )
    except Exception as exc:
      raise VlmBackendError(f"Gemini request failed: {exc}") from exc

    text = (response.text or "").strip()
    return VlmResponse(
      text=text,
      model=self._model,
      provider=self.provider,
      raw=response,
    )
