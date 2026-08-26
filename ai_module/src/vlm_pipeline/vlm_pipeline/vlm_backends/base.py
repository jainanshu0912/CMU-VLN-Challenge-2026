"""Abstract VLM backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

DEFAULT_BACKENDS: Dict[str, str] = {
  "ollama": "llama3.2",
  "gpt-4o": "gpt-4o",
  "openai": "gpt-4o",
  "claude": "claude-sonnet-4-20250514",
  "anthropic": "claude-sonnet-4-20250514",
  "gemini": "gemini-3.6-flash",
  "google": "gemini-3.6-flash",
}


class VlmBackendError(RuntimeError):
  """Raised when a backend cannot complete a request."""


@dataclass(frozen=True)
class VlmResponse:
  text: str
  model: str
  provider: str
  raw: Optional[Any] = None


class VlmBackend(ABC):
  """Provider adapter: prompt in, text response out."""

  @property
  @abstractmethod
  def provider(self) -> str:
    """Short provider name (openai, anthropic, google)."""

  @property
  @abstractmethod
  def model(self) -> str:
    """Model identifier passed to the provider API."""

  @abstractmethod
  def complete(
    self,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
  ) -> VlmResponse:
    """Run a single-turn completion and return the model text."""

  def is_available(self) -> bool:
    """Return True when credentials and SDK are ready."""
    try:
      self._check_availability()
      return True
    except (ImportError, VlmBackendError, ValueError):
      return False

  def _check_availability(self) -> None:
    raise NotImplementedError


def create_backend(
  backend_name: str,
  model: Optional[str] = None,
  api_key: Optional[str] = None,
) -> VlmBackend:
  """
  Create a backend from a short name.

  Accepted names: ollama, gemini, google, gpt-4o, openai, claude, anthropic.
  """
  try:
    from vlm_pipeline.vlm_backends.claude_backend import ClaudeBackend
    from vlm_pipeline.vlm_backends.gemini_backend import GeminiBackend
    from vlm_pipeline.vlm_backends.ollama_backend import OllamaBackend
    from vlm_pipeline.vlm_backends.openai_backend import OpenAIBackend
  except ImportError as exc:
    raise ImportError(
      "LLM backends are not in this checkout. Leave use_llm_parser:=false "
      "(the eval default) or restore the local adapter files."
    ) from exc

  normalized = backend_name.strip().lower()
  registry: Dict[str, Type[VlmBackend]] = {
    "ollama": OllamaBackend,
    "gpt-4o": OpenAIBackend,
    "openai": OpenAIBackend,
    "claude": ClaudeBackend,
    "anthropic": ClaudeBackend,
    "gemini": GeminiBackend,
    "google": GeminiBackend,
  }

  backend_cls = registry.get(normalized)
  if backend_cls is None:
    supported = ", ".join(sorted(registry))
    raise ValueError(f"Unknown VLM backend '{backend_name}'. Supported: {supported}")

  resolved_model = model or DEFAULT_BACKENDS.get(normalized)
  if resolved_model is None:
    raise ValueError(f"No default model configured for backend '{backend_name}'")

  if normalized == "ollama":
    return OllamaBackend(model=resolved_model)

  return backend_cls(model=resolved_model, api_key=api_key)
