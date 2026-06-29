"""VLM backend adapters for GPT-4o, Claude, and Gemini."""

from vlm_pipeline.vlm_backends.base import (
  DEFAULT_BACKENDS,
  VlmBackend,
  VlmBackendError,
  VlmResponse,
  create_backend,
)
from vlm_pipeline.vlm_backends.claude_backend import ClaudeBackend
from vlm_pipeline.vlm_backends.gemini_backend import GeminiBackend
from vlm_pipeline.vlm_backends.ollama_backend import OllamaBackend
from vlm_pipeline.vlm_backends.openai_backend import OpenAIBackend

__all__ = [
  "ClaudeBackend",
  "DEFAULT_BACKENDS",
  "GeminiBackend",
  "OllamaBackend",
  "OpenAIBackend",
  "VlmBackend",
  "VlmBackendError",
  "VlmResponse",
  "create_backend",
]
