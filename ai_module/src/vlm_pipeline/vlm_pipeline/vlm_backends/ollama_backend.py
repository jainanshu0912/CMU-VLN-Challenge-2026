"""Ollama local LLM backend (free, no API key)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from vlm_pipeline.vlm_backends.base import VlmBackend, VlmBackendError, VlmResponse


class OllamaBackend(VlmBackend):
  """Local models via Ollama HTTP API (https://ollama.com)."""

  def __init__(
    self,
    model: str = "llama3.2",
    base_url: Optional[str] = None,
  ) -> None:
    self._model = model
    self._base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

  @property
  def provider(self) -> str:
    return "ollama"

  @property
  def model(self) -> str:
    return self._model

  def _check_availability(self) -> None:
    try:
      with urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=2) as response:
        if response.status != 200:
          raise VlmBackendError(f"Ollama returned status {response.status}")
    except urllib.error.URLError as exc:
      raise VlmBackendError(
        f"Ollama is not reachable at {self._base_url}. "
        "Install from https://ollama.com and run: ollama pull llama3.2"
      ) from exc

  def complete(
    self,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **_: object,
  ) -> VlmResponse:
    self._check_availability()

    messages = []
    if system:
      messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
      "model": self._model,
      "messages": messages,
      "stream": False,
      "options": {
        "temperature": temperature,
        "num_predict": max_tokens,
      },
    }

    request = urllib.request.Request(
      f"{self._base_url}/api/chat",
      data=json.dumps(payload).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )

    try:
      with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
      raise VlmBackendError(f"Ollama request failed: {exc}") from exc

    text = (body.get("message", {}).get("content") or "").strip()
    return VlmResponse(
      text=text,
      model=self._model,
      provider=self.provider,
      raw=body,
    )
