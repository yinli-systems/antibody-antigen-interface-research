"""Azure OpenAI adapter — same LLMProvider surface as local/openai/bedrock."""

from __future__ import annotations

from typing import Callable

import httpx

from providers.base import LLMProvider


class AzureOpenAILLM(LLMProvider):
    name = "azure"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.deployment = deployment
        self._post = post or httpx.post

    def complete(self, prompt: str) -> str:
        if not (self.endpoint and self.api_key and self.deployment):
            raise RuntimeError("Azure OpenAI env vars missing; use ABCITE_PROVIDER=local")
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            "?api-version=2024-10-21"
        )
        r = self._post(
            url,
            headers={"api-key": self.api_key},
            json={"messages": [{"role": "user", "content": prompt}]},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
