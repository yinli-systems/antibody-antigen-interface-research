from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx

from providers.base import LLMProvider, VisionEncoder
from abcite_core.vision import LocalImageEncoder


class OpenAILLM(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._post = post or httpx.post

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY missing; use ABCITE_PROVIDER=local")
        r = self._post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Cite retrieved chunks only. Abstain if evidence is insufficient.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class OpenAIVision(VisionEncoder):
    name = "openai-vision"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._fallback = LocalImageEncoder()
        self._post = post or httpx.post

    def encode_image(self, path: Path) -> list[float]:
        return self._fallback.encode_image(path)

    def caption(self, path: Path) -> str:
        if not self.api_key:
            return self._fallback.caption(path)
        import base64

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
        r = self._post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Caption this antibody/epitope figure in one sentence.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
                            },
                        ],
                    }
                ],
            },
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
