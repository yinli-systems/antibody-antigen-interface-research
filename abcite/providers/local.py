from __future__ import annotations

from abcite_core.vision import LocalImageEncoder
from providers.base import LLMProvider


class LocalLLM(LLMProvider):
    name = "local"

    def complete(self, prompt: str) -> str:
        return (
            "Extractive local provider: compose only from retrieved chunks. "
            "If chunks do not support a residue-level overlap claim, abstain."
        )


class HistogramVision(LocalImageEncoder):
    """Back-compat alias for LocalImageEncoder (byte n-gram + SVG text hashing)."""

    name = "histogram"
