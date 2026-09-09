"""AWS Bedrock adapter — same LLMProvider surface (UK secondary cloud)."""

from __future__ import annotations

from providers.base import LLMProvider


class BedrockLLM(LLMProvider):
    name = "bedrock"

    def __init__(self, region: str, model_id: str) -> None:
        self.region = region
        self.model_id = model_id

    def complete(self, prompt: str) -> str:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("boto3 not installed; use ABCITE_PROVIDER=local") from exc
        client = boto3.client("bedrock-runtime", region_name=self.region)
        resp = client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        return resp["output"]["message"]["content"][0]["text"]
