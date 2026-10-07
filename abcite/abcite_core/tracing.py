"""Optional LangSmith tracing. Missing keys must never crash the API."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger("abcite.trace")


def tracing_enabled() -> bool:
    flag = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or ""
    return flag.lower() in {"1", "true", "yes"}


def _api_key() -> str:
    return os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or ""


def emit_run(name: str, trace_id: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> bool:
    """POST a run if tracing is on. Returns True iff a remote emit was attempted successfully."""
    if not tracing_enabled():
        return False
    key = _api_key()
    if not key:
        log.info("LangSmith tracing flagged on but no API key; skipping emit")
        return False
    payload = {
        "name": name,
        "run_type": "chain",
        "id": trace_id,
        "inputs": inputs,
        "outputs": outputs,
        "session_name": os.environ.get("LANGSMITH_PROJECT")
        or os.environ.get("LANGCHAIN_PROJECT")
        or "abcite",
    }
    try:
        url = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com/runs")
        r = httpx.post(
            url,
            headers={"x-api-key": key},
            json=payload,
            timeout=5.0,
        )
        if r.status_code >= 400:
            log.warning("LangSmith emit status=%s body=%s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as exc:  # tracing must not break /query
        log.warning("LangSmith emit skipped: %s", exc)
        return False
