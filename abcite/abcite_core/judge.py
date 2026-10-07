"""Faithfulness judges: heuristic (CI default) + optional LLM with disk cache."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from abcite_core.metrics import faithfulness as lexical_faithfulness

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "judge_v1.txt"
CACHE_DIR = ROOT / "evals" / ".cache"


@dataclass
class JudgeResult:
    score: float
    supported: bool
    rationale: str
    backend: str
    prompt: str | None = None


class FaithfulnessJudge(Protocol):
    def score(self, question: str, answer: str, cited_texts: list[str], status: str) -> JudgeResult: ...


def judge_prompt(question: str, answer: str, cited_texts: list[str], status: str) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else "Score faithfulness 0-1."
    cites = "\n".join(f"- {t[:400]}" for t in cited_texts) or "(none)"
    return (
        f"{template}\n\nSTATUS: {status}\nQUESTION: {question}\nANSWER: {answer}\n"
        f"CITED CHUNKS:\n{cites}\n"
    )


class HeuristicJudge:
    """Deterministic lexical judge used in CI. Calibrate against human labels."""

    backend = "heuristic"

    def score(self, question: str, answer: str, cited_texts: list[str], status: str) -> JudgeResult:
        s = lexical_faithfulness(answer, cited_texts, status)
        return JudgeResult(
            score=s,
            supported=s >= 0.25 or status == "abstain",
            rationale="lexical token overlap of answer vs cited quotes",
            backend=self.backend,
            prompt=judge_prompt(question, answer, cited_texts, status),
        )


class FakeLLMJudge:
    """Records the prompt contract; scores with the heuristic. Used in unit tests."""

    backend = "fake-llm"
    last_prompt: str | None = None

    def __init__(self, inner: HeuristicJudge | None = None) -> None:
        self.inner = inner or HeuristicJudge()

    def score(self, question: str, answer: str, cited_texts: list[str], status: str) -> JudgeResult:
        prompt = judge_prompt(question, answer, cited_texts, status)
        FakeLLMJudge.last_prompt = prompt
        inner = self.inner.score(question, answer, cited_texts, status)
        return JudgeResult(
            score=inner.score,
            supported=inner.supported,
            rationale="fake LLM wrapping heuristic; prompt recorded",
            backend=self.backend,
            prompt=prompt,
        )


class LLMJudge:
    def __init__(self, complete: Callable[[str], str], cache_dir: Path = CACHE_DIR) -> None:
        self.complete = complete
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def score(self, question: str, answer: str, cited_texts: list[str], status: str) -> JudgeResult:
        prompt = judge_prompt(question, answer, cited_texts, status)
        key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return JudgeResult(**raw, prompt=prompt)
        text = self.complete(prompt)
        parsed = _parse_judge_json(text)
        path.write_text(json.dumps(parsed), encoding="utf-8")
        return JudgeResult(prompt=prompt, **parsed)


def _parse_judge_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        return {"score": 0.0, "supported": False, "rationale": "unparseable judge output", "backend": "llm"}
    data = json.loads(text[start : end + 1])
    score = float(data.get("score", 0.0))
    return {
        "score": max(0.0, min(1.0, score)),
        "supported": bool(data.get("supported", score >= 0.5)),
        "rationale": str(data.get("rationale", ""))[:400],
        "backend": "llm",
    }


def build_judge() -> FaithfulnessJudge:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or ""
    if not key:
        return HeuristicJudge()
    from abcite_core.settings import Settings
    from providers import get_llm

    llm = get_llm(Settings())
    if llm.name == "local":
        return HeuristicJudge()
    return LLMJudge(llm.complete)
