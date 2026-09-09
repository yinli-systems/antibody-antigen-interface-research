from abcite_core.tracing import emit_run, tracing_enabled


def test_tracing_off_does_not_crash(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    assert tracing_enabled() is False
    assert emit_run("x", "id", {}, {}) is False


def test_tracing_on_posts(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://example.test/runs")
    seen: dict = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers")

        class R:
            status_code = 200
            text = "ok"

        return R()

    monkeypatch.setattr("abcite_core.tracing.httpx.post", fake_post)
    assert emit_run("abcite.query", "tid", {"q": "x"}, {"status": "answered"}) is True
    assert seen["url"].endswith("/runs")
    assert seen["headers"]["x-api-key"] == "ls-test"
