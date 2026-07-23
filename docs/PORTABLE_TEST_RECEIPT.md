# Portable test receipt

- Environment: macOS arm64, Python 3.12.13
- Command: `.venv/bin/pytest -q`
- Collected: 278 tests
- Passed: 239
- Failed: 38
- Skipped: 1
- Additional subtests passed: 33

The skipped test requires `SPPIDER_OFFICIAL_MODEL_DIR` and the upstream released
checkpoint files.

The 38 failures are integration checks bound to the original physical repository
layout, byte-identical protocol digests, or non-portable evidence files. The
curated repository intentionally moves and sanitizes those files. The original
internal execution previously passed its environment-bound suites; this receipt
does not convert those historical results into portable passes.
