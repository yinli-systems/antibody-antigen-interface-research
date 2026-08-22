from __future__ import annotations

from scripts.audit_hf_esm2_conversion_parity import passes_parity


def test_passes_parity_requires_all_three_numerical_gates() -> None:
    thresholds = {
        "maximum_threshold": 1e-2,
        "mean_threshold": 1e-3,
        "cosine_threshold": 0.999999,
    }
    assert passes_parity(0.0087, 0.00059, 0.99999994, **thresholds)
    assert not passes_parity(0.0101, 0.00059, 0.99999994, **thresholds)
    assert not passes_parity(0.0087, 0.00101, 0.99999994, **thresholds)
    assert not passes_parity(0.0087, 0.00059, 0.999998, **thresholds)
