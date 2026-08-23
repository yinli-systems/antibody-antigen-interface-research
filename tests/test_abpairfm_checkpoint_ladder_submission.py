from __future__ import annotations

from scripts.maybe_submit_abpairfm_checkpoint_ladder_probe import selected_target


def test_selected_target_uses_first_checkpoint_with_bounded_overshoot() -> None:
    assert selected_target(249_999_999) is None
    assert selected_target(250_000_000) == 250_000_000
    assert selected_target(252_633_502) == 250_000_000
    assert selected_target(270_000_000) == 250_000_000
    assert selected_target(270_000_001) is None
    assert selected_target(499_999_999) is None
    assert selected_target(500_000_000) == 500_000_000
    assert selected_target(520_000_000) == 500_000_000
    assert selected_target(520_000_001) is None
    assert selected_target(1_000_000_000) == 1_000_000_000
    assert selected_target(1_020_000_000) == 1_000_000_000
