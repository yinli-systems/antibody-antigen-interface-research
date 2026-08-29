from audit_pairwise_decontamination import audit_hits


def _hit(row: str, query_chain: str, instance: str, target_chain: str, identity: float) -> str:
    return (
        f"{row}|{query_chain}\tasep|{instance}|{target_chain}|sha256={'a' * 64}"
        f"\t{identity:.3f}\t100\t1.000\t1.000\t1e-20\t100\n"
    )


def test_requires_both_chains_to_match_same_instance() -> None:
    first = "1" * 64
    second = "2" * 64
    lines = [
        _hit(first, "heavy", "complex-a", "heavy", 0.91),
        _hit(second, "heavy", "complex-a", "heavy", 0.99),
        _hit(first, "light", "complex-a", "light", 0.93),
        _hit(second, "light", "complex-b", "light", 0.99),
    ]
    result = audit_hits(lines, source_pair_count=3)
    assert result["identity_grid"]["0.90"]["excluded_same_development_instance"] == 1
    assert result["identity_grid"]["0.90"]["kept_pairs"] == 2
    assert result["identity_grid"]["0.95"]["excluded_same_development_instance"] == 0
    assert result["frozen_composite_rule"]["excluded_pairs"] == 2


def test_ignores_cross_chain_hits_fail_closed_in_audit_count() -> None:
    row = "3" * 64
    result = audit_hits(
        [_hit(row, "heavy", "complex-a", "light", 1.0)],
        source_pair_count=1,
    )
    assert result["counts"]["target_chain_mismatch_hits"] == 1
    assert result["identity_grid"]["0.50"]["excluded_same_development_instance"] == 0
