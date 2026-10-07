# AF3 semantic recovery is provenance, not accuracy

[Atlas](../README.md) · [Cohorts](../COHORTS.md)

**Evidence level:** `engineering` — Execution/provenance evidence, not model accuracy

Task 0 originally failed after prediction due to a one-second valid CIF timestamp difference. Stricter semantics certified unchanged outputs while preserving the original failure.

| Field | Value |
|---|---|
| workstream | Execution & provenance |
| cohort | Frozen 40-chain completion; dated operational snapshots |
| inputs | Sequence/all-atom/CIF/confidence and seed/sample checks |
| seeds | 78 tests reported by readout, not rerun here |
| estimand | Execution and byte/semantic identity |

## What this does not establish

The semantic-audit branch snapshot was partial; later centered-RePaRank authority covers the completed cohort. Neither an old pending state nor a recovery test is an accuracy result.

## Pinned evidence

1. [extensions/execution_recovery_v1/SEMANTIC_AUDIT_READOUT.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/859a7145bfb10bab87ca72a7dd1fbebe5d6a2399/extensions/execution_recovery_v1/SEMANTIC_AUDIT_READOUT.md)
   - Branch `kevin/af3-semantic-audit-20260930`; commit `859a7145bfb10bab87ca72a7dd1fbebe5d6a2399`.
   - Git blob `af37fc28cd84bff68394bb4b6c3bdcb6dc99b9cd`; SHA-256 `69d9219a3ffa7693f4d4a3fe8c1f857851d16648444d6b903a2ac594161ce2bd`.
2. [extensions/execution_recovery_v1/HANDOFF.md](https://github.com/yinli-systems/antibody-antigen-interface-research/blob/a5729c49e5d32fc05cd09a10f824a7cc6974fb6b/extensions/execution_recovery_v1/HANDOFF.md)
   - Branch `kevin/execution-recovery-20260930`; commit `a5729c49e5d32fc05cd09a10f824a7cc6974fb6b`.
   - Git blob `982c53dd2b90a8d5a52f940458992c8b6247944a`; SHA-256 `24e18374d8a714fd7f5ca2e65dd946cff1370c0685fa03290af63f56d2a6717c`.

**Audit boundary:** committed source bytes and reporting values are checked. Original predictions, analysis implementation and biological interpretation are not independently reproduced; raw artifacts may remain off-Git.
