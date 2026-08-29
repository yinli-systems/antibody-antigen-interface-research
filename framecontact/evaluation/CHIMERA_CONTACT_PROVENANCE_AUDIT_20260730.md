# CHIMERA contact provenance and adapter-v3 audit

Date: 2026-07-30 (Asia/Shanghai)

## Outcome

The proposed feature-only atom14 label reconstruction is falsified. The pinned
CHIMERA feature arrays do not contain a complete residue-specific atom set and
cannot reproduce the release's official contact annotations. The replacement
adapter preserves the official contact tuples and uses PDB CA coordinates only
to anchor residue identities back onto the sequence-aligned feature axes.

The replacement passes the predeclared 95% coverage gate on both evaluation
splits, but it was developed after external-label access. Any rerun using it is
secondary robustness evidence, not a new confirmatory frozen-test result.

## Provenance audit

The score-independent audit used the 375 complexes accepted by the original
exact-axis adapter and compared every provided contact tuple with the pinned
feature arrays and the pinned first-model PDB structure.

| Check | Result |
|---|---:|
| Complexes | 375 |
| Official contact tuples | 16,721 |
| Recovered within 4.5 A by feature atom14 | 11,226 (67.1371%) |
| Feature-atom14 minima matching stored distance at 0.01 A | 5,597 (33.4729%) |
| Full-PDB all-atom minima matching stored distance at 0.01 A | 16,721 (100%) |
| Maximum full-PDB versus stored distance error | 0.0049998 A |
| Hydrogen-only contacts at the 4.5-A threshold | 584 |
| Complexes containing at least one hydrogen-only contact | 35 |
| Complexes whose complete contact set is recovered by feature atom14 | 4/375 |

Manual atom-name inspection confirms the mechanism: the fixed arrays omit
residue-specific distal atoms such as TYR `OH`, THR `OG1`, and several TRP ring
atoms. Some pinned PDB structures also include hydrogens that determine a stored
minimum contact distance. Consequently, neither “atom14” nor “heavy atom” is an
accurate description of the release label materialization.

Machine-readable evidence:

- `evaluation/CHIMERA_CONTACT_PROVENANCE_AUDIT_V1.json`
- SHA-256: `56bcf46517aa2a56e76b696760a9226c0fac14c730e1921b61582787156d6517`
- reproducer: `scripts/audit_chimera_contact_provenance.py`
- `model_scores_read=false`; `metrics_computed=false`

## Adapter-v3 contract

1. Verify the pinned feature/PDB artifacts against the materialization manifest.
2. Use the feature sequences and `N/CA/C` arrays as the model axes and frames.
3. Parse all supported first-model PDB residue atoms and explicit insertion codes.
4. Accept a PDB-to-feature anchor only when CA coordinates agree to 0.001 A,
   residue types agree, and the one-to-one anchors are monotonic.
5. Map the provided official contact tuples onto those axes. Stored all-atom
   distance is used only to disambiguate collapsed insertion-code candidates.
6. Marginalize mapped official pairs into paratope/epitope labels.
7. Fail closed on absent positives, unresolved ambiguity, duplicate pairs,
   annotation disagreement, or artifact drift.

No contacts are recomputed, added, or removed by a new geometric cutoff.

## Coverage-only result

| Split | Eligible | Usable | Excluded | Coverage | 95% gate |
|---|---:|---:|---:|---:|---:|
| `antigen_fold.test` | 292 | 283 | 9 | 96.9178% | pass |
| `temporal.test` | 293 | 288 | 5 | 98.2935% | pass |
| Unique union | 561 | 548 | 13 | 97.6827% | pass |

Across the 548 usable complexes, 24,405 official contact pairs were mapped. A
stored distance resolved 710 ambiguous pairs across 125 complexes. All uniquely
identified pairs matched the pinned PDB minimum distance at 0.01 A. The 13
unique exclusions are:

- 12 complexes where an official contact residue is absent from the
  coordinate-anchored feature axes;
- 1 complex where duplicate annotation keys cannot be uniquely resolved by the
  stored distance.

Machine-readable evidence:

- `evaluation/CHIMERA_PDB_ANCHORED_COVERAGE_AUDIT_V3.json`
- SHA-256: `b6af3948745ab633fdc4dd726ad60e51b8e6aa6436c120b9c6cce7b58d1a0a52`
- `model_scores_read=false`; `metrics_computed=false`
- `confirmatory_interpretation_permitted=false`

The audited 548-record cache was materialized at
`/Volumes/CHIMERA_EXT/chimera-pdb-anchored-cache-v3` (12 MiB). Its manifest
SHA-256 is
`1e465fda22bf4614237fcecff8f7a371300d8ddcfd5e01afc805a3dab2356254`;
the cache-bound coverage audit SHA-256 is
`e345dc8715c8a2724cfdfe471479f75960dc333bd356d61aae1eb1028491788f`.

## Bound authority

- Dataset: `mansoorbaloch/chimera-bench`
- Pinned revision: `54d5a2a1e59a329db7a6c09184652ed225188940`
- Materialization manifest SHA-256:
  `a64333508007254c5d301e6e0eecddfd7591fd1e82b8ecf7c209cdb3a5440b12`

## Next allowed use

The adapter may now materialize an expanded cache and score the already frozen
checkpoints as a transparent secondary rerun. Architecture, loss, threshold, or
checkpoint selection must not use that rerun. A new stronger evidence claim
requires a prospectively frozen, decontaminated external protocol such as the
AsEP epitope-group benchmark, with coverage reported before inference.

AB03-ready artifacts are staged outside the source tree under `work/handoff-v6`:

- `geometry-chimera-eval-source-v6.tar.gz` and its generated receipt;
- `chimera-pdb-anchored-cache-v3.tar.gz` and a separate SHA-256 receipt;
- `scripts/run_chimera_secondary_adapter_v3_ab03.sh`, which binds the cache,
  audit, model, and checkpoint digests and writes a separate secondary output.
