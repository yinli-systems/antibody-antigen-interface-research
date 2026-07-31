# Partner-aware antibody-antigen interface modeling

This private research repository contains the original adapters, evaluation code,
cluster job templates, protocols, and lightweight evidence artifacts developed for
partner-aware antibody-antigen interface prediction on SAbDab2-derived data.

The repository covers three connected workstreams:

1. SPPIDER-seq adaptation for marginal paratope and epitope prediction.
2. Partner-retrieval experiments and specificity controls.
3. MIPE reproduction and residue-pair interaction diagnostics.

## Locked SPPIDER-seq result

The final ensemble was evaluated on 455 reused benchmark complexes grouped into 84
test clusters.

| Axis | AUROC | AUPRC | MCC |
|---|---:|---:|---:|
| Paratope | 0.948622 | 0.651089 | 0.602334 |
| Epitope | 0.671765 | 0.145445 | 0.126885 |

Cluster bootstrap intervals and fixed partner-permutation controls are included in
`results/sppider_seq/`. The paratope result is strong, while the epitope result and
partner-specificity evidence are modest. The project does not claim native
residue-pair prediction or world-leading performance.

## Partner-retrieval conclusion

The locked mean-pooled dual encoder did not pass its formal specificity gate. Its
paired confidence intervals overlapped strong raw-embedding and single-side
controls. This is retained as a reproducible negative result and as motivation for
the preregistered cross-attention reranker protocol.

## MIPE status

The released-code-compatible five-fold 800-epoch reconstruction and frozen
evaluation are complete. The native mean-bidirectional residue-pair macro AUPRC
is `0.00342` (95% cluster-bootstrap CI `0.00305--0.00381`), compared with
`0.04045` for the predicted-marginal outer-product baseline. The
validation-best epitope micro AUPRC is `0.2804`. These results support the
project's diagnosis that the reconstructed model does not recover useful native
pair correspondence, even when its marginal interface predictions are more
informative.

The supplied external split had pre-scoring schema access and is therefore
reported as a provided external test-split evaluation, not as a pristine blind
test. The released code, appendix, and paper disagree on important settings,
including learning rate, checkpoint selection, graph construction, and dataset
counts. Those discrepancies are documented rather than silently reconciled.

## PAIRFormer status

The prospective PAIRFormer path is blocked at Gate 0. Candidate-mask semantics,
partner-specific controls, calibration metrics, an explicit
contact--distance--orientation decoder, and an exact joint three-outer plus
nine-inner split formulation have been implemented and tested. They are
engineering candidates only: no PAIRFormer biological training, model
selection, held-out evaluation, or performance-improvement claim is authorized
until the exact split, independent audits, fold-local cache equality, label
provenance, and license receipts all pass.

## Antibody foundation-pretraining status

The authenticated OAS corpus contains 224,887,833 unique training records and
25,259,993,406 unique training residue tokens. Exact checkpoint recovery on the
real corpus passes, and a protocol-bound 1,024-array memory-mapped loader reduced
measured continuous-run data preparation from 19.19 seconds to 3.97 seconds in
the recovery gate.

An eight-RTX-5090, 600-update, two-repetition compile-only confirmation measured
202,795 useful residue tokens/s and 46.1% approximate 6N MFU, versus 150,991
tokens/s and 34.3% for eager execution. The candidate remains disqualified:
post-training masked cross-entropy was 2.03% worse than eager against a frozen
1% noninferiority margin, and compiled p95/p50 step-time ratios were 1.137 and
1.128 against a 1.10 ceiling. The failed gate, receipts, and telemetry are kept
under `results/pretraining/antibody_compile_only_confirmation_v1_1343024/`.
No production-scale pretraining or 16/32-GPU result is claimed from this pilot.

## Claim boundary

- The SPPIDER-seq adaptation uses concatenated VH and VL sequences and 4.5 A contact
  labels; these differ from the original paper's task semantics.
- The antibody side remains highly homologous across train and test: 97.58% of test
  instances have a high-coverage training neighbor at or above 70% identity.
- The 455-case benchmark was used in earlier evaluations and is not a pristine
  one-shot confirmatory test.
- Unknown cross-pairs are not measured non-binders.
- No clinical, prospective, external-generalization, or global state-of-the-art
  claim is authorized by the current evidence.

## Repository policy

Raw datasets, model weights, embeddings, full prediction matrices, third-party
source trees, credentials, and cluster-specific identities are intentionally not
stored in Git. Reproduction requires obtaining upstream assets under their own
licenses and verifying the pinned versions listed in `THIRD_PARTY.md`.

## Portable snapshot verification

The curated layout passes Python syntax compilation. On Python 3.12, the portable
test run produced 239 passes, one expected skip for unavailable official weights,
and 38 failures. The failures are physical-evidence integration checks that expect
the original cluster directory layout or byte-identical path-bearing protocol
files. Those files were deliberately relocated or sanitized in this private
snapshot. They must be rerun against the authenticated internal archive; they are
not represented as portable passes here.
