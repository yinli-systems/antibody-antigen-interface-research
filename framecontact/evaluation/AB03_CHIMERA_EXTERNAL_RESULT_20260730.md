# AB03 CHIMERA zero-shot external residue-interface result

## Outcome

The six frozen epoch-20 checkpoints (candidates `sequence_dense` and
`spatial_topology`; seeds 17, 42, and 73) were evaluated on the strict mapped
subset of the pinned CHIMERA benchmark on an NVIDIA A100-PCIE-40GB. ESM-2
embeddings used `facebook/esm2_t33_650M_UR50D` at revision
`08e4846e537177426273712802403f7ba8261b6c` and were generated with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

This is a real zero-shot external evaluation, but it is **not a confirmatory
frozen-test result**. The preregistered 95% adapter-coverage gate failed on both
co-primary splits. It is also not the canonical CHIMERA CDR-design benchmark.

## Coverage

| Split | Eligible | Strictly mapped | Coverage | 95% gate |
|---|---:|---:|---:|---:|
| `antigen_fold.test` | 292 | 192 | 65.7534% | fail |
| `temporal.test` | 293 | 196 | 66.8942% | fail |

The strict cache contains 375 unique complexes across the split union. Mapping
failures were determined before scoring and remained score-independent.

## Epitope metrics

`mAP` is mean per-complex AUPRC. `pAP` is pooled residue AUPRC. Precision,
recall, F1, and MCC use the frozen threshold 0.5.

| Split | Candidate | mAP | pAP | Pooled AUROC | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `antigen_fold.test` | `sequence_dense` | 0.572380 | 0.420891 | 0.809266 | 0.382552 | 0.496937 | 0.432306 | 0.379664 |
| `antigen_fold.test` | `spatial_topology` | 0.576836 | 0.252825 | 0.724627 | 0.238583 | 0.312021 | 0.270405 | 0.199659 |
| `temporal.test` | `sequence_dense` | 0.307776 | 0.273784 | 0.748427 | 0.222397 | 0.417810 | 0.290280 | 0.239567 |
| `temporal.test` | `spatial_topology` | 0.324708 | 0.255077 | 0.746921 | 0.199975 | 0.435566 | 0.274105 | 0.223944 |

The preregistered primary endpoint was `spatial_topology - sequence_dense` in
mean per-complex epitope AUPRC:

| Split | Delta | 95% group-bootstrap CI | Raw p | Holm p | Statistical success before coverage gate | Confirmatory support |
|---|---:|---:|---:|---:|---:|---:|
| `antigen_fold.test` | +0.004456 | [-0.008832, +0.017795] | 0.514349 | 0.514349 | false | false |
| `temporal.test` | +0.016933 | [+0.003391, +0.030752] | 0.013999 | 0.027997 | true | false |

The temporal mean-per-complex endpoint is positive before applying the coverage
gate. However, pooled epitope AUPRC is lower for `spatial_topology` on both
splits, and no confirmatory claim is allowed.

## Paratope metrics

| Split | Candidate | mAP | pAP | Pooled AUROC | Precision | Recall | F1 | MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `antigen_fold.test` | `sequence_dense` | 0.659569 | 0.612733 | 0.956902 | 0.377995 | 0.887269 | 0.530139 | 0.549002 |
| `antigen_fold.test` | `spatial_topology` | 0.680742 | 0.638800 | 0.958715 | 0.464944 | 0.847696 | 0.600516 | 0.603161 |
| `temporal.test` | `sequence_dense` | 0.730485 | 0.698137 | 0.966357 | 0.444912 | 0.898786 | 0.595194 | 0.600230 |
| `temporal.test` | `spatial_topology` | 0.755189 | 0.735117 | 0.972192 | 0.511609 | 0.898544 | 0.651990 | 0.651241 |

## Runtime and provenance

- ESM sequence coverage: 707/707 unique sequences; embedding runtime 48.005 s.
- Prediction coverage: 375 complexes, six runs, BF16; total runtime 48.868 s.
- GPU: NVIDIA A100-PCIE-40GB; PyTorch 2.5.1; CUDA 12.4.
- Bootstrap: 10,000 paired antigen-sequence-group replicates; seed 20260730;
  Holm correction across the two co-primary splits.
- ESM weight SHA-256:
  `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`.
- Adapter manifest SHA-256:
  `ad45a5a901045784aa7fdf038bf5e3d0d1018a26eb4c107c9b48fdcf11bc62af`.
- ESM embedding manifest SHA-256:
  `6f656ac194b35c5b530a871ae20c9dd488a32231a61dc8af1607e0a3980592c1`.
- Prediction manifest SHA-256:
  `d06606ab7210cda31891846d2cac621a9dd0ac1a6050c3832ad5587fad5661cb`.
- Result JSON SHA-256:
  `4b81ee66305280fe0693ce7c15b5a0aa61fbaa40519f2dec2da0922d29949125`.
- Source archive v6 SHA-256:
  `bb72a8d8adf5312aec31ef0ed4ca014dfb5d82dfb128daa9762e8a4e26a6416c`.

Remote authoritative artifacts are under
`/home/liyin/geometry-self-attention-frozen-eval-20260730/outputs/` and their
combined checksum receipt is
`/home/liyin/geometry-self-attention-frozen-eval-20260730/logs/chimera-frozen-final-v6-sha256.txt`.

## Engineering notes

The first complete attempt was quarantined before scoring because CUDA matmul
determinism was warn-only without `CUBLAS_WORKSPACE_CONFIG`. A second run added
a fail-closed determinism guard. The deterministic prediction run then exposed
a root-owned `0700` temporary-directory handoff bug; source v6 makes prediction
and result directories host-readable and prevents an empty manifest digest from
being passed to the scorer. No model, checkpoint, metric, threshold, mapping
rule, or ensemble weight was changed in either repair.
