# Authority-Validation Boundary Correction Report

Status: independently verified on 2026-07-23 CST. No model was trained or
evaluated during this correction.

## Authority and rule

The immutable authority split has file SHA-256
`e228d5106f5befabf3240ba6d134e2bbc142f948a5d40154a219599d40928d3b`
and contains 3,229 train, 216 validation, and 218 sealed-test identifiers.

Every authority-validation component without a threshold-qualified exact global
Needleman–Wunsch edge to authority train was retained. The threshold was at
least 80% aligned-region identity and at least 80% coverage of each full
sequence. Components with any crossing member were assigned to
`excluded_validation`; they were not moved into training. There was no target
size and no random subsampling.

## Feasibility and final counts

| Boundary | Train | Validation | Excluded validation | Eligible / excluded components | Final exact-NW crossings |
|---|---:|---:|---:|---:|---:|
| Concat antibody + antigen | 3,229 | 128 | 88 | 42 / 25 | 0 / 0 |
| Concat antibody + VH + VL + antigen | 3,229 | 43 | 173 | 15 / 52 | 0 / 0 / 0 / 0 |

For both boundaries, authority-train intersection with validation is zero, and
sealed-test intersection with every emitted partition is zero. The three emitted
partitions are pairwise disjoint and account exactly for all 3,445 development
identifiers.

## Immutable artifacts

- `concat_boundary_v2.json`: file SHA-256
  `87a90b8dbee7ff2bd8b110797346238026e4c50c5796cdc9e5e80c383e676c21`;
  scientific payload SHA-256
  `7eb54d002a4837ca712b9ef710cccc250cde5c475e66a772e397f9e6b847ee34`.
- `chain_boundary_v2.json`: file SHA-256
  `2bf2f0ba0709a466bbe6f57d1ee042e8d2406eb557d9a71cb92fb30d8ce51344`;
  scientific payload SHA-256
  `5bda6383f3cf9671bd74041cb16f42cdcb65f4fc5db2abd3278d70cee518d8b9`.
- `independent_verification_v2.json`: file SHA-256
  `3bd8186603d908a99e1fbf00ead3697dd276ccd798b4f91bcbdda81441bd80e8`;
  scientific payload SHA-256
  `44c6b763419087a0b21faae4b55e03aa0904380a71f4cec6d8df3945b48f6a29`.
- Protocol v1.4 correction: file SHA-256
  `34c3bbe82baf431c87fae0d54da965819e91c9762f000487dd076ba36b76eaf5`.

Independent exhaustive verification ran as Slurm job `1276567` and completed
successfully in 25 seconds. It recomputed every unique authority-train by
authority-validation comparison in concat-antibody, VH, non-empty VL, and
antigen views; it did not reuse the builder's edge inventory.

## Interpretation guardrail

The original authority validation pool was previously inspected by exploratory
ESM2/ESM2 retrieval work. These corrected subsets repair direct contamination
relative to the later AntiBERTy architecture and epoch selection, but remain
reused development confirmation subsets. They are not independent, external,
temporal, prospective, or frozen test sets.
