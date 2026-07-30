# Production OAS cached-loader recovery gate: job 1343051

This directory contains the small, immutable evidence exported from the
protocol-bound mmap-cache replay. The approximately 7.6 GB checkpoint remains
on the authenticated cluster run root and is intentionally excluded from Git.

## Result

- Slurm state: `COMPLETED`
- Elapsed time: `00:06:54`
- Exit code: `0:0`
- Exact final model equality: pass
- Exact final optimizer equality: pass
- Exact scientific trace equality: pass
- Exact terminal trainer-coordinate equality: pass
- Maximum cached read-only arrays: `1024`
- Frozen minimum soft open-file limit: `4096`
- Observed soft and hard open-file limits: `131072`
- SAbDab2 held-out access: false
- Downstream-label access: false

The uninterrupted and resumed terminal model SHA-256 is
`d3a97e0d20bdd534fb45afd64876df8466e2b0f6a1d2928c9d619d2f507187c7`.
The uninterrupted and resumed terminal optimizer SHA-256 is
`20329fb916ac11361a5b9452c496d551d1f35ffc24bc543fbea95cc0de5242ae`.
Both equal the preceding grouped-read replay, demonstrating that the descriptor
cache did not change training semantics.

## Evidence identity

- Source archive SHA-256:
  `8e4768f933758ce4bf97c5da1761e014f8376c0da1f36cfd7d6f410cbb864eeb`
- Gate protocol SHA-256:
  `01dcc33da654d119f1068057edbd28efba0532c805b9faf9f7297e504e11f621`
- Independent audit file SHA-256:
  `6dac4ab3f5a1de3f9c70abdc217cdf3913aa8db46ee48ee6e74f78547f265538`
- Independent audit receipt self-hash:
  `42ef213ad569aa8874a057613351a587787edf996d4c1b3c7a9df82854dc8b5f`
- Frozen auditor SHA-256:
  `c24a7161dbfcd10d79c272b621cd64f706f7ede54726c17d430be00bf78edb04`

The self-hash was recomputed from canonical strict-ASCII JSON after local copy.
The scientific fields in the continuous trace were also compared independently
with the concatenated stage-one and resumed traces while excluding only measured
wall-clock seconds.

## Performance boundary

The four uninterrupted updates processed 114,766 residue tokens. Data
preparation took 3.9722 seconds and GPU updates took 3.9540 seconds. Relative to
job `1343046`, preparation fell by 71.26 percent; relative to the pre-grouping
job `1343030`, it fell by 79.30 percent. Measured end-to-end preparation plus
GPU-update time fell from 23.23 to 7.93 seconds.

This gate validates the real-OAS loader, descriptor-cache optimization, and
exact recovery only. It does not authorize production pretraining, scale-out,
a biological metric, or a state-of-the-art claim.
