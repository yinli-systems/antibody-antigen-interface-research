# Production OAS recovery gate: job 1343046

This directory contains the small, immutable evidence exported from the
newly frozen one-GPU production-corpus recovery replay. The approximately 7.6
GB checkpoint remains on the authenticated cluster run root and is intentionally
excluded from Git.

## Result

- Slurm state: `COMPLETED`
- Elapsed time: `00:07:19`
- Exit code: `0:0`
- Exact final model equality: pass
- Exact final optimizer equality: pass
- Exact scientific trace equality: pass
- Exact terminal trainer-coordinate equality: pass
- SAbDab2 held-out access: false
- Downstream-label access: false

The uninterrupted and resumed terminal model SHA-256 is
`d3a97e0d20bdd534fb45afd64876df8466e2b0f6a1d2928c9d619d2f507187c7`.
The uninterrupted and resumed terminal optimizer SHA-256 is
`20329fb916ac11361a5b9452c496d551d1f35ffc24bc543fbea95cc0de5242ae`.

## Evidence identity

- Gate protocol SHA-256:
  `e093ad7a90e4dc0210d15308428210a221c666c579edd7708ee2756552f96752`
- Independent audit file SHA-256:
  `89dd12ca7999de230642d4e7f0b6c0d0ee168286bdccc56236a089f7ac79df4b`
- Independent audit receipt self-hash:
  `de7914b594dff66031ebd26f60a39ba747afa49b5da91d39816eb1e8b52b844e`
- Frozen auditor SHA-256:
  `3564e7d6fac56fa333c61367432ab16bc842afb45911c3c22b14af780a5f1539`

The audit self-hash was recomputed from canonical strict-ASCII JSON after the
artifact was copied locally. The scientific fields in the continuous trace
were also compared independently with the concatenated stage-one and resumed
traces while excluding only measured wall-clock seconds.

## Performance boundary

The four uninterrupted updates processed 114,766 residue tokens. Data
preparation took 13.8222 seconds and GPU updates took 4.0458 seconds. The
grouped update read was 27.96 percent faster in data preparation than job
`1343030`, but the data path remains slower than compute. This gate validates
loader correctness and exact recovery only; it does not authorize production
pretraining, scale-out, a biological metric, or a state-of-the-art claim.
