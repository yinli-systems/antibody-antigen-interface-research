# Recovered frontier programme state — 2026-09-30

This branch resumes from `14a385feaec24b3141331963ba71dda788a164e1`.
No second correspondence-v2 run was launched.

## Verified materialized assets

- Generic frozen-representation integration: 49 focused tests passed.
- Development-only predicted structure authority: 3,345 sequence-indexed units
  (2,174 antibody chains; 1,171 antigen chains), exact N/CA/C/O coordinates,
  no contact-label or test access. There are 177 repeated sequences with more
  than one predicted geometry; selection is confidence then byte-hash, not labels.
- SaProt: 3,345 real Foldseek 3Di token records completed and receipt-bound.
  AA-only input is not used as a structural substitute.
- SaProt-650M and ProSST-2048 official model files are pinned in
  `MODEL_RUNTIME_AUTHORITY.json`; ProSST quantisation/extraction is not complete.
- OOD metadata freeze: 14 post-cutoff PDB candidates. The pre-frozen sequence
  audit admitted 0/14. No coordinate/contact/performance read occurred, and the
  gate is not relaxed after seeing this outcome.

## Full-candidate retrieval recovery

| Job | Terminal state | Meaning |
|---|---|---|
| 1639898 | FAILED 1:0 | empty hash shard; preserved failure |
| 1639900 | FAILED 1:0 | checkpoint/feature-provenance mismatch; preserved failure |
| 1639912 | COMPLETED 0:0 | one-anchor, 217-instance-candidate engineering smoke |

Job 1639912 is not a large benchmark: the candidate list repeats exact antibody
and antigen sequences and contains only one anchor. Validation actually has 122
unique antibodies and 119 unique antigens, with up to six known positives per
anchor. The score-dissociation campaign therefore freezes a unique-sequence
122 x 119 Cartesian product with multi-positive metrics before full execution.

## Not completed at recovery

No SaProt/ProSST frozen-feature extraction or downstream training, layer sweep,
learned realignment, complete unique-sequence all-vs-all result, admitted untouched
OOD measurement, RePaRank replay, co-folder comparison, or affinity experiment.
The presence of code, weights, tokens, cloned repositories or smoke tests is not
reported as completion.
