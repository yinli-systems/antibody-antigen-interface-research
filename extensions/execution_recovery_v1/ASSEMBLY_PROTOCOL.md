# Post-prediction structure authority assembly

This assembler is a new downstream stage, not a modification of the live AF3
plan. All 40 independently complete task receipts, all sample hashes and exact
selected models must verify first. Incomplete campaigns produce no output tree.
The existing base manifest and each base PDB are hash-checked and left untouched.
The completion sequence set must equal the exact missing keys induced by the
frozen 217-case validation metadata, not a selected or approximate subset.

Selected CIF models become N/CA/C/O-only chain-local PDBs. Sequence, atom set,
chain identity and roundtrip coordinate precision (<=0.0006 Angstrom componentwise
for PDB's three decimals) are verified. The separate combined manifest must cover
all 217 validation complexes. Runtime bytes are fully rehashed before COMPLETE.
No current SaProt/ProSST feature artifact/source specification is modified.

This creates inputs for a later, newly pinned RePaRank structure evaluation. It
is not that evaluation, a co-folding benchmark, or an accuracy result. Even a
complete authority does not establish training-overlap freedom or OOD validity.
Until all 40 AF3 tasks finish, assembly is implementation + tested admission
logic only; do not call it an assembled structural dataset.
