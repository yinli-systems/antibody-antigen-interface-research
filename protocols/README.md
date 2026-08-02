# Protocol index

The `protocols/` tree is an append-only research record. Superseded files remain
present so that historical claims, failures, and solver attempts can be audited.

## Current partner-geometry path

- [`SABDAB2_PARTNER_GEOMETRY_R2_PROTOCOL_V1.md`](SABDAB2_PARTNER_GEOMETRY_R2_PROTOCOL_V1.md)
  defines the partner-aware native residue-pair task and execution lattice.
- [`sabdab2_partner_geometry_r2_v1.json`](sabdab2_partner_geometry_r2_v1.json)
  freezes the machine-readable scientific protocol.
- [`sabdab2_partner_geometry_r2_model_contract_v1.json`](sabdab2_partner_geometry_r2_model_contract_v1.json)
  freezes the model and tensor contract.
- [`SABDAB2_GEOMETRY_CONFIRMATORY_PROTOCOL_V1.md`](SABDAB2_GEOMETRY_CONFIRMATORY_PROTOCOL_V1.md)
  governs the earlier intra-chain geometry confirmation study.

## Exact score-free split path

- [`V6_A_SCORE_FREE_EXACT_SOLVER_PROTOCOL.md`](V6_A_SCORE_FREE_EXACT_SOLVER_PROTOCOL.md)
  defines the exact score-free authority, joint formulation, and fail-closed
  solver requirements.
- [`V6_B_LOGIC_BASED_DECOMPOSITION_PROTOCOL.md`](V6_B_LOGIC_BASED_DECOMPOSITION_PROTOCOL.md)
  records the production feasibility and decomposition boundary.
- [`V6_C_LINEAR_INFEASIBILITY_DIAGNOSIS_PROTOCOL.md`](V6_C_LINEAR_INFEASIBILITY_DIAGNOSIS_PROTOCOL.md)
  defines the constraint-family diagnosis that may follow independently certified
  linear-projection infeasibility.

The versioned JSON registrations, submission records, compatibility evidence, and
attempt receipts adjacent to these documents are part of the evidence chain. A
later filename does not silently invalidate an earlier attempt.

## Partner-retrieval history

The `PARTNER_RETRIEVAL_PROTOCOL_V1*` sequence contains the original protocol and
its amendments through the execution lock. The `PAIRFORMER_*`, `PAIRSCALE_*`, and
`TRAIN_ONLY_*` documents record subsequent development-only gates and split work.
Negative or blocked results are intentionally retained.

## Structured protocol evidence

Subdirectories such as `identity80_development/`,
`authority_validation_nw_boundaries/`, and `train_only_s2_*` contain large
machine-readable artifacts. Some are mirrored under `results/` to preserve their
original publication and result lineages. Do not remove or consolidate these files
without updating every bound manifest and verifier.

## Change policy

1. Do not edit an evidence-bearing protocol after execution.
2. Add an amendment, new version, or new attempt instead.
3. State exactly which earlier artifact is superseded and which evidence remains
   valid.
4. Keep timeout, infeasibility, and disagreement evidence.
5. Never loosen a threshold inside an existing protocol.
