# RePaRank structure evaluation resume protocol

Purpose: complete the official RePaRank sequence+structure evaluation on the exact existing 217-complex validation grid **without repeating verified inference** from failed job `1640868`.

## Frozen compatibility gate

An old raw pair-score NPZ may be reused only if all of the following pass:

1. its instance is one of the frozen 217 validation IDs;
2. its six-field NPZ schema is exact and the score definition is `negative_euclidean_distance`;
3. antigen residue indices, antibody residue indices, and binary labels are byte/value-identical to the current frozen contact-map candidate grid;
4. every score is finite FP32 and has exactly the frozen candidate-pair count;
5. every heavy/light/antigen sequence used by that instance exists in both the old incomplete structure authority and the new strict authority;
6. each old/new chain PDB has the manifest-declared SHA-256 and the old and new PDB SHA-256 are identical for every reused chain;
7. checkpoint, official RePaRank source commit, canonical manifest, split manifest, both structure manifests, and executor source are hash-bound before execution.

Any failed condition removes authority to reuse that NPZ; the wrapper fails closed rather than silently recomputing or mixing inputs.

## Missing-only inference

After the gate, a new output tree copies verified old NPZ bytes unchanged and runs the official sequence+structure checkpoint only for the frozen missing instance IDs. It uses the **new 217/217 strict structure authority**, the same candidate grid, the same official RePaRank checkpoint, and the same negative-Euclidean score definition as the original run. Test and OOD labels are not opened.

## Aggregation

Aggregation revalidates all 217 NPZs against the frozen candidate axes and labels, then reports the same endpoints as the existing sequence run: macro mean-per-complex pair AP, equal-component bootstrap, pooled pair AP/AUROC, epitope macro AP, and paratope macro AP. Sequence and structure results are comparable only on this same grid. Official checkpoint training-accession overlap with this validation cohort remains unknown, so the result is an exposed pretrained validation diagnostic, not a clean SOTA claim.
