# FrameContact follow-up status (2026-08-16)

## Evidence boundary

This note is a development-only handoff. No frozen, official, or external test
prediction or score has been loaded, used for checkpoint selection, or used for
architecture selection. The one-time official test remains sealed.

## Implemented and pushed

- `a77fad0` registers the six-condition surface-ratio joint-coupling matrix and
  the three-condition epitope-group pair-to-node attribution matrix.
- `8d81076` adds authenticated attribution aggregation, case-study aggregation,
  ratio aggregation wrappers, and a fail-closed receipt for distance-bin,
  distogram, and 10 Å neighborhood auxiliaries.
- Local validation: `11 passed` across the follow-up protocol, aggregator, and
  existing joint-coupling tests.
- The attribution aggregate reports adjacent pair-to-node reductions only; it
  does not interpret epitope/paratope labels as independent biological tasks.

## Live development jobs

| job | matrix | current state | output root |
|---:|---|---|---|
| 1412482 | surface-ratio joint coupling, 6 conditions × 5 seeds | 30/30 running at last check; no result receipt yet | `.../asep/development-runs/joint-coupling-ratio-v1` |
| 1412514 | epitope-group aggregation attribution, 3 conditions × 5 seeds | 15/15 running at last check; no result receipt yet | `.../asep/development-runs/joint-attribution-dev-v1` |

The first ratio submission, 1412448, is not scientific evidence: all members
failed before training because relative Slurm log paths resolved under an
unavailable home directory. The replacement 1412482 uses absolute log paths and
has clean `RUNNING`/exit-code-0 scheduler evidence.

## Pending execution

1. Wait for 1412482 and 1412514 to produce authenticated result/checkpoint
   receipts.
2. Submit the ratio aggregate and the attribution aggregate using the pinned
   wrappers.
3. Submit the 15-run same-antigen case-study array after scheduler association
   capacity is available; aggregate it by condition and seed, without changing
   any checkpoint.
4. Only after all development artifacts are reviewed should the architecture,
   threshold/calibration, checkpoint rule, and baseline manifest be frozen for a
   possible single official test.

## Intentionally not run

The current authenticated AsEP cache has native contact labels but no
authenticated pair-distance or 10 Å neighborhood-label authority. Therefore no
distance-bin/distogram or 10 Å auxiliary is fit or reported. The explicit
fail-closed receipt is
`framecontact/protocols/FRAMECONTACT_ASEP_DISTANCE_AUXILIARY_BLOCKED_V1_RECEIPT.json`.

