# Local EDM consistency: post-v1 research candidate

## Status and evidence boundary

This is a **post-registration research candidate**, not part of the frozen
`framecontact_distogram_exploratory_v1` matrix.  It must not be used to reinterpret
or select that experiment.  The current evidence is limited to a synthetic
numerical test and a real RTX 4090 engineering smoke; there is no trained-model,
validation, test, generalization, benchmark, or novelty result yet.

## Motivation

The registered reverse-triangle penalty is necessary but weak.  In particular,
a locally constant predicted cross-distance block satisfies

`|D(i,j) - D(k,j)| <= d_intra(i,k)`

for every neighbor triangle, even when the combined antibody--antigen distances
cannot be realized by points in three-dimensional Euclidean space.  The v4 GPU
smoke already exposed this practical failure mode: the triangle branch executed
but its loss was zero on an almost-constant initial distance prediction.

## Candidate constraint

For a small, prediction-routed CDR-by-antigen-surface patch, form the complete
squared distance matrix

```text
        [ D_Ab^2     D_cross^2 ]
D^2  =  [                       ]
        [ D_cross^2T D_Ag^2    ]
```

where `D_Ab` and `D_Ag` are computed within the two monomers and `D_cross` is
predicted by the distogram head.  No cross-chain input displacement or common
input coordinate frame is used.  With `J = I - 11^T/n`, calculate

`B = -0.5 J D^2 J`.

A complete Euclidean distance matrix is realizable in 3D only if `B` is
positive semidefinite and has rank at most three.  The candidate loss therefore
penalizes:

1. negative eigenvalues of `B`; and
2. positive eigenvalues outside its largest three.

The implementation normalizes squared distances by a detached per-complex scale
to prevent scale shrinkage from becoming an escape route and to keep losses
comparable across complexes.

## Relation to prior work

- The standard EDM characterization and distance-matrix completion background
  are described by Dokmanic et al., *Euclidean Distance Matrices: Essential
  Theory, Algorithms and Applications*:
  <https://arxiv.org/abs/1502.07541>.
- Molecular distance geometry and Cayley--Menger constraints have a long history
  in reconstructing protein geometry from distance information:
  <https://doi.org/10.1016/j.ins.2020.12.072>.
- CDPred predicts inter-chain distance maps rather than only binary contacts:
  <https://www.nature.com/articles/s41467-022-34600-2>.
- DeepInter/DeepTMP and ProTact use triangle-aware interaction updates that
  couple inter-chain pairs to known intra-chain geometry:
  <https://www.nature.com/articles/s42256-023-00741-2> and
  <https://www.nature.com/articles/s42003-026-09809-2>.

The possible contribution here is narrower: a differentiable **spectral local
EDM feasibility penalty** on predicted antibody--antigen cross distances while
the input monomer frames remain independently transformable.  This is an
inference from the reviewed methods, not an established novelty claim.  A wider
paper, code, and patent search is required before describing it as novel.

## Synthetic falsification result

The authenticated RTX 4090 smoke (`Slurm 1345132`) checked two constructions:

- a true joint 3D embedding: EDM loss `0`;
- independent rigid transforms of the two input monomer frames: EDM loss `0`;
- a constant 2-by-2 cross-distance block: reverse-triangle loss `0`, EDM loss
  `0.0624997243`, gradient L1 `0.249999404`.

Receipt:

`/ssd/scxi253/model_deployment/data/framecontact_20260730/smoke/local-edm-diagnostic-v1-4090-1345132.json`

Receipt SHA256:

`48ae6c15950ebb12594db90e550371ee11386e97a7cf2c17c9ec7bc750f74942`

The result proves that the diagnostic distinguishes one concrete degeneracy;
it does not show that the regularizer improves learned contacts.

An integrated CUDA/BF16 smoke later exposed and fixed one required numerical
boundary: CUDA `eigvalsh` cannot consume BF16, so the local Gram construction
and eigendecomposition now run in differentiable FP32 with autocast disabled.
Slurm `1345879` then produced EDM loss `0.0063204966`, a nonzero gradient in the
distogram head, exact distogram/contact projection, and zero independent-
monomer SE(3) discrepancy in both the measured FP32 and BF16 forwards. Receipt
SHA256:
`bd72d1befe500ef51766482ab70ba5afee0ae632c6cbf3cdcef037262783e2e5`.
This remains synthetic engineering evidence, not a trained improvement.

## Development-only pilot required before registration

After the frozen binary/distogram/triangle matrix is complete and its selection
is sealed, run a small resource pilot on train/validation roles only:

1. measure eigendecomposition wall time, peak memory, and gradient stability for
   8/16/24 CDR residues by 16/32/48 antigen patch residues;
2. log the fraction of complexes with nonzero PSD and rank penalties at
   initialization and during training;
3. compare the existing triangle condition to `triangle + local EDM` using the
   same five seeds and selected-checkpoint rule;
4. require higher contact mean-per-complex AUPRC, P@Lmin/5, fixed-threshold
   precision, and fewer isolated false-positive components, while retaining at
   least 95% recall and not degrading reciprocal partner margin by more than
   0.01;
5. keep the current SAbDab2 test and AsEP official labels closed during this
   development comparison.

If the penalty is nearly always zero, numerically unstable, or too expensive,
do not promote it.  If it passes, freeze a new protocol and source manifest
before any new external opening.
