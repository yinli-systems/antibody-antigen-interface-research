# Triangle training-view repair and relaunch

Cutoff: 2026-08-02 CST.

## Failure

All five original `framecontact_distogram_triangle` tasks in Slurm array
`1345108` failed before the first optimizer step with:

```text
KeyError: 'epitope_unary_logits'
```

The model forward already emitted the unary logits. The memory-saving cognate
training view omitted them before calling `FrameContactObjective`, while the
localized antigen-patch router required `epitope_unary_logits`.

## Repair

The frozen v4 source was not modified. A new read-only source was derived:

```text
/ssd/scxi253/model_deployment/data/framecontact_20260730/
  residue-head-source-v5-triangle-view-fix
```

The only semantic repair was to retain `paratope_unary_logits` and
`epitope_unary_logits` in `_cognate_training_view`. No model architecture,
objective, loss weight, protocol, dataset, split, seed, or evaluation rule was
changed.

Authenticated source:

- manifest SHA-256: `f88531948427df684d877aa263c9efc0980672ea3483c39b0a95421a27c6e1da`;
- tree SHA-256: `3d26b6efe307a3b80d67399cbc6229407e9a8a97ad8b7c62afe5156b98a64fad`;
- patched trainer SHA-256: `af60226f170e9bb7d79a57ed0cf5d9a364b455094fca42d2664b4b30c935cd24`.

## Validation

- Local contract/view tests: 5 passed.
- Remote contract/model/objective tests: 31 passed in 10.01 seconds.
- Real authenticated development complex, RTX 4090, BF16:
  - forward, backward, optimizer step passed;
  - 203 parameter tensors had finite gradients;
  - distogram/contact projection max absolute error was 0;
  - frozen test was not accessed;
  - Slurm job `1347509`, receipt SHA-256
    `df705a9764f30022fee316930ab50f97ddfd17744f2630007b39a1c41a80bb68`.

An earlier repair smoke used a nonmatching pair-geometry cache and produced
non-finite gradients. It was not treated as a pass. Repeating with the exact
authenticated pair-geometry authority used by the frozen protocol reproduced
the prior v4 finite-gradient result.

## Formal relaunch

The repaired five-seed triangle array is Slurm job `1347510`.

At the first audit:

- tasks 0-2 were running beyond the original 33-second failure point;
- their error logs were empty;
- tasks 3-4 were pending because the public `gpu_4090` partition had zero free
  cards;
- moving them to `hp_4090` was rejected because the account group lacks access;
- they were not moved to RTX 5090, preserving one GPU type across the paired
  five-seed condition.

The original failure logs and temporary output directories remain recoverable
under:

```text
/ssd/scxi253/model_deployment/data/framecontact_20260730/
  failure-evidence/triangle-view-contract-1345108
```

## Evidence boundary

This establishes a repaired, authenticated, runnable training path. It does not
establish triangle-model quality, convergence, frozen-test performance, external
generalization, or benchmark improvement.
