# PAIRScale Development-Only Nonconfirmatory Protocol V1

## Status and purpose

This protocol creates a narrow development authority for train-only engineering
while the confirmatory Gate 0 remains blocked. It does not modify, replace, or
weaken the confirmatory Gate 0 policy. Artifacts produced under this protocol
cannot be promoted into the confirmatory evidence chain.

The only authorized permission is `development_only_nonconfirmatory`. It may be
used for:

- `overfit_sanity`;
- `architecture_ablation`;
- `loss_debugging`;
- `systems_profiling`; and
- `baseline_development`.

## Frozen data boundary

Every authorization must bind all of the following externally anchored
train-only artifacts:

- the authority-train manifest;
- the engineering-cache authorization manifest;
- the engineering-cache manifest; and
- the exact membership digest of the 1,830 authority-train examples.

Before a decision is issued, the controller must independently supply the
complete authority-train identifier set. Its count and membership digest must
equal the frozen policy. The requested subset must be nonempty, contain no
duplicates, and be an exact subset of that authority set. The subset manifest
is separately content-addressed so that changing ordering, metadata, or
membership invalidates the authorization.

Only the literal source split `train` is accepted. Validation identifiers,
validation sequences, validation labels, frozen or external test data, test
predictions, and test-derived thresholds have no authorized input channel.

## Forbidden uses

The following remain forbidden:

- confirmatory or paper-result claims;
- validation- or test-driven model or checkpoint selection;
- outer-fold evaluation or external-test inference;
- checkpoint promotion into a confirmatory run;
- claims of biological generalization, state of the art, or world leadership;
- use of a development receipt to satisfy any confirmatory Gate 0 receipt; and
- re-anchoring a changed policy, request, subset, authority, cache, or source
  artifact from inside the worker process.

Development metrics may be recorded only as debugging evidence on an explicitly
train-only subset. They are not estimates of held-out performance.

## Trust and launch boundary

The policy digest and the expected decision digest must reach the launcher from
an immutable control plane outside the worker output directory. A digest is a
provenance binding, not a digital signature or proof of organizational
independence. A self-authored policy is not made trustworthy merely by hashing
it.

The controller recomputes the decision from the complete authority membership.
The launcher then verifies all of these values before data loading:

- expected policy digest;
- expected decision digest;
- exact purpose;
- exact run identifier;
- exact subset-manifest digest; and
- the `development_only_nonconfirmatory` permission.

Any disagreement, malformed input, unknown purpose, unknown identifier,
duplicate identifier, non-ASCII ambiguity, non-train split, forbidden access
flag, confirmatory request, or missing external digest blocks authorization.

## Claim boundary

The canonical claim boundary is:

> Authority-train-only development evidence for debugging and systems
> engineering; no validation, test, model-selection, confirmatory, biological
> generalization, SOTA, or world-leading claim.
