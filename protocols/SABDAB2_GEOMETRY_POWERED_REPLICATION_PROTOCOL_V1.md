# SAbDab2 intra-chain geometry: powered replication protocol v1

Status: prospective preregistration. Development-only. This protocol authorizes
one replication of an existing development comparison. It does not authorize
contact training, model selection, outer-fold inference, frozen-test access, or
any benchmark-improvement claim.

Date: 2026-09-08

## Relationship to earlier records

This is a new numbered protocol. It supersedes nothing, reinterprets nothing,
and leaves every earlier attempt, receipt, and gate outcome intact. The strict
gate recorded in `SABDAB2_GEOMETRY_CONFIRMATORY_PROTOCOL_V1.md` remains failed.

It exists because the design analysis in
[`GEOMETRY_GATE_POWER_ANALYSIS_V1.md`](../results/sabdab2_structure_self_attention/GEOMETRY_GATE_POWER_ANALYSIS_V1.md)
showed the earlier gate could not have decided the question it was asked:

- three unanimous replicates cannot clear a `P(A>B) > 0.5` criterion at
  `alpha = 0.05`, because the exact bound for `k = n` is `alpha ** (1 / n)`,
  which first exceeds 0.5 at `n = 5`;
- power on the joint endpoint was 0.357 at three replicates and would be 0.640
  at the five later preregistered.

## The pilot cannot size this run

The three archived A100 seeds give a joint effect size of `d = 1.088`. Sizing on
that number would be unsound: its one-sided 95% lower confidence limit is
**`-0.254`**. Three replicates are consistent with a true effect that is zero or
negative, so the pilot supports no sample-size claim at all, and a protocol that
sized on the point estimate would be assuming its own conclusion.

The design is therefore sized by cost rather than by the pilot. One run of all
modes takes about 54 seconds on the recorded hardware, so a large replicate
count is affordable and no tradeoff needs to be made.

**Replicate count: 24 seeds.** At the pilot between-seed standard deviation this
detects `d >= 0.52`, a raw joint macro AUPRC difference of about `0.0044`, at
power 0.80 and `alpha = 0.05` one-sided — roughly half the pilot's point
estimate. The pilot standard deviation is itself uncertain at `n = 3`; 24
replicates are chosen to leave margin for that, not because the pilot justifies
exactly 24.

## Smallest effect of scientific interest

`+0.005` joint macro AUPRC, absolute, against the topology-only reference. This
is declared before the run and is not derived from the observed effect. A
difference smaller than this is reported but is not treated as support for the
geometry encoding, regardless of its p-value.

## Frozen design

| Item | Value |
|---|---|
| Entry point | `scripts/train_sabdab2_structure_ablation.py` |
| Primary comparison | `distance_direction` minus `topology_only` |
| Primary endpoint | per-complex joint macro AUPRC |
| Secondary endpoints | per-complex paratope AUPRC, per-complex epitope AUPRC |
| Descriptive modes | `distance`, `full_pose` |
| Replicates | seeds `1 .. 24`, one Slurm array task each |
| Epochs | 3, fixed final epoch, no checkpoint selection |
| Cohort | the authenticated structure cache; 2,676 train and 36 validation complexes |
| Resampling unit | the 16 exact antibody-antigen endpoint-connected components |
| Precision | bf16, matching the recorded pilot |

The seed sets `random`, `numpy`, `torch`, and the CUDA generators, so it varies
initialisation and data order together. The data split is fixed by the
authenticated authority and is **not** resampled. Randomising the split as well
would give a better estimator at equal cost, and not doing so is a known
limitation of this design rather than an oversight.

## Multiplicity

The primary endpoint is the joint macro AUPRC and is the **only** primary
hypothesis, so no multiplicity adjustment applies to it.

The two secondary endpoints form their own family of two, adjusted by Holm
step-down. The composite is excluded from that family: joint macro AUPRC is the
arithmetic mean of the two components to a residual of `5.55e-17`, so including
it would charge for a hypothesis that cannot vary independently of the others.

Any adjustment that is reported must be the adjustment that was computed. The
earlier receipts recorded `3 * p` under a field named for Holm; this protocol
requires the field name and the method to agree, and requires the unadjusted
values to be recorded alongside.

## Decision rule

Both conditions must hold on the primary endpoint:

1. **Significant.** The exact Clopper-Pearson one-sided 95% lower bound on the
   fraction of seeds with a positive difference exceeds `0.5`.
2. **Meaningful.** The lower end of the paired 95% bootstrap interval for the
   mean difference exceeds the smallest effect of scientific interest, `+0.005`.

The estimate, its interval, the seed-level win fraction and its exact bound are
reported whether or not either condition holds. A null result is a result; it is
recorded with the same completeness as a positive one.

## Preflight gates

The run does not start unless all of the following pass, and each writes a
receipt:

1. `python3 scripts/verify_partner_pose_contract.py` — the model consumes no
   bound inter-chain geometry. A rigid motion of either chain must leave every
   output unchanged, and the probe must report a live sensitivity rather than an
   inconclusive verdict.
2. `python3 scripts/verify_protocol_source_bindings.py` — no protocol-governed
   source file has unregistered bytes beyond the recorded baseline.
3. The structure-cache manifest digest matches
   `2580e41ced744ade8a16fa94ae4b9b22bda5e85b7477a71ae1344ed732bf3e6d`.

## Stop rules

- A failed preflight stops the run. It is not overridden.
- No frozen or outer test is read. The 416 native-release test cases stay
  excluded before any mmCIF member is opened.
- No checkpoint selection. Epoch 3 is reported for every mode and seed.
- Seeds are fixed in advance and run to completion. Seeds are not added,
  dropped, or reordered after any result is seen.
- The thresholds, the smallest effect of scientific interest, and the decision
  rule above may not be changed after the first result is observed. A change
  requires a new numbered protocol with an explicit supersession link.

## Claim boundary

Whatever the outcome, this run is development evidence about an intra-chain
geometry encoding on a marginal paratope/epitope task with a 271,670-parameter
model, three epochs, and 36 validation complexes forming 16 independent
components. It is not partner-specific evidence, not a native residue-pair
result, not a frozen-test result, and not a benchmark-improvement claim. It does
not unblock Gate 0 and carries no authority over the R2 workstream.
