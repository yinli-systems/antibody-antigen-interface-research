# Corrections and interpretation boundaries

This page supersedes stronger presentation wording without changing the original
experimental artifacts. Historical README versions remain in the [archive](../archive/README.md).

## Correspondence sensitivity is observed, not a universal theorem

Aligned/shuffled interventions show sensitivity to correspondence in the tested
models and cohorts. They do **not** prove that alignment is mathematically
necessary for useful transfer in every possible architecture. Same-amino-acid
shuffling preserves residue type and the feature multiset; it explicitly
**breaks** residue-to-feature correspondence. Calling it a
“correspondence-preserving control” was inaccurate.

Only ProteinMPNN's aligned-minus-shuffled contrast confirmed under the registered
217-case holdout read. ESM-IF1 did not. The holdout shares 32 cases with a separate
baseline's exposed test; the 185-case sensitivity is reported but must not
replace the registered answer. See the [complete holdout report](../results/holdout_confirmation_v1/README.md).

## Macro AP, pooled AP and component weighting are different

RePaRank's source receipt records macro pair AP `0.03780779120862989`, pooled AP
`0.008738584064940456`, and pooled AUROC `0.7860643617166425`. Its equal-component
AP estimate is `0.04322164360822419`, with interval
`[0.02633102266268606, 0.06218666145165621]`.

Earlier conversational values `0.0137` pooled AP, `0.814` AUROC, or `0.0401927`
equal-component AP did not match that receipt. They are not alternative runs and
are not used in the generated presentation. The [raw receipt](../extensions/frontier_multimodel_v1/evidence/reparank_same_protocol_v1/sequence_full/evaluation.json)
and [headline registry](../results/HEADLINE_REGISTRY.json) are authoritative.

Comparing macro AP directly with pooled contact prevalence mixes aggregation
units. A pooled prevalence calculation is descriptive; it does not supply a
matched finite-sample random-AP experiment or prove significance. AP is not
accuracy. No “times random” claim is used as a headline.

## RePaRank is a baseline, not our new model

Downloading and evaluating an official checkpoint establishes a baseline
measurement. It is not a new algorithmic gain by this project. Training-overlap
is unknown and budgets differ. Its 217-case development result is not directly
rankable against FrameContact's separate 549-case historical evaluation.

Node AP and residue-pair AP also have different prevalence and candidate spaces.
Their absolute magnitudes alone do not prove a localisation/correspondence
mechanism; controlled ablations and matched-task comparisons are needed.

## Preserve counterevidence

The [122×119 full-grid diagnostic](../extensions/frontier_multimodel_v1/SCORE_DISSOCIATION_READOUT.md)
did not robustly reproduce the earlier small-panel MRR improvement after
corruption. Descriptive length/surface/score correlations are not proof of causal
bias. Unobserved partners are not experimentally verified non-binders.

The [objective ablation](../results/objective_component_localisation_v1/PART_TWO.md)
changes the training procedure, including scheduler/selection behaviour. It does
not isolate the ranking gradient, and improved node/contact endpoints do not
establish improvement on every other task.

The old 455-case cache was invalid for the selected FrameContact checkpoint;
the [corrected 549-case evaluation](../paper/neurips_2026_framecontact/CHECKPOINT_TEST_ALIGNMENT_20260905.md)
remains an exposed historical test. The separate SPPIDER evaluation is not
retroactively invalidated merely because it also used 455 cases: checkpoint,
split and provenance must be checked per experiment.

The current 14-candidate OOD admission failed 14/14. Structural feature
extraction completion is not model-training completion; neither a canary nor a
partial layer sweep proves model superiority. [Operational evidence](../results/operations/README.md)
keeps failures and incomplete work separate from final measurements.
