# Rescue qualification and fixed score-correction experiments

**Completed:** full 217-complex, 9.64M-pair score-correction diagnostic; real-feature
permutation-equivariance and missingness checks; 48 contract tests.
**Not completed:** trained rescue contact qualification across all three seeds.
The first full-seed evaluation is submitted separately, not a training rerun.

Read [READOUT.md](READOUT.md), [PROTOCOL.md](PROTOCOL.md), and
[RESCORE_PROTOCOL.md](RESCORE_PROTOCOL.md). Research and the architectural proof
are in [RESEARCH.md](RESEARCH.md). Original source, failure records, jobs and
frozen protocols were not overwritten. This extension is on its own branch.

## Code

`qualification_core.py`: exact mask-preserving transforms, content-canonical
assignment, checkpoint selection verification and complete result denominators.
`real_probe.py`: validates the frozen matcher's permutation algebra against
actual stored features without training or contact labels.
`rescore.py`: fixed five-arm transformation of already computed pair scores;
no new model inference or GPU allocation.
`qualified_evaluate.py`: prepare a fully hash-bound supplemental evaluation after
all existing rescue models finish, evaluate a seed under a scheduler allocation,
then require all three seeds for paired/component aggregation. It does not submit
jobs and cannot choose a model using contact AP.

## Tests

```bash
python -m unittest discover -v -s extensions/rescue_qualification_v2 -p 'test_*.py'
```

CPU dependencies for these tests: numpy, scipy, torch. Actual evaluation also
uses the original repository's pandas, sklearn and feature/model dependencies.
The committed source was tested on ParaCloud; results are in
`evidence/20260930/evidence_attempt3/UNIT_TESTS.txt`.

## Reproduction and no-duplication rule

The exact original commands, source/input hashes and result records are kept in
`evidence/20260930/`. Do not reuse an existing output path. To reproduce the fixed
rescoring, use the executable source snapshot bound by RESCORE_PLAN.json and a
new output location. Rebuilding a plan requires the same candidate files and
must be labelled a new run; do not relabel the original manifest.

For trained rescue evaluation, the already prepared plan is:
`/ssd/scxi253/frontier_multimodel_v1/frontier_performance_v1/rescue_qualification_v2_20260930/evidence_attempt3/EVALUATION_PLAN.json`.
Do not submit another seed101 job: `1641906` already owns that evaluation. Only
consider seeds202/303 after inspecting seed101's terminal receipt. No automatic
follow-on submission or background assistant monitoring is configured here.

All outcomes are development diagnostics. No new OOD, affinity, co-folder
benchmark, blind validation, contact-accuracy gain or SOTA claim is established.
