# Additive correspondence research extension

**First campaign completed.** See [READOUT.md](READOUT.md) for the measured effects and limitations, and [EXECUTION.md](EXECUTION.md) for the terminal job and parity evidence.

Base: `kevin/representation-ablation-v1@55cfd137aee2b3b749f244fcb76dd11d30dda2c5`.
New work: `kevin/correspondence-upgrade-20260930`.
The original source and evidence-freeze artifacts are unchanged.

## Implemented here

- Chain-preserving perturbations, same-amino-acid controls and explicit wrap diagnostics.
- Base-feature/presence-mask invariants and known-inverse restoration checks.
- Existing-checkpoint multi-encoder GPU runner with validation-only ID firewall.
- Input/source/checkpoint byte binding and preserved failures; exclusive result creation.
- Reciprocal length-matched known-pair retrieval diagnostics with tie-aware metrics.
- Complete seed-by-complex records and strict crossed seed/cluster aggregation.
- A standalone contract test suite, separate from historical repository test failures.

## Commands

```bash
python extensions/correspondence_v2/test_core.py
python extensions/correspondence_v2/run.py freeze --plan /new/attempt/PLAN.json
python extensions/correspondence_v2/run.py run --plan /new/attempt/PLAN.json \
  --plan-sha256 EXTERNALLY_RECORDED_DIGEST --task 0 --output /new/attempt/out --smoke
# Formal submission is manual after smoke inspection. Never silently resubmit.
python extensions/correspondence_v2/aggregate.py --plan /new/attempt/PLAN.json \
  --results /new/attempt/out --output /new/attempt/ANALYSIS.json
```

## Remaining work — NOT implemented or completed by this extension

| Workstream from the research proposal | Current boundary |
|---|---|
| Modern SaProt/ProSST representation | Needs admitted model, structure-token pipeline, residue mapping and a separate training protocol. |
| Layer sweep / width-matched new encoder training | Existing features do not contain all layers; extraction and training are not launched. |
| Spatial-displacement mechanism | Needs audited coordinates aligned to each source row; row displacement is not angstrom distance. |
| Learned corruption rescue | Known inverse here is only a correctness control. Train/validation separation for a learned recovery module remains necessary. |
| Large all-vs-all partner-specificity benchmark | This extension is only a bounded matched-panel diagnostic; unknown partners are not verified negatives. |
| Fresh OOD confirmation | Frozen representation project says no unread test remains. New cohort admission/decontamination must precede any scoring. |
| RePaRank comparison | Official code/checkpoints/license and candidate semantics must be admitted before a comparable run. Not executed here. |
| PCGrad / GradNorm / objective factorial | Already in historical source and experiments; do not count reimplementation as new work. |
| External leakage audit / DataSAIL comparison | Requires new cohort and sequence/structural similarity data; no new zero-leakage claim. |
| Co-folder comparison and affinity endpoint | Separate resources/protocols/endpoints; not run and not claimed as complete. |
| Main-branch publication cleanup | No blind merge of divergent research branches and no modification of frozen artifacts. |

## Primary-source verification

Read on 2026-09-30 before this implementation:

- SaProt official source: https://github.com/westlake-repl/SaProt .
  The 35M/650M frozen-feature setup expects structure-aware tokens. Do not call
  an AA-only embedding extraction an equivalent structure-aware baseline.
- RePaRank official source: https://github.com/NaturalAntibody/reparank .
  Sequence and structure modes have different input contracts; its README
  references 4.5-A labels and separate threshold calibration. CC BY-NC 4.0.
- PCGrad original source: https://github.com/tianheyu927/PCGrad and
  https://arxiv.org/abs/2001.06782 . It is prior art, not a novel contribution here.
- https://arxiv.org/abs/2606.04228 . Overlapping frozen-representation selection
  question; not evidence that this project's correspondence mechanism is new.

These references establish implementation constraints, not a comprehensive
novelty certificate, baseline reproduction, or an independent verification of
every literature number quoted in the previous conversation.
