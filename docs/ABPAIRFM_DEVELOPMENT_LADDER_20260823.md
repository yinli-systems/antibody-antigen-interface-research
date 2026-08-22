# ABPairFM development ladder — 2026-08-23

## Claim boundary

All measurements in this note use frozen development data. Test access remains
closed. Retrieval scores are uncalibrated ranking scores; partner swaps are hard
ranking controls, not measured biological noninteractions. No SOTA claim is
authorized by these results.

## Completed head-free retrieval controls

The 250M-token Stage-A0 checkpoint, base ESM2-3B, base ESM2-650M, MINT-650M,
and PPLM-650M were evaluated on identical AsEP and SAbDab2 development panels.
The paired-bootstrap unit is the canonical exact pair (10,000 replicates).

| Model | AsEP MRR | AsEP R@1 | SAbDab2 MRR | SAbDab2 R@1 |
|---|---:|---:|---:|---:|
| Stage-A0 3B, 252.6M tokens | 0.1270 | 0.0265 | 0.1789 | 0.0551 |
| Base ESM2-3B | 0.1322 | 0.0265 | 0.1689 | 0.0460 |
| Base ESM2-650M | 0.1342 | 0.0412 | 0.1711 | 0.0454 |
| MINT-650M | 0.1370 | 0.0412 | 0.2136 | 0.0914 |
| PPLM-650M | 0.1242 | 0.0294 | 0.2129 | 0.0908 |

At 250M tokens, Stage-A0 is not a credible retrieval improvement. On SAbDab2,
Stage-A0 minus base ESM2-650M is +0.0077 MRR (95% CI -0.0154 to +0.0311)
and +0.0097 R@1 (95% CI -0.0121 to +0.0315). Stage-A0 remains below MINT
and PPLM on SAbDab2; both paired confidence intervals exclude zero for MRR and
R@1. Base ESM2-3B also does not credibly improve over base ESM2-650M on this
probe, so raw 3B scale alone is not the supported explanation.

Jobs: Stage-A0 250M `1460681`, base ESM2-3B `1460682`, base ESM2-650M
smoke `1460744`, and base ESM2-650M full `1460746`.

## Authenticated 650M scale control

The official ESM2-650M safetensors checkpoint was converted to the ABPairFM
checkpoint schema. The first conversion omitted rotary `inv_freq`; that failed
parity and remains preserved as failure evidence. The corrected conversion
loaded 569 compatible state entries; only the 33 pair-aware `cross_scale`
tensors were initialized. Single-chain parity passed with cosine similarity
0.9999999404, mean absolute logit error 0.0005880, and maximum absolute logit
error 0.0086908 (job `1460724`). This gate establishes conversion parity only.

## Stage-A1 explicit multitask pilot

The frozen v3 source admission contains 22,400 train-only records: 12,800
unique PPIRef structural pairs and 9,600 STRING pairs. Exact cross-source pair
overlap is zero. The manifest contains 1,175,392 source positive contacts.
Before GPU launch, all 3,200 microbatches were materialized on CPU, yielding
965,025 sampled positive contacts, 3,800,199 sampled native noncontacts, and
7,506,328 residue tokens without a batch failure.

The first launch (`1460743`) failed before training because a meta-initialized
FSDP2 model cannot directly expose an empty model state for DCP-only restore.
The failure was not hidden. The retry first authenticates/materializes the
official ESM2-3B checkpoint and then restores the immutable 252.6M-token
Stage-A0 DCP, matching the already verified transfer-probe restore path.

The retry (`1460747`) completed all 200 train-only updates and returned
`PASS_STAGE_A1_MULTITASK_PILOT_CHECKPOINT_VERIFIED`. The immutable DCP receipt
and collective restore both match content SHA-256
`ea00f5113c64f8edf1e17122e7921c2e4bffec328d07a02afa8d321cf4d6ffd7`.
Mean total loss decreased from 3.3982 over the first 20 updates to 3.1001 over
the final 20; MLM decreased from 2.6827 to 2.5528, and sparse contact BCE from
1.1020 to 1.0921. The pilot processed 7,506,328 residue tokens at 3,312
residue tokens/s with finite gradients and no test access.

This is an objective-health pass, not a production-promotion result. Local
retrieval loss collapsed from 0.8224 to 0.0061, showing that the in-batch
negatives became too easy to support a strong interaction-learning claim.
Allocated memory peaked at 91.7337% against the frozen 92% limit, leaving only
0.266 percentage points of gate headroom. Production Stage-A1 therefore remains
closed pending cross-rank, cluster-aware hard negatives and a lower-memory
configuration; batch size must not be increased on the current setup.

## Automated next gates

The Stage-A0 successor watchdog now submits the first immutable passing
checkpoint in each frozen window automatically:

- 500M target: first checkpoint in [500M, 520M] residue tokens;
- 1B target: first checkpoint in [1.0B, 1.02B] residue tokens.

Each probe reuses the frozen development panel and creates a submission receipt.
It does not access test data. Contact AUPRC, Brier/ECE, OOD, temporal, and
partner-swap task-level transfer remain separate development gates; head-free
retrieval results do not stand in for those measurements.
