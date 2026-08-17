# Frontier update — 2026-08-17

## New primary sources

- [AAMFM](https://arxiv.org/abs/2607.20057) proposes an antigen-specific antibody multimodal foundation model. Its differentiator is a cross-modal adapter that injects antigen geometry and epitope annotations into a shared antibody–antigen latent space. This is a relevant future adapter/downstream design reference, not evidence for changing the current OAS pretraining corpus or scale gate. The source is an arXiv preprint; no implementation, checkpoint, or license was independently verified in this audit.
- [EpiBench](https://arxiv.org/abs/2608.06022) combines structural antibody–antigen contacts with functional B-cell assays and deep-mutational-scanning escape measurements across five connected epitope tasks. It is a candidate post-pretraining transfer/evaluation panel, not a replacement for the frozen AsEP development protocol. The source is an arXiv preprint; dataset access, code, and licensing were not independently verified in this audit.
- [LF3DRoPE](https://arxiv.org/abs/2608.01092) remains the current geometry-heavy AsEP epitope-only comparison and explicitly tests rigid-transform invariance. [EpiFormer](https://arxiv.org/abs/2606.04154) remains the early interleaved cross-attention comparison. Neither source changes the current decontaminated OAS recovery gate.

## Decision boundary

The current scale ladder remains unchanged: complete the four-RTX-5090 recovery, then shape/throughput/tail-latency/telemetry, then the 30B-residue-token campaign. Do not add AAMFM/EpiBench labels to OAS pretraining, do not use them for checkpoint selection, and do not substitute a 4090 allocation for the frozen 5090 recovery protocol. Revisit them only after the pretraining campaign has a sealed receipt and an explicitly registered downstream protocol.
