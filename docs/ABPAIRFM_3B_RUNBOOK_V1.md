# ABPairFM 3B runbook v1

**Status:** Gate 0 in progress; public train/validation materialization only.

## Research decision

ABPairFM starts from the public `esm2_t36_3B_UR50D` checkpoint instead of
training a 3B encoder from random initialization. Stage A then changes the
attention semantics to PPLM-style hybrid attention: chain-local RoPE for
within-chain logits and position-free logits with learned per-head scales plus
the published fixed +1 bias for cross-chain attention. The training kernel uses
an exact chain-grouped lifted SDPA formulation, while the explicit dense matrix
path is reserved for numerical audits and frozen prediction export. Stage B
specializes the same model on barcode-linked native VH/VL pairs with the AbCDR
80% CDR-focused / 20% whole-chain curriculum.

This is deliberately stronger than concatenating two independently encoded
chains. It is also a continued-pretraining claim, not a claim that all 3B
parameters were learned from the project data.

## Scale accounting

Three quantities must remain separate in every receipt:

1. unique biological interactions or native cell-linked VH/VL pairs;
2. deterministic training examples after structural-view augmentation;
3. non-padding residue tokens actually consumed by optimizer updates.

Stage A requires 2.5M--25M post-dedup pair examples and consumes 5B residue
tokens. Stage B requires at least 5.5M admitted native VH/VL pairs and consumes
2B residue tokens. The campaign therefore has a frozen minimum of 7B useful
pair residues. Repeated exposure is reported and never relabeled as unique
data.

## Current evidence and blockers

- PPLM reports more than 3.3M sequence pairs and provides a 321,794,724-byte
  training-data archive. The paper says the code is MIT, while the current
  GitHub page says PolyForm Noncommercial. ABPairFM therefore copies no PPLM
  code: it implements Eq. 1 clean-room from the CC BY 4.0 paper and uses only
  the MIT ESM implementation for checkpoint-compatible naming and numerics.
  Dataset training admission remains blocked until its own license receipt is
  resolved. The archive also omits per-edge STRING evidence scores and database
  version, so it is now a reproduction/reference source rather than a required
  production-training source. The published PPLM model is 650M; ABPairFM tests
  whether the same pair-aware mechanism transfers at 3B.
- PINDER 2024-02 reports 2,456,152 train examples and up to 25,130,994
  structural-view combinations. Only index and metadata are being materialized
  initially; `test_set_pdbs.zip` is explicitly excluded.
- PPIRef is already authenticated on ParaCloud and contains 745,501 PDB files,
  but remains unnormalized and not admitted for training.
- Production sequence-only pairs will come from the versioned STRING v12
  physical-links release under CC BY 4.0, after an explicit high-confidence
  evidence-channel filter. Generic functional associations and score-free
  pairs are not admitted.
- The public AbCDR set contributes more than 1.6M processed native pairs. It is
  a high-quality starting source, not evidence that the 5.5M gate has passed.
- The official ESM2-3B checkpoint is not yet authenticated locally.
- The fixed runtime has PyTorch 2.7.1, CUDA 12.8, SDPA, and FSDP2. It does not
  contain Transformers, fair-esm, FlashAttention, safetensors, or PINDER, and
  no packages will be installed into it.

The public PINDER index contains split metadata for every role. Reading that
public index is reported separately from using test rows: normalization may
retain only train/validation rows, and no test structure archive or test row is
authorized for training, model selection, or analysis.

Warm-start leakage is a separate scientific limitation. ESM2 was pretrained on
UniRef50, so even perfect decontamination of the new Stage A/B corpora cannot
prove that the backbone never saw homologues of a development target. Every
result must therefore report both the audited new-corpus overlap and the
unverifiable historical-backbone exposure boundary; it may not be called a
strict from-scratch OOD result.

## Execution order

1. Finish create-only source downloads and SHA-256 inventories.
2. Inspect PPLM, PINDER, PPIRef, and AbCDR schemas; count valid records, useful
   residues, duplicates, clusters, studies, donors, and clone components.
3. Implement an ESM2-compatible 3B loader and blockwise hybrid attention using
   only the fixed PyTorch runtime. Verify exact base-weight loading and the
   source-derived parameter receipt on CPU/meta before allocating GPUs.
4. Run a four-`gpu_4090` FSDP2 capacity/correctness smoke with exact same-world
   recovery. No `--mem` argument is permitted.
5. Run a fixed-work four-to-eight GPU benchmark. Production requires at least
   0.70 strong-scaling efficiency, p95/median step time at most 1.20, peak
   memory below 92%, and no sample omission or duplication.
6. Train Stage A and audit its checkpoint before Stage B. Train Stage B only
   after 5.5M native pairs pass provenance, deduplication, CDR numbering, and
   study/donor/clone-component isolation.
7. Evaluate only on frozen SAbDab2/AsEP development roles. Contact, retrieval,
   matched-hard partner swap, calibration, and OOD outputs are frozen. Test
   remains sealed until a development report is accepted and explicit approval
   is received.

## Public comparison targets

- ESM2-3B base and AbCDR-ESM2 are sequence-representation controls.
- PPLM and PPLM-Contact are pair-aware PPI/contact controls, with published
  metrics treated as author-reported unless rerun on an identical split.
- MIPE, EpiFormer, SPPIDER-seq, and FrameContact remain the project-specific
  SAbDab2/AsEP controls under their existing frozen protocols.

No cross-paper number is called a win. Promotion requires identical examples,
splits, candidate sets, label mapping, and bootstrap units.
