# SaProt / ProSST structural-feature and layer-sweep protocol v1

Frozen 2026-09-30 before the first successful model feature extraction.

## Hypothesis and source constraints

The current ESM-IF1 gain may not generalise to modern structure-aware protein
language models, and the final layer may not be the best transferable residue
representation. SaProt-650M receives genuine Foldseek AA+3Di tokens; AA-only
input is forbidden. ProSST-2048 receives tokens from the official 2048-state GVP
quantizer. The official model files, code commits, tokenizers, quantizers,
structure authority and every output shard are content-hashed.

Primary sources: the official SaProt implementation states that its 35M/650M
frozen embeddings require structural tokens and documents Foldseek conversion;
the official ProSST implementation documents the 2048-state quantizer and
structure-token model input. SaProt is MIT licensed. ProSST is
CC-BY-NC-ND-4.0; repository integration is research-only and no upstream code or
weight is redistributed.

## Data firewall

Only the existing development structure authority is eligible: 3,345 unique
side/sequence units (2,174 antibody-chain units and 1,171 antigen units) selected
from train/validation predicted structures without contact labels. No closed test
structure, test label, pair contact artifact or OOD label may be loaded.
Sequences without admitted predicted structure remain missing in downstream
feature stores and are zero-filled with the existing explicit presence bit; they
are never synthesized or dropped.

## Frozen extraction grid

- SaProt-650M, hidden size 1280, hidden-state indices `{0, 8, 16, 24, 33}`.
- ProSST-2048, hidden size 768, hidden-state indices `{0, 3, 6, 9, 12}`.

Index 0 is the input embedding output and the maximum index is the final
transformer layer. Intermediate indices are approximately depth-quantile probes.
All registered layers are emitted from the same forward pass. No metric is read
during extraction and no layer is added or removed after downstream results.

Sequences longer than 1,000 residues use deterministic windows with 128-residue
overlap, uniform residue-space averaging and complete-coverage checks. Model
special tokens are removed before alignment. Outputs are float16 residue tensors
in the repository's content-addressed representation format; inference uses
bfloat16 autocast but accumulation and overlap merging use float32.

## Downstream layer sweep

Each layer becomes one generic representation source, with strict expected model
name/revision and source-specification hashes. The downstream architecture,
training split, optimizer, epoch budget, checkpoint rule and metrics are identical
across arms. Each of the ten real-feature layers is trained with seeds 1, 2 and 3.
Width-matched random-projection controls are required separately for the 1280- and
768-dimensional sources. ESM-2-only and the existing ESM-IF1 arm are reported as
historical controls, never borrowed as same-run seeds.

The layer sweep is development selection only. Primary endpoint is macro
per-complex pair AP; epitope/paratope AP, partner retrieval, outer-product margin,
training variance and complex/component-level bootstrap are secondary. A best
validation layer may guide a later protocol, but cannot be called blind, SOTA or
confirmed without a newly admitted cohort.

## Execution gates

1. CPU contract tests and source hashes.
2. One-record real-token SaProt extraction and one-record official ProSST
   quantisation/extraction on scheduled GPUs.
3. Exact sequence/row count, finite tensor, model identity and token-source checks.
4. Full token/extraction shards with no overwrite; failed attempts remain.
5. Independent coverage audit before any downstream training.
6. One-seed training smoke, then the full frozen 10-layer x 3-seed training grid.

Smoke tests, installed weights, extracted interfaces and partial shards are not
reported as completed training or performance results.
