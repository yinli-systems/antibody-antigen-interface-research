# MSA Pairformer baseline audit

Status: source/availability audit only. No checkpoint has been loaded and no
FrameContact train, validation, frozen-test or external example has been scored.

## Verified upstream facts

- Paper: *Expanding the scope of protein language modeling to protein-protein
  interactions with MSA Pairformer*, Cell 2026, DOI
  `10.1016/j.cell.2026.06.029`.
- Official repository: <https://github.com/yoakiyama/MSA_Pairformer>.
- Audited upstream `main` HEAD on 2026-08-01:
  `875363570df1ae484cf725aba382790444223005`.
- Official package version in that revision: `1.0.2`, Python `>=3.10`, Torch
  `>=2.5.0`.
- The repository contains a modified MIT “Coffee edition” license that permits
  use, modification and redistribution while retaining its notice.
- The official Hugging Face repository exposes a 223 MB model file, a second
  223 MB cuEquivariance model file, and small contact heads. The weight files
  are PyTorch pickle-based `.bin` artifacts, not safetensors.
- The official example uses up to MSA depth 512, an explicit complex chain-break
  index, BF16 inference and optional HHsuite filtering. It returns final MSA and
  pair representations plus C-beta/contact predictions.

## Scientific comparability boundary

MSA Pairformer is not a drop-in replacement for frozen ESM residue embeddings.
It receives an MSA and therefore has a richer information channel and a separate
preprocessing budget. Its Cell interface-contact results are not automatically
evidence for antibody-specific epitope localization, same-antigen partner
retrieval, antigen-fold generalization or the current contact definition.

Antibody--antigen complexes also lack the simple ortholog-pairing assumptions
available for many conserved protein complexes. Query-only inference is always
defined but may discard the method's main advantage; independently generated
MSAs do not create inter-chain coevolution; and paired homolog MSAs require a
separately justified pairing procedure. These three modes must never be merged
under one baseline name.

## Safe execution gate

Before any benchmark run:

1. pin the exact source revision, package lock and weight SHA256;
2. download in a networked acquisition step, then load only in a network-off
   container;
3. prefer `torch.load(..., weights_only=True)` if compatible, or convert once
   to a hash-bound safetensors artifact in the isolated environment;
4. record the MSA database versions, search date, HHsuite/MMseqs2 parameters,
   sequence coverage, depth and chain-pairing rule per complex;
5. audit all sequence/structure sources against SAbDab2 validation, test and
   external roles;
6. first smoke exactly one train-role complex and verify masks, chain break,
   output shape, finite BF16 values, peak memory and deterministic rerun;
7. only then pre-register an information-matched comparison. Query-only and
   MSA-enabled results must be reported separately.

The most useful near-term role is an explicitly information-richer contact
upper bound and pair-trunk representation probe. It should not delay the frozen
five-condition development matrix or be used to select the current model.
