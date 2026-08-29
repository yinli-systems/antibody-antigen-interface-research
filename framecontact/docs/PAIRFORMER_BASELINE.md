# Authenticated Boltz-1 Pairformer baseline bridge

## What is verified

The baseline uses the official Boltz source at revision
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` and the official Boltz-1 checkpoint
with SHA-256
`fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5`.
The extractor keeps `torch.load(weights_only=True)` and allowlists only the
OmegaConf container types required by the official Lightning metadata. It never
falls back to unrestricted pickle loading.

The extracted artifact contains 2,544 `pairformer_module.*` tensors and the
official 48-block architecture:

| Field | Value |
|---|---:|
| single width | 384 |
| pair width | 128 |
| blocks | 48 |
| attention heads | 16 |
| pairwise heads | 4 |
| Pairformer parameters | 147,400,704 |

The artifact SHA-256 is
`9307036782c09f988c0d428a9e8c5bdcde604d1742e357982ad3b2ec6d27282f`.
The extraction receipt and local real-weight smoke are bound in
`baselines/BASELINE_LOCK_V4.json`.

## Structure adapter

`OfficialBoltzPairformerMonomerEncoder` is the bridge from the SAbDab2 batch
contract to the frozen trunk:

1. build robust right-handed residue frames from real `N/CA/C`;
2. preserve the explicit missing/degenerate-frame mask;
3. form same-chain invariant distance RBF, local direction, relative rotation,
   and sequence-offset pair features;
4. project frozen ESM and pair features into the official single/pair widths;
5. run the strictly loaded official Pairformer with kernels disabled;
6. project single outputs to the task width while exposing the pair output.

Antibody and antigen structures may be independently rotated and translated.
No cross-monomer coordinate subtraction occurs. The dual wrapper instantiates
two separate trunks and adapters, satisfying the no-parameter-sharing contract
at an explicit memory cost.

## Reproduction

Install the pinned adapter dependencies in the project environment:

```bash
pip install -r baselines/requirements-boltz-pairformer.txt
```

Extract once into a new path:

```bash
python scripts/extract_boltz1_pairformer_checkpoint.py \
  --checkpoint /path/to/boltz1_conf.ckpt \
  --boltz-source-root /path/to/boltz \
  --output /new/path/boltz1_pairformer.pt \
  --receipt /new/path/BOLTZ1_PAIRFORMER_EXTRACTION_RECEIPT.json
```

Then run a strict synthetic smoke:

```bash
python scripts/smoke_boltz1_pairformer_adapter.py \
  --artifact /path/to/boltz1_pairformer.pt \
  --receipt /path/to/BOLTZ1_PAIRFORMER_EXTRACTION_RECEIPT.json \
  --boltz-source-root /path/to/boltz \
  --output /new/path/BOLTZ1_PAIRFORMER_SMOKE.json \
  --device cuda
```

Both scripts refuse to overwrite their outputs. The smoke uses synthetic tensors
and labels itself as non-benchmark evidence.

The development-only identical-split entry is isolated from the frozen
eight-condition registry:

```bash
python scripts/train_pairformer_baseline.py \
  --pairformer-artifact /path/to/boltz1_pairformer.pt \
  --pairformer-receipt /path/to/BOLTZ1_PAIRFORMER_EXTRACTION_RECEIPT.json \
  --boltz-source-root /path/to/boltz \
  --experiment adapted_boltz1_pairformer \
  ...the same authenticated SAbDab2 development arguments from the AB03 runbook...
```

Its hash-bound authority is
`protocols/framecontact_pairformer_baseline_v1.json`. The wrapper preserves the
base trainer's contact, marginal, partner-retrieval, calibration, checkpoint,
and receipt paths, but disables FrameContact-specific internal-head diagnostics
because the official Pairformer does not expose that interface. It records the
checkpoint, extraction receipt, source revision, and tensor count in every
development result. Five-seed execution has not started.

## What is not verified

- Native Boltz preprocessing, recycling, structure heads, or end-to-end
  structure prediction are not reproduced.
- The frozen Pairformer trunk has not yet completed the identical SAbDab2
  antigen-fold/temporal five-seed protocol.
- No frozen SAbDab2 or official AsEP test label has been opened for this model.
- The local receipt is CPU-only; AB03 A100 latency, memory, and numerical
  reproduction remain pending.
- New input/output projections are randomly initialized task adapters, so this
  is an adapted pretrained baseline rather than a zero-shot Pairformer result.

Consequently, the current evidence supports “authenticated real checkpoint and
runnable structure adapter,” not “Pairformer improves the benchmark.”
