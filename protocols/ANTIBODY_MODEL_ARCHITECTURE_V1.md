# Antibody model architecture v1

**Status:** CPU/meta correctness verified; GPU and distributed gates pending

The phase-A model is a residue-level bidirectional encoder. It uses pre-RMSNorm
blocks, bias-free QKV and output projections, per-head Q/K RMS normalization,
rotary positions, exact PyTorch scaled-dot-product attention, SwiGLU feed
forwards, and a token embedding tied to the MLM output projection. Training and
attention dropout are zero. No downstream label, antigen identity, affinity,
contact, prediction, or SAbDab2 held-out value is an input.

The vocabulary contains five special tokens, the twenty standard amino acids,
and an explicit `X` token. Heavy, kappa, lambda, unknown-light, paired-heavy,
and paired-light type IDs are distinct. Phase A currently uses only unpaired
types. Native heavy--light continuation remains a separately gated phase.

## Exact presets

| Preset | Layers | Width | Heads | SwiGLU hidden | Exact parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `abf_35m` | 10 | 512 | 8 | 1,376 | 31,648,768 |
| `abf_300m` | 24 | 1,024 | 16 | 2,816 | 308,365,312 |
| `abf_650m` | 32 | 1,280 | 20 | 3,456 | 634,513,920 |

The preset names are scale labels; the exact counts above are authoritative.
The `abf_35m` model is the correctness and systems candidate. A larger preset
may be promoted only after the previous scale passes correctness, recovery,
memory-headroom, throughput, and held-out OAS loss-per-GPU-hour gates.

## Verified evidence

The create-only verification runner used Python 3.10.20 and PyTorch
2.13.0+cu130. It passed deterministic MLM corruption replay, CPU forward replay,
selected-target loss, backward, tied-embedding gradient, and meta-device exact
parameter counting.

- family SHA-256: `1943ab2c538e7d10f9273e21a6a47cb5e1cd972e00df0a91fa81a86e39dc4015`
- verification receipt file SHA-256: `452906c10488d7a05acd12503ccbc4905e60651b2fa579c2420853d40f39ad81`
- receipt payload SHA-256: `833106bee34de7bd8cfeee17d374ab650eacb0263898f2b37fa88acbd4349e15`

This evidence does not show GPU correctness, distributed parity, checkpoint
recovery, useful throughput, validation loss, downstream transfer, or state of
the art. Those claims remain closed until their numbered gates pass.
