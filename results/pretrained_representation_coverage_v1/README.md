# Frozen representation coverage v1

Status: development-only feature extraction. No label, contact, prediction or
score was read, and no training, selection or evaluation is authorized by it.
The frozen test split was **refused**, not counted — see below.

Date: 2026-09-10

Machine-readable: [`coverage.json`](coverage.json), which embeds both extraction
receipts verbatim. The artifacts themselves stay on the cluster.

## What exists

| encoder | width | sides | artifacts | source |
|---|---:|---|---:|---|
| AntiBERTy | 512 | heavy and light chains | 2,558 | pip package weights |
| ESM-IF1 | 512 | antibody and antigen, from backbone | 2,968 | `esm_if1_gvp4_t16_142M_UR50` |

Artifacts are keyed by sequence digest and shared across instances, which is why
the counts are far below the instance counts.

## Coverage of the development splits

| | train | validation |
|---|---:|---:|
| instances | 3,229 | 217 |
| single-domain (no light chain) | 822 | 120 |
| AntiBERTy, antibody side | **3,229 (100%)** | **217 (100%)** |
| ESM-IF1, antibody side | 3,227 (99.9%) | 215 (99.1%) |
| ESM-IF1, antigen side | 3,226 (99.9%) | 217 (100%) |
| all three present | 3,224 (99.8%) | 215 (99.1%) |

The gaps are ESM-IF1's alone and are a property of the data: ten units were
dropped for `backbone_length_mismatch`, where the stored backbone and the
sequence disagree in length. Those instances are kept in the cohort with the
missing features zero-filled and a presence column set to 0, because dropping
them would make any ablation incomparable to the ten-seed baseline measured on
the full cohort.

**A quarter of this dataset has no light chain** (822 of 3,229 train, 120 of 217
validation). That is not a gap. AntiBERTy encodes such an antibody from its
heavy chain alone, and the concatenation along the residue axis is one part
rather than two.

## The manifest's split is not the split

The canonical manifest divides its 3,663 rows into `train` (3,208) and `test`
(455). The double-cold split divides the same instances into train (3,229),
validation (217) and test (217). These are different divisions of the same
cohort, and the disagreement is large:

- Parsing the manifest's `train` rows reaches **88.7%** of the split's train
  instances and **73.3%** of its validation instances.
- The 217 instances of the frozen test split are **split across both manifest
  rows**: 185 sit under manifest `train` and 32 under manifest `test`.

So selecting by manifest split cannot express "the development instances and
nothing else". It either leaves a quarter of validation unextracted, or reaches
it by parsing rows that include the frozen split.

`--instance-split-manifest` selects by instance id instead. The completing run
recorded exactly that:

```
splits_parsed      = {"train": 3023, "test": 423}
instances_selected = 3446              (train 3229 + validation 217)
dropped_units      = {"instance_not_selected": 217, ...}
```

The 217 not selected are the frozen test instances, reached by neither route.
Naming `test` in an instance selection raises rather than being counted: every
other split here is recoverable by other means, and a frozen one is not.

## Provenance

| | |
|---|---|
| manifest sha256 | `548bcfab75923ec6621c1a3fef1aa6673c034f64a9bdc752e7ee15b41b4a4e63` |
| split manifest sha256 | `3da484ae31bd6b54c4bec4073cb980c88d9ab42c3213ca30933ff29186d1e396` |
| extraction | `slurm/extract_pretrained_representations.sbatch`, gpu_4090 |
| backbones | `sabdab2_full_backbones.pt`, 3,663 entries |
