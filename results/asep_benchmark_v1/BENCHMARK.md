# AsEP epitope-group test measurement

Status: the single test measurement registered in
`protocols/ASEP_TEST_MEASUREMENT_V1.md`. One configuration, frozen by the
registered rule before the split was read, measured once.

Date: 2026-09-10

Machine-readable: [`test_measurement.json`](test_measurement.json).

## Convention

Matched to the published one, which the literature states explicitly:
threshold-dependent metrics use a probability threshold **selected on the
validation split to maximise MCC**, applied unchanged to test, and metrics are
**pooled over residues** rather than averaged per complex.

The selected threshold was 0.8465 (sd 0.049) — the model's scores sit high, the
same shift seen on the sabdab2 cohort.

## Result, ten seeds

| | mean | sd | 95% CI |
|---|---:|---:|---|
| MCC | **0.1893** | 0.0056 | [0.1859, 0.1924] |
| precision | 0.1754 | 0.0125 | [0.1675, 0.1821] |
| recall | 0.4073 | 0.0588 | [0.3761, 0.4447] |
| F1 | 0.2432 | 0.0060 | [0.2396, 0.2466] |
| **AUROC** | **0.7515** | 0.0046 | [0.7487, 0.7541] |
| AUPRC | 0.1708 | 0.0050 | [0.1677, 0.1736] |

Test prevalence 0.0636, so AUPRC is 2.7x the floor. The test split is harder
than validation, whose prevalence is 0.114.

## Against the published table

Epitope-group split, from Table 2(b) of arXiv 2608.01092:

| method | MCC | precision | recall | AUROC | F1 |
|---|---:|---:|---:|---:|---:|
| ESMFold | 0.018 | 0.113 | 0.034 | – | 0.046 |
| ESMBind | 0.002 | 0.082 | 0.076 | 0.500 | 0.064 |
| EpiPred | −0.006 | 0.089 | 0.158 | – | 0.112 |
| MaSIF-site | 0.046 | 0.164 | 0.174 | – | 0.128 |
| WALLE | 0.077 | 0.143 | 0.266 | 0.544 | 0.145 |
| PEPNet | 0.139 | 0.230 | 0.143 | 0.612 | 0.177 |
| PEPNet+LE | 0.156 | 0.250 | 0.155 | 0.627 | 0.191 |
| GF3DRoPE | 0.159 | 0.212 | 0.216 | 0.706 | 0.212 |
| LF3DRoPE | 0.171 ± 0.010 | **0.248** | 0.189 | 0.690 | 0.215 |
| **this work** | **0.189** | 0.175 | **0.407** | **0.752** | **0.243** |

Higher on MCC, F1, AUROC and recall; **lower on precision** than LF3DRoPE and
PEPNet+LE. The MCC intervals do not overlap: [0.186, 0.192] against
[0.161, 0.181].

AUROC is the cleanest line in the table because it needs no agreement about
where anyone put a threshold.

## What this does not establish

1. **This compares against published numbers, not re-run models.** No baseline
   was re-executed here. Preprocessing differences, the definition of an epitope
   residue, and the corrected against original split file (169 against 170 test
   instances) could each absorb part of the difference.
2. **The problem setup may differ.** This model predicts residue **pairs** and
   derives node marginals from them, so it trains on which antibody residue
   contacts which antigen residue. A method trained on node labels alone uses
   strictly less of the same data. That is a difference in setup, not in
   architecture, and it has not been checked against each published method.
   Until it is, the defensible statement is that this number is higher in this
   setup, not that the method is better.
3. **Precision is lower.** For a use that needs every reported residue to be
   right, LF3DRoPE's operating point is preferable. The advantage here is in
   ranking quality and recall.
4. Ten seeds of one configuration on one split. The `epitope_ratio` split was
   not measured.

## Provenance

| | |
|---|---|
| protocol | `protocols/ASEP_TEST_MEASUREMENT_V1.md`, with its amendment |
| arm | `esm2_esmif1`, frozen on validation before test was read |
| selection | `results/asep_benchmark_v1/arm_selection.json` |
| split | AsEP `epitope_group`, sha256 `678a20cd…` |
| test instances | 169 |
