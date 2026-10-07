# Antibody-Antigen Interface Prediction: Benchmark Evaluation Summary

Date: 2026-09-04

## 1. Cross-Dataset Generalization on SAbDab2 Benchmark

**Evaluation Set**: SAbDab2 2026-07-21 Frozen Benchmark (canonical `ab_ag` cluster split, 455 test complexes, 18,473,412 candidate residue pairs).
**Model Setting**: Zero-shot transfer from AsEP pretraining (no fine-tuning on SAbDab2).

| Method | Training Data | Evaluation Split | Pair Micro AUROC | Pair Macro AUROC | Cohen's d | Negative Noise Mean | Epitope Micro AUROC | Epitope Micro AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPPIDER-seq** | SAbDab2 | SAbDab2 455 Test | 0.6016 | — | — | — | 0.6096 | 0.1151 |
| **EpiFormer** | SAbDab2 | SAbDab2 455 Test | *Not in retained logs*\* | — | — | — | **0.8008** | **0.2775** |
| **Ours: 1D Pretrain** | AsEP-derived | SAbDab2 455 Test | 0.6220 | 0.5662 | 0.3659 | 0.2150 | 0.7272 | 0.2210 |
| **Ours: 3D+Margin** | AsEP-derived | SAbDab2 455 Test | **0.6904** | **0.6618** | **0.6419** | **0.0788** | 0.6820 | 0.1567 |

*\*Note: EpiFormer's published architecture includes a bipartite interaction output matrix; however, in the benchmark release evaluation logs, only the marginal epitope projection was retained.*

### Key Observations
- **Residue-Pair Ranking Enhancement**: Adding 3D EGNN geometry and Margin Separation significantly elevates Pair Micro AUROC from 0.6220 to 0.6904 (+0.0684 gain) and Macro AUROC from 0.5662 to 0.6618 (+0.0956 gain).
- **Negative Noise Suppression**: The negative score mean is suppressed by 63.3% (0.2150 -> 0.0788), and Cohen's d increases by 75.4% (0.3659 -> 0.6419).
- **Pair vs. Node Objective Dynamics**: While 3D+Margin drives strong pairwise separation, the absence of an explicit auxiliary node consistency loss leads to conservative marginal epitope pooling (0.7272 -> 0.6820). Future iterations will introduce a multi-task joint objective ($\mathcal{L}_{\text{pair}} + \lambda_1 \mathcal{L}_{\text{margin}} + \lambda_2 \mathcal{L}_{\text{consistency}}$) to mutually reinforce both objectives.

---

## 2. In-Domain Ablation on Held-out AsEP-derived Test Set

**Evaluation Set**: 172 complexes (105 with complete 3D structural backbones, ~1.90 million candidate residue pairs).

| Metric | 1D Baseline | 3D + Margin Model | Delta |
| :--- | :---: | :---: | :---: |
| **Pair Micro AUROC** | 0.8556 | **0.8791** | **+0.0235** |
| **Pair Macro AUROC** | 0.8251 | **0.8463** | **+0.0212** |
| **Cohen's d (Separation)** | 1.7922 | **1.9365** | **+0.1443** |
| **Precision @ 0.5** | 2.14% | **13.43%** | **+11.29% (6.3x)** |
| **Negative Noise Mean** | 0.2173 | **0.0784** | **-63.9%** |
| **Epitope Micro AUROC** | 0.8144 | **0.8086** | -0.0058 |
| **Paratope Micro AUROC** | 0.8399 | **0.8511** | **+0.0112** |

---

## 3. False Positive Suppression via Phase B Candidate Pruning

Evaluated across both test benchmarks at 50% recall:

| Benchmark Test Split | Candidate Pair Space | Top-30% Contact Retention | **FP Reduction (at 50% Recall)** |
| :--- | :---: | :---: | :---: |
| **AsEP-derived Test Set** | 1.90M pairs | **63.9%** | **-54.7%** (129k -> 58k) |
| **SAbDab2 455 Test Set** | **18.47M pairs** | **56.3%** | **-51.5%** (4.83M -> 2.34M) |

