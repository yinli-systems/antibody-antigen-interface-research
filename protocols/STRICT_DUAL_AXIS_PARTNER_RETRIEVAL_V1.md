# Strict dual-axis antibody–antigen partner retrieval v1

Status: **frozen implementation contract; execution blocked until the split authority passes**.

## Scientific target

The primary question is whether fold-local contact/correspondence auxiliary supervision improves partner specificity beyond an otherwise parameter-identical pair-aware reranker under strict dual-axis homology separation. The primary preregistered contrast is **R2 − R1** on bidirectional partner retrieval. Contact/interface prediction is a mechanistic secondary endpoint and affinity is outside this protocol.

## Split boundary

Training may start only after a create-only assignment plus an independent audit both authorize training. The boundary is exactly **foldwise direct-80/80 four-view homology-separated C3, S2-like development**: direct sequence edges require at least 80% aligned-residue identity and 80% reciprocal coverage, inclusively, in any required view (VH, non-empty VL, concatenated antibody, antigen). Three outer folds and three nested inner folds per outer are frozen. Source validation and source test remain sealed. A failed solver or infeasible assignment does not authorize threshold relaxation.

The current split solve is Slurm job `1643164`. No H0 or learned model may run from its provisional files before the final assignment and independent audit receipts pass `authority.py`.

## Candidate and negative semantics

Observed exact antibody–antigen pairs are positives. Every unobserved cross-pair in a finite candidate library is an **unlabeled ranking candidate**, not an experimentally verified non-binder. Binary binder/non-binder AUROC is not a primary endpoint.

The fixed ladder is:

1. **H0-kmer**: no-training 3-mer Jaccard homology transfer through observed training pairs.
2. **H0-NW**: no-training exact global NW-family homology transfer, Biopython 1.83/BLOSUM62, both orientations.
3. **D0**: global AntiBERTy/ESM-2 dual encoder with symmetric multi-positive contrastive training.
4. **L1**: sequence-derived CDR-slot late interaction. CDRs come only from pinned ANARCII/IMGT sequence numbering; no structural interface masks are accepted.
5. **R1**: pair-aware two-layer bidirectional cross-attention reranker over the exact same fold/seed D0 checkpoint used by R2.
6. **R2**: parameter-identical R1 plus fold-local contact/correspondence auxiliary supervision.

## CALM / Ab-CASLR input boundary

CALM's full-sequence retrieval is a valid sequence-only conceptual reference. CALM also reports a masked variant whose epitope/paratope inputs are derived from structural contact annotations; that is a **privileged-input oracle/reference**, not a fair baseline for our sequence-native deployment setting. L1 therefore uses only sequence-derived CDR masks and never a test epitope/paratope/contact mask.

Ab-CASLR motivates CDR-aware local interaction, but its reported Hits@K values are not directly compared numerically to ours because the candidate libraries and split protocol differ.

## R1/R2 fairness

Historical `ProtocolReranker` used no contact head for R1 and a contact head for R2. Strict v1 does not alter that historical code. Instead, `MatchedBudgetReranker` instantiates the same 64-d contact head, parameter names, shapes, forward graph, optimizer budget, D0 bytes, seeds, training pair order and top-64 candidate sets for both R1 and R2.

R1 is forbidden from reading contact labels. Its auxiliary path is a label-free zero-valued function of the contact-head logits. R2 alone loads contact labels from fold-local training pairs and adds weight 0.10 class-balanced contact BCE. Thus the intended R2−R1 difference is auxiliary correspondence supervision, not capacity or candidate-set changes.

## L1 CDR authority

ANARCII is pinned to version 2.0.8, source commit `e0d8f192f5a861e03a50918f114d0f5735e42333`, wheel SHA-256 `6ac81caef04a9cad1af1ea320b35bdced620fe2bfdd0b8b17b98131495b584ca`, model SHA-256 `03e9d07d13a3ec98a9d9fe57a0b2ff0853218aa2d7bdd50a25b6d83ff2bea2e0`, and parameter SHA-256 `4d8e5fc9d8580f5e26a190a21853dd3f3770fbf83ce45a46a53d92b083063c39`. IMGT CDR ranges are 27–38, 56–65 and 105–117 on each present chain. The six slots are H1/H2/H3/L1/L2/L3; missing light-chain slots are absent, not zero-positive pseudo-residues.

## Metrics and uncertainty

Report antibody→antigen and antigen→antibody separately plus their bidirectional mean. Per-query endpoints are MRR and R@1/R@5/R@10. Chance recall is computed exactly from finite library size and each anchor's positive degree; chance enrichment is reported alongside raw recall.

For paired model comparisons, preserve identical query/candidate/positive hashes. Report:

- paired query bootstrap;
- grouped bootstrap over frozen evolutionary components;
- crossed antibody/antigen axis resampling as an exploratory conditional interval; and
- all three fixed training seeds `{17,42,73}` rather than selecting the best seed.

The primary contrast is R2−R1. D0/H0, L1/D0 and R1/D0 are secondary fixed comparisons.

## Score/contact dissociation diagnostics

For the same trained model, report partner score and contact/interface AP separately. Diagnose score aggregation, per-query score scale, candidate library size, positive degree and within-query calibration. A contact-AP gain cannot be restated as partner-specificity gain; partner MRR cannot be restated as contact localization.

## Compute budget

Once split authority passes, the fixed outer-fold program is:

- H0-kmer + H0-NW: deterministic CPU/no-training evaluation on three outer folds;
- D0: 3 outer folds × 3 seeds = 9 learned runs, 58 fixed epochs;
- L1: 3 × 3 = 9 learned runs, 40 fixed epochs;
- R1: 3 × 3 = 9 learned runs, 40 fixed epochs;
- R2: 3 × 3 = 9 learned runs, 40 fixed epochs;
- total learned outer runs = **36**, with each R1/R2 pair reusing identical D0 checkpoint bytes.

The nested inner folds remain frozen and available for any strictly preregistered calibration/selection, but strict v1 architecture/hyperparameters are already frozen and cannot be changed after outer scores are opened.

## Hard stops

Do not execute if an identical experiment hash already has a complete receipt. Do not train if the split is not independently authorized. Do not relax the 80/80 boundary after infeasibility. Block R2 if its fold-local contact-cache provenance fails, while preserving any legally completed H0/D0/L1/R1 work. Any candidate-library/positive hash drift, zero-positive query, R1/R2 parameter mismatch, sealed test/OOD access or out-of-fold cache entry is a hard failure.
