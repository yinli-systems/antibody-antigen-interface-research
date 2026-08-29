# Partner-specific epitope modeling: reply, evidence boundary, and execution decision

Date: 2026-07-30

## The scientifically correct answer to the teacher

The result already reported for `spatial_topology` comes from two independent,
parameter-unshared intra-chain encoders. Each chain receives frozen sequence
embeddings and only its own N/CA/C-derived local geometry. The residue heads in
that experiment do **not** receive the opposite partner. Therefore, the result
shows that an antigen's internal spatial topology can improve localization of
epitope-like surface regions; it does not show that the prediction changes for
different antibodies against the same antigen.

That limitation is exactly the distinction between antibody-agnostic binding
propensity and antibody-specific epitope prediction. The next model must be
evaluated on fixed-antigen/different-antibody counterfactuals, not just on
ordinary residue AUPRC.

## Ready-to-send Chinese reply

老师您好，您指出的是目前这版结果最关键的边界。现在汇报的
`spatial-topology` 模型中，antibody 和 antigen 分别经过参数不共享的
self-attention；每一侧只使用自身的 ESM 表征以及 N/CA/C 构建的局部几何，
当前 epitope/paratope residue head 并没有显式读取对面的 partner 信息。

因此，这一版结果能够说明单体内部空间拓扑有助于定位“更像 epitope”的区域，
但还不能说明模型学到了 antibody-specific epitope。也就是说，对同一个 antigen
换不同 antibody 时，当前模型的 epitope 输出原则上不会随 partner 充分变化，
这确实是下一步需要解决的核心问题。

我准备把模型升级为 interaction-first 架构：先保留 antibody/antigen 各自的
geometry-aware self-attention，再由二者的 residue 表征显式预测完整的
Ab-residue × Ag-residue contact matrix；随后用该 contact 表征控制双向
cross-attention，最后从 interaction-updated features 和 contact marginals
联合输出 epitope/paratope。这样不需要直接比较两个未对齐单体坐标系中的 xyz，
但每个 antigen residue 的 epitope score 会显式依赖当前 antibody。

评测上我会增加两组严格的 partner counterfactual：一组固定 antigen、替换
31 个匹配的错误 antibody，直接回答您刚才的问题；另一组固定 antibody、替换
错误 antigen。主指标不会只看 residue AUPRC，而是看正确 partner 相对错误
partner 的 interaction-only margin、top-1 retrieval 和 MRR。interaction-only
评分会去掉 antibody/antigen 各自的单体 propensity，避免模型只靠 CDR 或抗原
表面先验取巧。同时保留 partner shuffle、geometry shuffle 和 unseen-antigen-fold
测试。现有 spatial 结果我会明确标注为 monomer geometry evidence，不会把它表述为
partner-specific 的证据。

## What the literature says

### 1. AsEP and WALLE establish the correct task formulation

AsEP defines epitope prediction as antigen node classification conditioned on
the antibody graph, because different antibodies can bind different regions of
the same antigen. It also supplies exact antibody-antigen bipartite edges and
casts the finer task as residue-pair link prediction. WALLE independently
encodes the two graphs and applies a pair decoder, then aggregates predicted
pair edges to epitope nodes. This directly supports an explicit contact-first
formulation.

Limit: WALLE's unseen-epitope-group performance drops substantially, and its
published model uses relatively simple distance topology and late pair
decoding. It is a required external baseline, not the final architecture.

Sources:

- [AsEP/WALLE NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/15add6732964d5b1f0954058bf3ccc88-Paper-Datasets_and_Benchmarks_Track.pdf)
- [Official AsEP/WALLE repository](https://github.com/biochunan/AsEP-dataset)

### 2. PECAN and MIPE show that partner context and pair supervision matter

PECAN uses separate structural graph encoders, graph convolution, and a partner
context attention layer so that interface prediction on one protein depends on
the other. MIPE combines sequence and structure encoders and explicitly
supervises antibody-oriented and antigen-oriented interaction matrices against
the actual residue interaction map. Both are closer to the teacher's biological
question than an antigen-only classifier.

Limit: their original data/split choices are not a substitute for the new
Foldseek-audited antigen-fold split, and reported metrics cannot be transplanted
into SAbDab2.

Sources:

- [PECAN paper and code description](https://pmc.ncbi.nlm.nih.gov/articles/PMC7332568/)
- [MIPE IJCAI 2024 paper](https://www.ijcai.org/proceedings/2024/0669.pdf)

### 3. Recent partner-aware models favor early or interleaved fusion

SPPIDER-seq applies partner-conditioned cross-attention to ESM-2 residue
embeddings and demonstrates the basic sequence-only control that the prediction
for one residue can depend on the partner. EpiFormer interleaves geometric
message passing with bidirectional cross-attention rather than combining chains
only at the decoder; its official code exposes direction-specific cross gates.

Decision: keep the late contact-gated design as the clean first confirmatory
model, then preregister an interleaved contact-gated variant as an architectural
ablation. Cross-attention alone is not accepted as contact evidence; the pair
head still receives explicit contact supervision.

Caution: EpiFormer is a June 2026 preprint with a very young repository and is
evaluated on AsEP-style splits. It is valuable current evidence and a baseline,
but its headline gain is not yet a result for our structural-novelty setting.

Sources:

- [SPPIDER-seq 2026 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142516/)
- [EpiFormer preprint](https://arxiv.org/abs/2606.04154)
- [Official EpiFormer repository](https://github.com/mansoor181/epiformer)

### 4. Surface models are the strongest follow-up for discontinuous epitopes

MaSIF-search learns complementary surface fingerprints and can compare unbound
partners without first sharing a global coordinate frame. The newer SurfBind
uses patch-level molecular surfaces, binder-aware cross-attention, and
coarse-to-fine propagation, directly targeting discontinuous and
surface-driven epitopes.

Decision: do not put a full molecular-surface pipeline into the first
confirmatory run. First add cheap, auditable surface features (relative SASA,
surface mask, local curvature/normal when authenticated), then compare a
surface-patch model as a second-stage baseline. This isolates the effect of the
proposed local-frame encoder instead of changing every component at once.

Sources:

- [MaSIF Nature Methods paper](https://www.nature.com/articles/s41592-019-0666-6)
- [Official MaSIF repository](https://github.com/lpdi-epfl/masif)
- [SurfBind KDD 2026 paper/preprint](https://arxiv.org/abs/2606.23830)

## Architecture decision

1. **Monomer stage**: frozen ESM features plus parameter-unshared antibody and
   antigen self-attention. Compare dense, topology, distance, direction, and
   full local pose under identical initialization and parameter count.
2. **Pair stage**: explicit low-rank residue-pair interaction logits. Keep
   antibody unary and antigen unary terms for calibrated cognate contact
   prediction, but separate them from the jointly conditioned pair term.
3. **Interaction stage**: contact-gated bidirectional cross-attention. No raw
   inter-chain xyz enters because independently supplied monomers have no shared
   pose.
4. **Output stage**: exact residue-pair contacts plus contact-derived and
   interaction-updated epitope/paratope marginals.
5. **Follow-up ablation**: one late cross block versus interleaved cross blocks;
   antigen-to-antibody, antibody-to-antigen, bidirectional, and no-cross modes.

## Evaluation required to answer the teacher rather than merely improve AUPRC

### Primary: fixed antigen, different antibodies

- Freeze 31 wrong antibodies per antigen from label-free metadata.
- Match antibody format, length decile, and CDR residue-count decile.
- Exclude exact sequence, antibody cluster, and documented cognate groups.
- Report interaction-only cognate margin, top-1 retrieval, MRR, and rank.
- Cluster-bootstrap by antigen structural cluster.

### Reciprocal: fixed antibody, different antigens

- Freeze 31 wrong antigens per antibody.
- Match antigen class, length decile, and surface residue-count decile.
- Cluster-bootstrap by antibody cluster.

### Required negative controls

- Total-contact score versus interaction-only score.
- Unary/marginal-only score.
- Antigen-only spatial model.
- Sequence-only partner cross-attention.
- Partner shuffle in both directions.
- Geometry shuffle, frame corruption, graph rewiring, and coordinate noise.
- Independent antibody/antigen rigid transforms as an exact invariance test.

### Required frozen-test outputs

- Residue-pair contact mean-complex and pooled AUPRC.
- Epitope/paratope AUPRC, AUROC, MCC, precision, recall, and F1.
- Brier score, ECE-15, and reliability plots.
- Both retrieval directions with 10,000 cluster bootstraps.
- Per-seed results for seeds 17, 42, 73, 101, and 137.
- Case studies stratified by antigen length, epitope prevalence,
  discontinuity, and nearest training-patch novelty.

## Implementation completed locally on 2026-07-30

- Pair-contact logits are decomposed into joint interaction, antibody unary,
  antigen unary, and global bias components.
- The specificity objective can use only the joint interaction component, so
  monomer propensity cannot establish partner specificity.
- A reciprocal label-free panel builder freezes both fixed-antigen/wrong-
  antibody and fixed-antibody/wrong-antigen controls.
- Protocol v2 was frozen without modifying v1. It makes the fixed-antigen test
  primary and preserves the untouched test boundary.

This is implementation and protocol evidence only. The real SAbDab2 pair
authority, structural split audit, five A100 seeds, official external baselines,
and single frozen-test opening are still required before reporting a biological
benchmark result.
