# FrameContact research update — 2026-07-31

## 1. What is established

The current confirmatory experiment is an authenticated AsEP development-only
comparison. It has not opened the official AsEP test set, the SAbDab2 frozen
test, or any external benchmark.

Among conditions with all five seeds complete:

| condition | partner margin | contact macro AUPRC | contact pooled AUPRC | contact P@10 | epitope macro AUPRC | epitope pooled AUPRC | paratope macro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| distance | 0.001171 | 0.07039 | 0.03627 | 0.10306 | 0.46873 | 0.34506 | 0.70932 |
| spatial topology | 0.001167 | 0.06728 | 0.03499 | 0.09518 | 0.45513 | 0.33023 | 0.71425 |
| distance + direction | 0.001081 | 0.06628 | 0.03577 | 0.09529 | 0.44859 | 0.32788 | 0.72272 |
| full relative pose | 0.001101 | 0.06223 | 0.03068 | 0.09541 | 0.42888 | 0.29082 | 0.71779 |

The dense sequence-only condition currently has four completed seeds and is
stronger on partner margin, contact prediction, and epitope prediction
(respectively 0.002239, 0.08609, and 0.48971 macro AUPRC). This comparison is
not final until its fifth seed completes. It already falsifies the weak claim
that adding more local-pose channels is automatically beneficial.

## 2. Development-label topology diagnostic

An isolated protocol was frozen before running an authenticated analysis on all
170 AsEP validation complexes. This diagnostic deliberately uses validation
labels to measure label autocorrelation; it is not predictive performance.

| relation | mean leave-one-out neighbor-label AUPRC | mean positive-neighbor lift |
|---|---:|---:|
| sequence window 2 | 0.60744 | 7.1747 |
| sequence window 8 | 0.46232 | 4.4506 |
| sequence window 32 | 0.22234 | 2.0448 |
| spatial radius 6 Å | 0.66727 | 7.3477 |
| spatial radius 8 Å | **0.74129** | 6.8150 |
| spatial radius 12 Å | 0.72242 | 5.2049 |
| spatial radius 16 Å | 0.67232 | 3.9947 |

The within-complex 8 Å spatial minus eight-residue sequence-window difference
is +0.27897, with a 10,000-bootstrap 95% CI of [0.25675, 0.30264]. Expanding
from 8 Å to 16 Å lowers the diagnostic by −0.06897, 95% CI
[−0.08812, −0.05045].

The result supports a narrow conclusion: local epitope topology is strongly
concentrated near 8 Å, but a broad 16 Å graph dilutes that signal. It does not
show that a model can learn the signal, that geometry improves prediction, or
that the signal generalizes to test data.

## 3. Research synthesis

The literature suggests separating three problems instead of calling all of
them “3D positional encoding”:

1. **Within-chain geometric representation.** Local frames, invariant
   distances/directions, IPA-style learned points, GVP/EGNN-style geometric
   message passing, and local-pose attention bias all answer how one monomer
   should encode its own shape.
2. **Partner compatibility.** Antibody-specific epitope prediction is naturally
   a bipartite link-prediction problem, as formalized by
   [AsEP](https://openreview.net/pdf?id=722bb7420a150901f1cca32fd36cd8e2bb5ecb0c).
   A residue classifier that never receives the opposite partner cannot answer
   this question.
3. **Complex pose.** Cross-chain distances and directions become physically
   meaningful only after a relative pose is known or integrated over. Before
   that point, cross-chain exchange should use scalar invariant features rather
   than subtracting coordinates from unrelated frames.

[ProTact](https://www.nature.com/articles/s42003-026-09809-2) supports the
second and third separation: it encodes the two monomers into SE(3)-invariant
representations and refines an inter-chain interaction map using constraints
from both intra-chain geometries. The 2026
[PPLM](https://www.nature.com/articles/s41467-026-70457-5) similarly separates
intra-protein positional treatment from inter-protein attention and uses a
pair/contact trunk. The 2026
[EpiFormer preprint](https://arxiv.org/abs/2606.04154) uses interleaved
bidirectional cross-attention plus node, contact, and distance objectives; it is
a useful implementation precedent, not yet evidence that the same design will
generalize under our stricter structural-novelty splits.

## 4. New falsifiable method: topology-routed multiscale attention

The first adaptive method keeps one dense Q/K/V self-attention path and adds
only local SE(3)-invariant score increments:

\[
A_{ij}^{h} =
\frac{\langle q_i^h,k_j^h\rangle}{\sqrt{d_h}} +
\sum_{r\in\{6,8,12\}}\sum_{c\in\{\mathrm{RBF,dir,rot}\}}
\tanh(g_{r,c,h})\,\mathbf{1}[d_{ij}\le r]\,b_c(G_{ij}).
\]

This design differs from the failed “dense encoder plus spatial encoder plus
one scalar gate” in four testable ways:

- global and local evidence share Q/K/V rather than competing through two
  separately parameterized streams;
- every attention head can select a different spatial scale and geometric
  component;
- 16 Å is excluded from the learned geometric increments because the frozen
  diagnostic showed dilution at that scale;
- all 288 relation gates initialize to zero, so the initial self-attention
  function is exactly content-only.

This is an adaptive exploratory hypothesis, not a claimed novel result. It is
isolated from the confirmatory registry and cannot be promoted using a single
seed or a favorable secondary metric.

## 5. Engineering evidence

- Frozen confirmatory core source and regression-test hashes were restored and
  preserved after the new module was isolated.
- Full local suite: 223/223 tests passed.
- Real AsEP train-batch RTX 4090 BF16 smoke:
  - antibody/antigen lengths: 55/171;
  - forward, backward, and one optimizer step passed;
  - all 288 routing gates received finite nonzero gradients and changed;
  - peak allocated CUDA memory: 126,452,224 bytes;
  - independently rotating/translating antibody and antigen changed contact
    logits by at most 9.54e−7 and marginal logits by 2.38e−7;
  - no validation metric or test member was accessed by the smoke.

## 6. Next gated work

1. Run seed 17 on AsEP development train/validation.
2. Compare it with matched sequence-dense seed 17 under the frozen pilot gate:
   it must improve partner margin or contact macro AUPRC while neither degrades
   by more than 0.01.
3. Only if it passes, expand to five seeds and run the registered geometry
   interventions and head-routing analysis.
4. If the five-seed gate passes, implement a sparse bilateral pair refiner:
   update the explicit antibody–antigen pair map through sparse neighbors on
   both monomers, without using unaligned cross-chain vectors.
5. Freeze development selection before any one-time official/frozen test
   opening.

## 7. Ready-to-send update

老师您好，我继续把 self-attention 几何编码部分做了更严格的验证。当前 AsEP
development 的完整五 seed 条件中，distance、spatial-topology、
distance+direction 和 full relative pose 都没有超过 dense
sequence-only；尤其 contact 和 epitope 指标更低、方差也更大。因此目前证据
不支持“继续增加 full-pose 特征就会更好”。

我另外冻结并运行了一个只使用 validation labels 的拓扑诊断。170 个 complex
上，8 Å 空间邻域的 leave-one-out label-autocorrelation AUPRC 为 0.7413，
而 8-residue 序列窗口为 0.4623；配对差值为 +0.2790，95% CI
[0.2568, 0.3026]。但半径从 8 Å 扩大到 16 Å 后反而下降 0.0690。这个分析
只是标签拓扑统计，不是模型预测结果，但说明局部信号集中在约 8 Å，当前统一
16 Å 空间图可能引入了较多无关邻居。

基于这个结果，我实现了一个隔离的 topology-routed multiscale attention：
保留 dense 全局 self-attention，只在 6/8/12 Å 上加入 radial、direction 和
relative-rotation 的 SE(3)-invariant bias，并让每个 attention head 独立学习
各尺度/分量的 gate。所有 gate 从 0 初始化，所以初始函数严格等于
content-only，也不会再复制两套 Q/K/V。

这套新模块已经通过 223 项测试和真实 AsEP batch 的 RTX 4090 BF16
forward/backward；独立旋转和平移两条链后，contact logits 最大变化
9.54e−7。现在正在按冻结协议运行 seed 17 pilot。只有 partner margin 或
contact macro AUPRC 改善、且另一项下降不超过 0.01，才会扩展到五 seed；
否则会把它作为被证伪的方法停止，不会用 test set 挑结果。后续若五 seed
成立，再继续做显式 contact-map 的双侧 sparse topology refinement。
