# 给老师的更新：五 seed 轨迹与 matched epoch 18

老师您好，我又做了一次不触碰 test 的五-seed 同步审计。目前
unary-only 五个 seed 同步到 epoch 21，distogram 同步到 epoch 18；其余
contact-marginal、contact-hybrid 和 triangle 条件仍因 array 并发上限排队。
当前实际运行的是 ParaCloud RTX 4090，不是 A100。

在 matched epoch 18，distogram 相对 unary-only 的 pair contact AUPRC
变化很小（0.01046 -> 0.01090，+0.00044，3/5 seeds 为正）。比较一致的变化
是 precision 提高（0.01202 -> 0.01466，5/5 seeds 为正），但 recall 同时
下降（0.30741 -> 0.22911，5/5 seeds 都下降），predicted-positive ratio
也从 2.85% 降到 1.72%。因此目前更像是模型把 contact prediction 变得
更稀疏、更保守，而不是已经稳定提高了 pair ranking。

residue 指标也是混合的：epitope pooled AUPRC 为 +0.00388，但只有 3/5
seeds 为正；paratope pooled AUPRC 为 -0.03022。最差方向的 partner margin
只改善 +0.00138，seed 一致性不足。随着 epoch 增加，distogram 自身的
precision 从 0.00867 上升到 0.01466，但 recall 从 0.83818 降到 0.22911，
同时 contact AUPRC 从 0.01484 降到 0.01090，所以不能用“训练更久”或
“最新 epoch”直接选模型。

这一结果目前既不是单纯好消息，也不是最终失败：它说明 distogram 确实
改变了 precision--recall 和稀疏性，但还没有证明 contact factorization、
residue aggregation 或 partner specificity 更好。真正关键的比较要等
contact-hybrid 完成后做 distogram vs binary-hybrid，并等 triangle 完成后
按预先冻结的规则检查它是否提高 precision/P@K、同时保留至少 95% 的
distogram recall。

我同步核对了最新一手工作：EpiFormer 支持在 encoder 层间做双向 early
fusion；ProTact 支持“两个单体先独立做 SE(3)-invariant encoding，再用
cross-chain tensor 和 trigonometry updates”的 pose-free 路线；AgForce
则说明 partner-conditioned 模型很容易走单链 shortcut，因此仅有
cross-attention 并不等价于真正利用 partner。基于这个风险，我已经把
一个默认关闭的 counterfactual contact-lift bottleneck 做成了候选：只有
去掉 pair interaction 后消失的正 contact probability 才能进入 partner
retrieval pooling，纯 unary propensity 的贡献严格为零；同时记录 zero-lift、
entropy 和 effective support，防止模型通过死门或全链 diffuse pooling
伪装成 specificity。该候选目前只有工程测试，没有训练结果，我不会把它
写成性能提升。

下一步我会保持当前作业不动，等后续条件完成后按冻结规则比较；开发选择
全部结束后，再把选中的模型、同代码 geometry ablation 和 partner-aware
sequence baseline 放到严格 double-cold split 上重训。Frozen SAbDab2 和
AsEP official test 仍然只在选择全部冻结后一次性打开。

## 更短的聊天版

老师您好，我做了最新五-seed 同步审计。unary-only 已同步到 epoch 21，
distogram 到 epoch 18，其他 contact-hybrid/triangle 条件还在排队。

matched epoch 18 上，distogram 的 contact precision 从 0.01202 提高到
0.01466（5/5 seeds 为正），但 recall 从 0.30741 降到 0.22911（5/5
都下降）；contact AUPRC 只提高 +0.00044，epitope AUPRC +0.00388，
paratope AUPRC -0.03022。因此现在更准确的结论是 distogram 让预测更
稀疏、偏高 precision，但还没有证明 pair ranking 或 partner specificity
稳定改善，也不能按最新 epoch 选模型。

我会等 contact-hybrid 和 triangle 完成后，分别做 distogram vs binary
hybrid、triangle vs distogram 的冻结比较；triangle 必须在提高
precision/P@K 的同时保留至少 95% recall 才会进入下一阶段。最新研究里
EpiFormer/ProTact 支持 early fusion 和 invariant pair trunk，AgForce 也
提示需要主动防止 partner blindness，所以我补了一个默认关闭的
counterfactual contact-lift bottleneck 候选，纯单链 unary 信息不能通过
这条 retrieval 路径。目前它只有工程测试，没有性能结论。Frozen test 和
AsEP official test 都还没有打开。
