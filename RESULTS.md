# Results — start here

> **Integrated update:** all 31 starting branch HEADs are now in the combined source history. Read [current branch progress](results/integration_20261007/BRANCH_PROGRESS.md) and [neutral branch names](results/integration_20261007/BRANCH_RENAMES.json). The 30-branch atlas below remains its original frozen audit snapshot. Final cleanup leaves one active branch, `main`; the other 31 merged references are preserved by source history and backup tags. [Actual cleanup receipt](archive/2026-10-07/REMOTE_CLEANUP_RECEIPT.json) · [Validation and restored evidence](results/cleanup_20261007/README.md).

[**Open the results atlas**](results/atlas/README.md) · [Frozen 30-branch snapshot](results/atlas/BRANCHES.md) · [Every evidence location](results/atlas/FAMILIES.md) · [Audit findings](results/atlas/REVIEW_FINDINGS.md)

The 7 October 2026 snapshot organizes the repository into **37 result cards**, **30 branch pages**, and **337 evidence-location pages**. Its complete inventory contains **4,913 distinct path/blob versions** from all 30 frozen branch HEADs; **1,530** match the explicitly documented evidence-location filter. These are file/version counts, not independent experiments.

## What to read

| Question | Entry |
|---|---|
| What are the results, controls and limitations? | [Result cards](results/atlas/README.md) |
| What did each branch contain at the original audit? | [Frozen branch snapshot](results/atlas/BRANCHES.md) |
| What is integrated now, and how is the repository organized? | [Current progress](results/integration_20261007/BRANCH_PROGRESS.md) · [Repository map](docs/REPOSITORY_LAYOUT.md) |
| Which scores can be compared? | [Cohort and estimator guide](results/atlas/COHORTS.md) |
| Which newer claims still need terminal evidence attached? | [Audit findings](results/atlas/REVIEW_FINDINGS.md) |
| Where are all the smaller studies, receipts, logs and protocols? | [Complete evidence-location catalogue](results/atlas/FAMILIES.md) |
| How should future tables and figures be structured? | [Presentation guide](results/atlas/PRESENTATION_GUIDE.md) |
| How can I search every tracked version interactively? | [Self-contained HTML explorer](results/atlas/index.html) |

The HTML file works offline in a browser; GitHub itself displays its source rather than hosting an interactive website. The Markdown cards and branch links work directly on GitHub. No public site has been deployed, and no private repository evidence has been published outside this repository.

## Corrections readers should not miss

The earlier README operations block is a **historical snapshot**, not live status. The final structural-PLM receipt records **36/36 tasks completed with NO-PROMOTION**; the corrected centered RePaRank structure receipt records **217/217 cases**. Later EPIC manuscript claims are separately labelled **report-only** where matching terminal campaign artifacts were not identified in the inspected branch HEADs. Their evidence may exist off-Git.

The old 17-snapshot / 747-link archive and all original reports remain unchanged. No result is deleted to make the presentation look better, and a duplicate artifact on several branches is not counted as additional replication.

## Verify the presentation

```bash
python3 tools/evidence/test_atlas.py
python3 tools/evidence/build_atlas.py --check
python3 tools/evidence/verify.py --history
python3 tools/evidence/verify_integration.py
python3 tools/evidence/verify_cleanup.py --remote --require-clean
```

The atlas checks **87 numeric fields**, source hashes, branch membership and generated-page integrity. These checks are not independent scientific replication. No training, inference, GPU scheduling, unpublished cluster inspection or contact-label reconstruction was performed by this documentation audit. See the [verification receipt](results/atlas/VERIFICATION.json).

## 中文阅读入口

先看上面的 **Results atlas**：每张卡片都注明结论、证据等级、数据集与划分、模型输入、种子、指标聚合方式、局限和固定提交来源。**All branches** 则逐个列出 30 个分支的作用、提交、相对 main 的文件差异与阅读入口。

特别注意：37 张卡片不是 37 个都已独立复现的实验。已核对指标、已提交研究报告、仅有新文稿陈述、未完成实验协议和工程执行记录被明确区分。10 月 4 日的新 EPIC 结果不会因为文稿写得较新，就自动获得“原始证据已验证”的标记。

全部起始分支的历史和进度已经整合进 main。最终清理后只保留一个活跃分支 main；另外 31 个已合并分支引用已归档清理，原始提交、实验结果、失败记录和备份标签仍可还原。上面的 30 分支目录是最初的固定审查快照，不是当前活跃分支列表。
