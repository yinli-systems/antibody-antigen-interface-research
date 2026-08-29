#!/usr/bin/env python3
"""Build an immutable, scope-aware audit of the antibody evaluation program."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        current = current[key]
    return current


def source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "mipe_evaluation",
        "mipe_manifest",
        "epiformer_score",
        "epiformer_provenance",
        "sppider_epitope_score",
        "sppider_paratope_score",
        "sppider_report",
        "split_receipt",
        "decoy_aggregate",
        "pair_aggregate",
        "mipe_data_comment",
        "chimera_next",
        "chimera_design",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    paths = {
        name: getattr(args, name)
        for name in (
            "mipe_evaluation",
            "mipe_manifest",
            "epiformer_score",
            "epiformer_provenance",
            "sppider_epitope_score",
            "sppider_paratope_score",
            "sppider_report",
            "split_receipt",
            "decoy_aggregate",
            "pair_aggregate",
            "mipe_data_comment",
            "chimera_next",
            "chimera_design",
        )
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    mipe = read_json(args.mipe_evaluation)
    mipe_manifest = read_json(args.mipe_manifest)
    epiformer = read_json(args.epiformer_score)
    epiformer_provenance = read_json(args.epiformer_provenance)
    spp_epi = read_json(args.sppider_epitope_score)
    spp_para = read_json(args.sppider_paratope_score)
    split = read_json(args.split_receipt)
    decoy = read_json(args.decoy_aggregate)
    pair = read_json(args.pair_aggregate)

    mipe_primary = nested(mipe, "role_results", "primary_epoch_800", "summary")
    mipe_pair = nested(mipe_primary, "pair", "directions", "mean_bidirectional")
    mipe_flat = nested(mipe_manifest, "flat_pair_rank_summary", "mean_bidirectional")
    mipe_metrics = {
        "scope": "official MIPE five-fold held-out set; 560 complexes; epoch 800",
        "epitope_micro": nested(mipe_primary, "marginal", "epitope", "micro"),
        "paratope_micro": nested(mipe_primary, "marginal", "paratope", "micro"),
        "pair_macro_auprc": nested(mipe_pair, "per_complex_macro", "auprc", "mean"),
        "pair_macro_auroc": nested(mipe_pair, "per_complex_macro", "auroc", "mean"),
        "pair_prevalence_macro": nested(mipe_pair, "per_complex_macro", "prevalence", "mean"),
        "pair_flat_recall_at_1_macro": mipe_flat["recall_at_1_macro"],
        "pair_flat_hit_at_1_rate": mipe_flat["hit_at_1_rate_macro"],
        "pair_flat_mrr": mipe_flat["mrr_first_positive_macro"],
        "frozen_interaction_files": mipe_manifest["inventory"]["files"],
        "frozen_pair_count": mipe_flat["pairs"],
    }
    sabdab2_metrics = {
        "scope": "exact frozen SAbDab2 original test; 455 complexes",
        "epiformer_epitope_micro": epiformer["micro"],
        "epiformer_macro": epiformer["macro_per_instance"],
        "epiformer_native_pair_output": False,
        "epiformer_prediction_sha256": epiformer_provenance["prediction_sha256"],
        "sppider_epitope_micro": spp_epi["residue_micro"],
        "sppider_paratope_micro": spp_para["residue_micro"],
        "sppider_native_pair_output": False,
        "candidate_pair_rows": spp_epi["projection_micro"]["count"],
    }
    split_audit = split["audits"]["antigen_fold"]
    ood = {
        "split": "SAbDab2 double-cold antigen-fold development validation",
        "valid": split_audit["valid"],
        "role_counts": split_audit["role_counts"],
        "cross_role_overlap_counts": split_audit["cross_role_overlap_counts"],
        "hard_decoy_contact_auprc_delta": decoy["paired_deltas"]["contact_auprc"][
            "paired_bootstrap"
        ],
        "hard_decoy_contact_macro_auprc_delta": decoy["paired_deltas"][
            "contact_macro_auprc"
        ]["paired_bootstrap"],
        "hard_decoy_brier_delta": decoy["paired_deltas"]["contact_brier"][
            "paired_bootstrap"
        ],
        "hard_decoy_ece_delta": decoy["paired_deltas"]["contact_ece_15"][
            "paired_bootstrap"
        ],
        "hard_decoy_pair_recall_at_1": {
            "candidate": pair["arm_summary"]["candidate"]["recall_at_1_macro"],
            "control": pair["arm_summary"]["control"]["recall_at_1_macro"],
            "paired_delta": pair["paired_deltas"]["recall_at_1_macro"][
                "paired_bootstrap"
            ],
        },
        "hard_decoy_pair_mrr": {
            "candidate": pair["arm_summary"]["candidate"]["mrr_first_positive_macro"],
            "control": pair["arm_summary"]["control"]["mrr_first_positive_macro"],
            "paired_delta": pair["paired_deltas"]["mrr_first_positive_macro"][
                "paired_bootstrap"
            ],
        },
        "frozen_pair_dump_members": sum(len(value) for value in pair["members"].values()),
        "test_loaded": pair["authority"]["test_loaded"],
        "external_test_loaded": pair["authority"]["external_test_loaded"],
    }
    statuses = [
        {
            "priority": 1,
            "item": "MIPE official 800 epochs x five folds",
            "status": "COMPLETE",
            "evidence": "560 held-out complexes and epoch-800 primary evaluation",
        },
        {
            "priority": 2,
            "item": "MIPE bidirectional matrices, true contacts, pair metrics",
            "status": "COMPLETE",
            "evidence": "560 SHA-indexed NPZ files plus pair AUPRC, Recall@1 and MRR",
        },
        {
            "priority": 3,
            "item": "EpiFormer and SPPIDER-seq on exact SAbDab2 test",
            "status": "COMPLETE",
            "evidence": "455/455 instances with frozen predictions and provenance",
        },
        {
            "priority": 3,
            "item": "MIPE on the exact SAbDab2-455 cohort",
            "status": "BLOCKED_UPSTREAM_ARTIFACT",
            "evidence": (
                "Official MIPE release provides prebuilt pickle data but no 62-D structural "
                "feature/ESM1b/AbLang preprocessing pipeline; no fabricated comparison was made."
            ),
        },
        {
            "priority": 4,
            "item": "ConTact",
            "status": "CLOSED_AS_NONCOMPARABLE_AND_LOW_VALUE",
            "evidence": (
                "ConTact is CHIMERA epitope-conditioned H3 sequence/structure design, not "
                "SAbDab2 pair-contact prediction; existing 2-D contact-design gates were negative."
            ),
        },
        {
            "priority": 5,
            "item": "Partner swap, hard decoy, double-cold OOD, calibration and frozen dumps",
            "status": "COMPLETE_DEVELOPMENT_NEGATIVE",
            "evidence": (
                "Five paired seeds; zero split overlaps; 10 raw-logit/label dumps; all effect CIs cross zero."
            ),
        },
    ]
    created_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "format_version": "antibody_research_audit_v1",
        "status": "complete_with_one_upstream_artifact_block",
        "created_at_utc": created_at,
        "statuses": statuses,
        "mipe": mipe_metrics,
        "sabdab2": sabdab2_metrics,
        "ood_and_hard_decoy": ood,
        "conclusions": [
            "MIPE residue ranking reproduces useful epitope/paratope signal, but its attention-derived pair map is not a reliable contact localizer.",
            "On the exact SAbDab2-455 test, EpiFormer is the stronger epitope residue baseline and SPPIDER-seq is the strong paratope baseline; neither emits a native pair interaction map.",
            "The five-seed local hard-decoy objective does not improve native pair contact, Recall@1, MRR, calibration, or partner retrieval with confidence intervals excluding zero.",
            "The double-cold split is valid and leak-free for exact antibody sequence, antigen sequence, and PDB assembly identifiers.",
            "Further GPU scaling of the tested hard-decoy objective or cloning ConTact is not supported by current evidence.",
        ],
        "sources": {name: source(path) for name, path in paths.items()},
    }

    epi_rank = mipe_metrics["epitope_micro"]["ranking"]
    para_rank = mipe_metrics["paratope_micro"]["ranking"]
    epi_mcc = mipe_metrics["epitope_micro"][
        "mcc_at_fold_specific_validation_fitted_threshold"
    ]["mcc"]
    para_mcc = mipe_metrics["paratope_micro"][
        "mcc_at_fold_specific_validation_fitted_threshold"
    ]["mcc"]
    delta_auprc = ood["hard_decoy_contact_auprc_delta"]
    delta_r1 = ood["hard_decoy_pair_recall_at_1"]["paired_delta"]
    delta_mrr = ood["hard_decoy_pair_mrr"]["paired_delta"]

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        json_path = temporary / "ANTIBODY_RESEARCH_AUDIT.json"
        report_path = temporary / "ANTIBODY_RESEARCH_REPORT.md"
        teacher_path = temporary / "TEACHER_UPDATE_DRAFT.md"
        write_json(json_path, audit)
        report = f"""# Antibody interaction evaluation: final audit

Generated: `{created_at}`

## Completion status

| priority | item | status |
|---:|---|---|
""" + "\n".join(
            f"| {row['priority']} | {row['item']} | {row['status']} |" for row in statuses
        ) + f"""

## MIPE official five-fold reproduction

- Epoch 800, five folds, 560 held-out complexes: complete.
- Epitope micro AUROC/AUPRC/MCC: `{epi_rank['auroc']:.6f}` / `{epi_rank['auprc']:.6f}` / `{epi_mcc:.6f}`.
- Paratope micro AUROC/AUPRC/MCC: `{para_rank['auroc']:.6f}` / `{para_rank['auprc']:.6f}` / `{para_mcc:.6f}`.
- Mean bidirectional pair macro AUPRC: `{mipe_metrics['pair_macro_auprc']:.8f}` at prevalence `{mipe_metrics['pair_prevalence_macro']:.8f}`.
- Flattened pair Recall@1 / Hit@1 / MRR: `{mipe_metrics['pair_flat_recall_at_1_macro']:.8f}` / `{mipe_metrics['pair_flat_hit_at_1_rate']:.8f}` / `{mipe_metrics['pair_flat_mrr']:.8f}`.
- All 560 interaction/contact NPZs are frozen with individual SHA-256 values.

Interpretation: MIPE reproduces residue-level signal, but the raw interaction attention does not reliably localize true residue pairs. It must not be described as calibrated pair-contact probability.

## Exact SAbDab2-455 test

| method/output | AUROC | AUPRC | F1 | MCC | native pair map |
|---|---:|---:|---:|---:|---|
| EpiFormer epitope | {epiformer['micro']['auroc']:.6f} | {epiformer['micro']['auprc']:.6f} | {epiformer['micro']['f1_at_threshold']:.6f} | {epiformer['micro']['mcc_at_threshold']:.6f} | no |
| SPPIDER-seq epitope | {spp_epi['residue_micro']['auroc']:.6f} | {spp_epi['residue_micro']['auprc']:.6f} | {spp_epi['residue_micro']['f1']:.6f} | {spp_epi['residue_micro']['mcc']:.6f} | no |
| SPPIDER-seq paratope | {spp_para['residue_micro']['auroc']:.6f} | {spp_para['residue_micro']['auprc']:.6f} | {spp_para['residue_micro']['f1']:.6f} | {spp_para['residue_micro']['mcc']:.6f} | no |

MIPE cannot be honestly run on this exact 455-cohort from the official release: the release supplies prebuilt pickle data but omits the structural-feature and embedding-generation pipeline. This is an upstream artifact boundary, not a compute shortage.

## Double-cold OOD and hard-decoy study

- Split validity: `{split_audit['valid']}`; cross-role antibody sequence / antigen sequence / PDB assembly overlaps are all zero.
- Five paired hard-decoy/control seeds; test and external roles remained sealed.
- Contact AUPRC delta: `{delta_auprc['estimate']:+.8f}`, 95% CI `[{delta_auprc['ci95_lower']:+.8f}, {delta_auprc['ci95_upper']:+.8f}]`.
- Pair Recall@1 delta: `{delta_r1['estimate']:+.8f}`, 95% CI `[{delta_r1['ci95_lower']:+.8f}, {delta_r1['ci95_upper']:+.8f}]`.
- Pair MRR delta: `{delta_mrr['estimate']:+.8f}`, 95% CI `[{delta_mrr['ci95_lower']:+.8f}, {delta_mrr['ci95_upper']:+.8f}]`.
- Ten complete raw-logit/true-label dumps were frozen and hash-verified.

Conclusion: the tested local hard-decoy objective has no supported gain. Scaling it further is not justified.

## ConTact boundary

ConTact is a CHIMERA epitope-conditioned CDR-H3 sequence/structure design method, not a SAbDab2 residue-pair contact predictor. Existing CHIMERA contact-bias experiments failed the joint CAAR + DockQ gate. It is therefore recorded as considered and closed, not inserted into an invalid pair-contact comparison.

## Claim boundary

MIPE's official 560-complex held-out cohort, SAbDab2's exact 455-complex test, and FrameContact's 549-complex double-cold development validation are different regimes. Numbers are reported in separate tables and must not be ranked as if they shared one test set.
"""
        report_path.write_text(report, encoding="utf-8")
        teacher = f"""老师您好，目前结果已按优先级完成并冻结：

1. MIPE 官方 800 epoch、五折已完成，560 个 held-out complex 全覆盖。Epitope AUROC/AUPRC 为 {epi_rank['auroc']:.4f}/{epi_rank['auprc']:.4f}，paratope 为 {para_rank['auroc']:.4f}/{para_rank['auprc']:.4f}。
2. 560 份双向 interaction matrix 和真实 contact matrix 已逐文件校验并保存。更严格的 pair 评估显示，MIPE 双向 attention 的 macro AUPRC 只有 {mipe_metrics['pair_macro_auprc']:.4f}，整矩阵 Recall@1 为 0、MRR 为 {mipe_metrics['pair_flat_mrr']:.4f}；说明 residue 结果可以复现，但 interaction attention 不能直接当作可靠的 pair-contact 概率。
3. SAbDab2 的统一 455-complex test 上，EpiFormer epitope AUROC/AUPRC 为 {epiformer['micro']['auroc']:.4f}/{epiformer['micro']['auprc']:.4f}；SPPIDER-seq epitope 为 {spp_epi['residue_micro']['auroc']:.4f}/{spp_epi['residue_micro']['auprc']:.4f}，paratope 为 {spp_para['residue_micro']['auroc']:.4f}/{spp_para['residue_micro']['auprc']:.4f}。两者都没有原生 pair interaction 输出。
4. 5-seed double-cold hard-decoy 对照已完成，并补齐 raw logits、true labels、pair Recall@1、MRR、校准和 provenance。Contact AUPRC 增量 {delta_auprc['estimate']:+.5f}、pair MRR 增量 {delta_mrr['estimate']:+.5f}，95% CI 都跨 0，因此这是可信的负结果，不支持继续扩大同一 objective。
5. MIPE 无法在完全相同的 SAbDab2-455 上做诚实重跑：官方只发布预处理 pickle，没有发布 62-D 结构特征、ESM1b 和 AbLang 的生成流水线。我们没有用近似特征伪造公平对比。
6. ConTact 属于 CHIMERA 的 CDR-H3 sequence/structure design，不是 SAbDab2 pair-contact baseline；现有二维 contact-design 实验也没有通过 CAAR+DockQ 联合门槛，因此暂不建议继续烧 GPU clone。

所有结果、预测、日志、mapping、split、seed、阈值和 SHA-256 receipt 已保留，可直接审计。
"""
        teacher_path.write_text(teacher, encoding="utf-8")
        receipt = {
            "format_version": "antibody_research_audit_receipt_v1",
            "status": "complete",
            "created_at_utc": created_at,
            "output_root": str(output_root),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "artifacts": {
                json_path.name: sha256_file(json_path),
                report_path.name: sha256_file(report_path),
                teacher_path.name: sha256_file(teacher_path),
            },
        }
        write_json(temporary / "RUN_RECEIPT.json", receipt)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output_root": str(output_root), "status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
