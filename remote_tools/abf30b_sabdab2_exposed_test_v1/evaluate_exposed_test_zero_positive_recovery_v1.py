#!/usr/bin/env python3
"""Run the pinned test evaluator with eligible-complex metric handling."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
import types


PINNED_EVALUATOR_SHA256 = "805ad46bf5f1c49b377c4b6f56e7c5143fcdac6b4f68a53f20da0ccd40f0dde3"
CONTACT_RECOVERY_SHA256 = "29fec12f6af8b5b31648ee32397238dd23959574b540d359d87c987940be159b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def take_option(name: str) -> str:
    index = sys.argv.index(name)
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def option_value(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def patched_evaluate_source(original: str) -> str:
    source = textwrap.dedent(original)
    initialization_old = '''    residue: dict[str, dict[str, list[np.ndarray]]] = {
        "paratope": {"labels": [], "logits": [], "per_complex_auprc": []},
        "epitope": {"labels": [], "logits": [], "per_complex_auprc": []},
    }
    contact_per_complex_auprc: list[float] = []
    contact_ranked_by_instance: dict[str, dict[str, Any]] = {}'''
    initialization_new = '''    residue: dict[str, dict[str, Any]] = {
        "paratope": {"labels": [], "logits": [], "per_complex_auprc": [], "per_complex_auprc_by_instance": {}, "total_complex_count": 0},
        "epitope": {"labels": [], "logits": [], "per_complex_auprc": [], "per_complex_auprc_by_instance": {}, "total_complex_count": 0},
    }
    contact_per_complex_auprc: list[float] = []
    contact_per_complex_auprc_by_instance: dict[str, float] = {}
    contact_total_complex_count = 0
    contact_ranked_by_instance: dict[str, dict[str, Any]] = {}'''
    contact_old = '''            contact_per_complex_auprc.append(
                float(average_precision_score(labels, _sigmoid_numpy(logits)))
            )
            ranked = ranked_pair_metrics(
                labels,
                _sigmoid_numpy(logits),
                antibody_length=int(output["antibody_mask"][batch_index].sum()),
                antigen_length=int(output["antigen_mask"][batch_index].sum()),
            )
            if str(instance_id) in contact_ranked_by_instance:
                raise ValueError(f"Duplicate evaluation instance: {instance_id}")
            contact_ranked_by_instance[str(instance_id)] = ranked'''
    contact_new = '''            contact_total_complex_count += 1
            identifier = str(instance_id)
            if identifier in contact_per_complex_auprc_by_instance:
                raise ValueError(f"Duplicate eligible evaluation instance: {instance_id}")
            if int(labels.sum()) > 0:
                contact_ap = float(average_precision_score(labels, _sigmoid_numpy(logits)))
                contact_per_complex_auprc.append(contact_ap)
                contact_per_complex_auprc_by_instance[identifier] = contact_ap
                contact_ranked_by_instance[identifier] = ranked_pair_metrics(
                    labels,
                    _sigmoid_numpy(logits),
                    antibody_length=int(output["antibody_mask"][batch_index].sum()),
                    antigen_length=int(output["antigen_mask"][batch_index].sum()),
                )'''
    residue_old = '''                residue[name]["labels"].append(local_labels)
                residue[name]["logits"].append(local_logits)
                residue[name]["per_complex_auprc"].append(
                    float(
                        average_precision_score(
                            local_labels, _sigmoid_numpy(local_logits)
                        )
                    )
                )'''
    residue_new = '''                residue[name]["labels"].append(local_labels)
                residue[name]["logits"].append(local_logits)
                residue[name]["total_complex_count"] += 1
                if int(local_labels.sum()) > 0:
                    local_ap = float(average_precision_score(local_labels, _sigmoid_numpy(local_logits)))
                    residue[name]["per_complex_auprc"].append(local_ap)
                    residue[name]["per_complex_auprc_by_instance"][str(instance_id)] = local_ap'''
    contact_result_old = '''            "mean_per_complex_auprc": float(np.mean(contact_per_complex_auprc)),
            "complex_count": len(contact_per_complex_auprc),'''
    contact_result_new = '''            "mean_per_complex_auprc": float(np.mean(contact_per_complex_auprc)),
            "per_complex_auprc_by_instance": contact_per_complex_auprc_by_instance,
            "complex_count": len(contact_per_complex_auprc),
            "eligible_complex_count": len(contact_per_complex_auprc),
            "total_complex_count": contact_total_complex_count,
            "zero_positive_complex_count": contact_total_complex_count - len(contact_per_complex_auprc),'''
    residue_result_old = '''            "mean_per_complex_auprc": float(np.mean(values["per_complex_auprc"])),
        }'''
    residue_result_new = '''            "mean_per_complex_auprc": float(np.mean(values["per_complex_auprc"])),
            "per_complex_auprc_by_instance": values["per_complex_auprc_by_instance"],
            "eligible_complex_count": len(values["per_complex_auprc"]),
            "total_complex_count": int(values["total_complex_count"]),
            "zero_positive_complex_count": int(values["total_complex_count"]) - len(values["per_complex_auprc"]),
        }'''
    replacements = (
        (initialization_old, initialization_new, "initialization"),
        (contact_old, contact_new, "contact eligible-complex"),
        (residue_old, residue_new, "residue eligible-complex"),
        (contact_result_old, contact_result_new, "contact result"),
        (residue_result_old, residue_result_new, "residue result"),
    )
    for old, new, name in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"training evaluator {name} patch anchor differs")
        source = source.replace(old, new)
    return source


def install_metric_patch(source_root: Path) -> None:
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts"))
    import train_framecontact as training
    source = patched_evaluate_source(inspect.getsource(training._evaluate))
    namespace = training.__dict__
    exec(compile(source, str(source_root / "scripts/train_framecontact.py"), "exec"), namespace)


def main() -> None:
    contact_recovery_path = Path(take_option("--contact-recovery-protocol"))
    contact_recovery_sha = take_option("--contact-recovery-protocol-sha256")
    metric_recovery_path = Path(take_option("--metric-recovery-protocol"))
    metric_recovery_sha = take_option("--metric-recovery-protocol-sha256")
    original_path = Path(take_option("--pinned-evaluator"))
    if sha256_file(contact_recovery_path) != contact_recovery_sha or contact_recovery_sha != CONTACT_RECOVERY_SHA256:
        raise ValueError("contact recovery SHA differs")
    if sha256_file(metric_recovery_path) != metric_recovery_sha:
        raise ValueError("metric recovery SHA differs")
    contact_recovery = json.loads(contact_recovery_path.read_text(encoding="utf-8"))
    metric_recovery = json.loads(metric_recovery_path.read_text(encoding="utf-8"))
    adapter_root = Path(contact_recovery["adapter_contact_root"])
    if (
        contact_recovery.get("status") != "FROZEN_AFTER_TEST_CONTACT_AUTHORITY_ADAPTER_FOR_SAME_OPENING"
        or metric_recovery.get("status") != "FROZEN_AFTER_ZERO_POSITIVE_METRIC_FAILURE_FOR_SAME_OPENING"
        or metric_recovery.get("contact_authority_evaluation_recovery_sha256") != contact_recovery_sha
        or metric_recovery.get("pooled_metrics_retain_all_samples") is not True
        or metric_recovery.get("selection_performed_on_test") is not False
        or sha256_file(adapter_root / "CONTACT_CACHE_MANIFEST.json") != contact_recovery["adapter_contact_manifest_sha256"]
    ):
        raise ValueError("metric execution recovery differs")
    if sha256_file(original_path) != PINNED_EVALUATOR_SHA256:
        raise ValueError("pinned evaluator SHA differs")
    source_root = Path(option_value("--source-root"))
    install_metric_patch(source_root)

    source = original_path.read_text(encoding="utf-8")
    split_old = '''    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root, args.embedding_root, None,
        split_role="test", split_path=args.split,
        split_sha256=protocol["data"]["split_sha256"],
    )'''
    split_new = '''    test_split_path = args.structure_root / "TEST_SPLIT_VIEW.csv"
    structure_authority = json.loads(structure_manifest.read_text(encoding="utf-8"))["authority"]
    test_split_sha256 = structure_authority["test_split_view_sha256"]
    require(test_split_path, test_split_sha256)
    if test_split_sha256 != contact_authority_recovery["test_split_view_sha256"]:
        raise ValueError("test split execution-recovery binding differs")
    base = Sabdab2StructureEmbeddingDataset(
        args.structure_root, args.embedding_root, None,
        split_role="test", split_path=test_split_path,
        split_sha256=test_split_sha256,
    )'''
    contact_old = '        (contact_manifest, protocol["data"]["contact_manifest_sha256"]),'
    contact_new = '        (contact_manifest, contact_authority_recovery["adapter_contact_manifest_sha256"]),'
    if source.count(split_old) != 1 or source.count(contact_old) != 1:
        raise RuntimeError("pinned evaluator data-authority patch anchor differs")
    source = source.replace(split_old, split_new).replace(contact_old, contact_new)
    audit_anchor = '            "cumulative_test_access_ordinal": 3,'
    source = source.replace(
        audit_anchor,
        audit_anchor
        + '\n            "contact_authority_execution_recovery_sha256": contact_authority_recovery_sha,'
        + '\n            "zero_positive_metric_recovery_sha256": zero_positive_metric_recovery_sha,'
        + '\n            "adapter_contact_manifest_sha256": contact_authority_recovery["adapter_contact_manifest_sha256"],',
    )
    receipt_anchor = '''        "selection_performed_on_test": False, "fresh_blind_test": False,
        "cumulative_test_access_ordinal": 3,'''
    source = source.replace(
        receipt_anchor,
        receipt_anchor
        + '\n        "contact_authority_execution_recovery_sha256": contact_authority_recovery_sha,'
        + '\n        "zero_positive_metric_recovery_sha256": zero_positive_metric_recovery_sha,'
        + '\n        "adapter_contact_manifest_sha256": contact_authority_recovery["adapter_contact_manifest_sha256"],',
    )
    module_name = "abf30b_exposed_test_zero_positive_recovery_component"
    module = types.ModuleType(module_name)
    module.__file__ = str(original_path)
    module.__package__ = None
    module.__dict__["contact_authority_recovery"] = contact_recovery
    module.__dict__["contact_authority_recovery_sha"] = contact_recovery_sha
    module.__dict__["zero_positive_metric_recovery_sha"] = metric_recovery_sha
    sys.modules[module_name] = module
    try:
        exec(compile(source, str(original_path), "exec"), module.__dict__)
        module.main()
    finally:
        sys.modules.pop(module_name, None)


if __name__ == "__main__":
    main()
