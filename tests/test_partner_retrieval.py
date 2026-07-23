#!/usr/bin/env python3
"""Focused tests for the standalone partner-retrieval training pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sabdab2_partner_retrieval import (
    PairRecord,
    build_positive_mask,
    cosine_warmup_multiplier,
    deduplicate_exact_pairs,
    directional_retrieval_metrics,
    embedding_corpus_fingerprint,
    length_bucket_batches,
    validate_config_integrity,
)


def _record(
    identifier: str,
    antibody: str,
    antigen: str,
    antibody_length: int = 120,
    antigen_length: int = 400,
) -> PairRecord:
    return PairRecord(
        identifier,
        antibody,
        antigen,
        antibody_length,
        antigen_length,
    )


class DataProtocolTests(unittest.TestCase):
    def test_exact_pair_deduplication_is_deterministic(self) -> None:
        records = [
            _record("z", "ab-a", "ag-a"),
            _record("a", "ab-a", "ag-a"),
            _record("b", "ab-b", "ag-a"),
        ]
        deduplicated, removed = deduplicate_exact_pairs(records)
        self.assertEqual(removed, 1)
        self.assertEqual({value.instance_id for value in deduplicated}, {"a", "b"})

    def test_length_bucketing_is_reproducible_and_lossless(self) -> None:
        records = [
            _record(
                f"id-{index}",
                f"ab-{index}",
                f"ag-{index}",
                100 + index * 3,
                300 + index * 17,
            )
            for index in range(11)
        ]
        first = length_bucket_batches(records, 4, 64, seed=7, epoch=3)
        second = length_bucket_batches(records, 4, 64, seed=7, epoch=3)
        self.assertEqual(first, second)
        self.assertTrue(all(len(batch) >= 2 for batch in first))
        observed = [value.instance_id for batch in first for value in batch]
        self.assertEqual(sorted(observed), sorted(value.instance_id for value in records))

    def test_warmup_cosine_schedule_reaches_bounds(self) -> None:
        values = [cosine_warmup_multiplier(step, 100, 10, 0.01) for step in range(100)]
        self.assertAlmostEqual(values[9], 1.0)
        self.assertAlmostEqual(values[-1], 0.01)
        self.assertTrue(all(0.01 <= value <= 1.0 for value in values))

    def test_config_integrity_rejects_tampering(self) -> None:
        from sabdab2_partner_retrieval import canonical_sha256

        config = {"model": {"dropout": 0.1}}
        config["config_sha256"] = canonical_sha256(config)
        validate_config_integrity(config)
        config["model"]["dropout"] = 0.9
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_config_integrity(config)

    def test_embedding_fingerprint_detects_content_changes(self) -> None:
        first_key = hashlib.sha256(b"first").hexdigest()
        second_key = hashlib.sha256(b"second").hexdigest()
        records = [_record("pair", first_key, second_key)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, payload in ((first_key, b"alpha"), (second_key, b"beta")):
                path = root / key[:2] / f"{key}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            original = embedding_corpus_fingerprint(root, records, "unit_test")
            (root / second_key[:2] / f"{second_key}.pt").write_bytes(b"changed")
            changed = embedding_corpus_fingerprint(root, records, "unit_test")
        self.assertNotEqual(
            original["content_manifest_sha256"],
            changed["content_manifest_sha256"],
        )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class RetrievalMetricTests(unittest.TestCase):
    def test_scheduler_state_resume_matches_uninterrupted_schedule(self) -> None:
        import torch

        from sabdab2_partner_retrieval import _scheduler

        reference_parameter = torch.nn.Parameter(torch.ones(()))
        reference_optimizer = torch.optim.AdamW(
            [reference_parameter], lr=1e-3
        )
        reference_scheduler = _scheduler(reference_optimizer, 12, 0.25, 0.01)
        for _ in range(5):
            reference_optimizer.step()
            reference_scheduler.step()
        optimizer_state = reference_optimizer.state_dict()
        scheduler_state = reference_scheduler.state_dict()

        resumed_parameter = torch.nn.Parameter(torch.ones(()))
        resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-3)
        resumed_scheduler = _scheduler(resumed_optimizer, 12, 0.25, 0.01)
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        for _ in range(4):
            reference_optimizer.step()
            reference_scheduler.step()
            resumed_optimizer.step()
            resumed_scheduler.step()
            self.assertAlmostEqual(
                reference_optimizer.param_groups[0]["lr"],
                resumed_optimizer.param_groups[0]["lr"],
            )

    def test_multi_positive_mask_preserves_all_observed_pairs(self) -> None:
        import torch

        mask = build_positive_mask(
            ["ab-a", "ab-b"],
            ["ag-a", "ag-b"],
            {("ab-a", "ag-a"), ("ab-a", "ag-b"), ("ab-b", "ag-b")},
            device="cpu",
        )
        expected = torch.tensor([[True, True], [False, True]])
        torch.testing.assert_close(mask, expected)

    def test_full_library_metrics_use_best_known_positive_rank(self) -> None:
        import torch

        similarities = torch.tensor(
            [
                [0.1, 0.8, 0.9],
                [0.7, 0.6, 0.5],
            ]
        )
        positives = torch.tensor(
            [
                [False, True, True],
                [False, True, False],
            ]
        )
        metrics = directional_retrieval_metrics(similarities, positives, (1, 2))
        self.assertAlmostEqual(metrics["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 0.75)


@unittest.skipUnless(
    importlib.util.find_spec("torch")
    and importlib.util.find_spec("pandas")
    and importlib.util.find_spec("pyarrow"),
    "PyTorch, pandas, and PyArrow are required",
)
class EndToEndPipelineTests(unittest.TestCase):
    def test_train_resume_validation_and_test_lock(self) -> None:
        import pandas as pd
        import torch

        from interaction_first.runtime import embedding_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare_dir = root / "prepared"
            embedding_dir = root / "embeddings"
            output_dir = root / "run"
            prepare_dir.mkdir()
            records = []
            splits = {"train": [], "validation": [], "test": []}
            split_sizes = {"train": 6, "validation": 4, "test": 2}
            instance_index = 0
            for split_name, split_size in split_sizes.items():
                for local_index in range(split_size):
                    instance_id = f"{split_name}-{local_index}"
                    antibody_key = hashlib.sha256(
                        f"antibody:{instance_id}".encode("ascii")
                    ).hexdigest()
                    antigen_key = hashlib.sha256(
                        f"antigen:{instance_id}".encode("ascii")
                    ).hexdigest()
                    antibody_length = 4 + instance_index % 3
                    antigen_length = 5 + instance_index % 4
                    records.append(
                        {
                            "instance_id": instance_id,
                            "ab_sequence_sha256": antibody_key,
                            "ag_sequence_sha256": antigen_key,
                            "ab_length": antibody_length,
                            "ag_length": antigen_length,
                        }
                    )
                    splits[split_name].append(instance_id)
                    for key, length in (
                        (antibody_key, antibody_length),
                        (antigen_key, antigen_length),
                    ):
                        path = embedding_path(embedding_dir, key)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        generator = torch.Generator().manual_seed(instance_index)
                        torch.save(
                            {
                                "format_version": "sppider_seq_embedding_v1",
                                "sequence_sha256": key,
                                "sequence_length": length,
                                "model_name": "synthetic",
                                "starts": [0],
                                "ends": [length],
                                "chunks": [
                                    torch.randn(length, 1280, generator=generator)
                                ],
                            },
                            path,
                        )
                    instance_index += 1
            pd.DataFrame(records).to_parquet(
                prepare_dir / "instances.parquet", index=False
            )
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "format_version": "sabdab2_interaction_first_v1",
                        "intended_use": "development_only",
                        "splits": splits,
                    }
                ),
                encoding="utf-8",
            )

            script = Path(__file__).with_name("sabdab2_partner_retrieval.py")
            base_command = [
                sys.executable,
                str(script),
                "train",
                "--prepare-dir",
                str(prepare_dir),
                "--embedding-dir",
                str(embedding_dir),
                "--split-manifest",
                str(split_path),
                "--output-dir",
                str(output_dir),
                "--embedding-dim",
                "8",
                "--attention-hidden-dim",
                "8",
                "--projection-hidden-dim",
                "16",
                "--dropout",
                "0",
                "--batch-size",
                "3",
                "--validation-batch-size",
                "4",
                "--bucket-width",
                "16",
                "--max-epochs",
                "1",
                "--patience",
                "1",
                "--min-train-pairs",
                "2",
                "--min-validation-pairs",
                "2",
                "--embedding-cache-gib",
                "0",
                "--device",
                "cpu",
                "--log-every",
                "1",
            ]
            environment = {**os.environ, "PYTHONPATH": str(script.parent)}
            first = subprocess.run(
                [*base_command, "--no-resume"],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.returncode, 0, msg=first.stderr)
            resumed = subprocess.run(
                [*base_command, "--resume"],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stderr)
            self.assertTrue((output_dir / "best.pt").is_file())

            evaluation_path = root / "validation.json"
            evaluate = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "evaluate",
                    "--prepare-dir",
                    str(prepare_dir),
                    "--embedding-dir",
                    str(embedding_dir),
                    "--split-manifest",
                    str(split_path),
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    "--output",
                    str(evaluation_path),
                    "--split",
                    "validation",
                    "--device",
                    "cpu",
                    "--embedding-cache-gib",
                    "0",
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(evaluate.returncode, 0, msg=evaluate.stderr)
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            self.assertIn("bidirectional_mrr", evaluation["metrics"])

            tampered_key = records[0]["ab_sequence_sha256"]
            tampered_path = embedding_path(embedding_dir, tampered_key)
            tampered_path.write_bytes(tampered_path.read_bytes() + b"tampered")
            rejected_embedding_change = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "evaluate",
                    "--prepare-dir",
                    str(prepare_dir),
                    "--embedding-dir",
                    str(embedding_dir),
                    "--split-manifest",
                    str(split_path),
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    "--output",
                    str(root / "tampered-validation.json"),
                    "--split",
                    "validation",
                    "--device",
                    "cpu",
                    "--embedding-cache-gib",
                    "0",
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected_embedding_change.returncode, 0)
            self.assertIn(
                "Frozen embedding corpus changed",
                rejected_embedding_change.stderr,
            )

            locked_test = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "evaluate",
                    "--prepare-dir",
                    str(prepare_dir),
                    "--embedding-dir",
                    str(embedding_dir),
                    "--split-manifest",
                    str(split_path),
                    "--checkpoint",
                    str(output_dir / "best.pt"),
                    "--output",
                    str(root / "test.json"),
                    "--split",
                    "test",
                    "--device",
                    "cpu",
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(locked_test.returncode, 0)
            self.assertIn("Frozen test evaluation is locked", locked_test.stderr)


if __name__ == "__main__":
    unittest.main()
