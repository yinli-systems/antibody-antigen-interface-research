#!/usr/bin/env python3
"""Contract tests for the staged SAbDab2 SPPIDER-seq adapter."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sabdab2_sppider_seq import (
    FORMAT_VERSION,
    PREDICTION_SCHEMA,
    BenchmarkLayout,
    atomic_write_npz,
    binary_metrics,
    build_partner_context_mapping,
    build_role_split_maps,
    command_assemble,
    command_ensemble_inference,
    command_prepare,
    load_prepared_instances,
)
from sppider_seq_core import (
    ChunkwiseInteractionModel,
    atomic_torch_save,
    merge_chunk_logits_to_full,
    safe_torch_load,
)
from sppider_seq_research import exact_threshold_optimum


class AdapterContractTest(unittest.TestCase):
    """Exercise authoritative labels, grouped splits, and final projections."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "delivery"
        self.frozen = (
            self.root / "data/nfs/yangwuyu/baseline-eval/frozen/sabdab2"
        )
        self.frozen.mkdir(parents=True)
        self.prepare_dir = Path(self.temporary.name) / "prepared"
        self.inference_dir = Path(self.temporary.name) / "inference"
        self.output_dir = Path(self.temporary.name) / "assembled"

        self.instances = pd.DataFrame(
            [
                {"instance_id": "train_a", "ab_seq": "ACD", "ag_seq": "FG"},
                {"instance_id": "train_b", "ab_seq": "HIK", "ag_seq": "LM"},
                {"instance_id": "train_c", "ab_seq": "NPQ", "ag_seq": "RS"},
                {"instance_id": "test_a", "ab_seq": "TVW", "ag_seq": "YA"},
            ]
        )
        self.instances.to_parquet(self.frozen / "instances.parquet", index=False)
        pair_rows = {
            "train_a": [(0, 0, False), (0, 1, True), (1, 0, False)],
            "train_b": [(0, 0, True), (1, 1, False), (2, 0, False)],
            "train_c": [(0, 0, False), (1, 1, True), (2, 0, True)],
            "test_a": [
                (0, 0, False),
                (0, 1, True),
                (1, 0, True),
                (2, 1, False),
            ],
        }
        pair_schema = pa.schema(
            [
                pa.field("instance_id", pa.string(), nullable=False),
                pa.field("ab_res_idx", pa.int64(), nullable=False),
                pa.field("ag_res_idx", pa.int64(), nullable=False),
                pa.field("label", pa.bool_(), nullable=False),
            ]
        )
        writer = pq.ParquetWriter(self.frozen / "pairs.parquet", pair_schema)
        for instance_id, rows in pair_rows.items():
            writer.write_table(
                pa.Table.from_pydict(
                    {
                        "instance_id": [instance_id] * len(rows),
                        "ab_res_idx": [row[0] for row in rows],
                        "ag_res_idx": [row[1] for row in rows],
                        "label": [row[2] for row in rows],
                    },
                    schema=pair_schema,
                )
            )
        writer.close()
        (self.frozen / "splits.json").write_text(
            json.dumps(
                {
                    "sabdab2": {
                        "train": ["train_a", "train_b", "train_c"],
                        "test": ["test_a"],
                    }
                }
            ),
            encoding="utf-8",
        )
        (self.frozen / "contamination.json").write_text("{}\n", encoding="utf-8")
        (self.frozen / "MANIFEST.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(self) -> pd.DataFrame:
        """Run preparation with the synthetic manifest hash exemption."""
        command_prepare(
            argparse.Namespace(
                benchmark_root=self.root,
                output_dir=self.prepare_dir,
                canonical_manifest=None,
                validation_fraction=0.34,
                seed=7,
                resume=True,
                skip_manifest_hash=True,
            )
        )
        return load_prepared_instances(self.prepare_dir)

    def test_grouped_role_splits_are_disjoint(self) -> None:
        """Repeated focal sequences must not cross train and validation."""
        frame = self.instances.set_index("instance_id")
        frame.loc["train_b", "ab_seq"] = frame.loc["train_a", "ab_seq"]
        frame.loc["train_b", "ag_seq"] = frame.loc["train_a", "ag_seq"]
        assignments = build_role_split_maps(
            frame,
            {"train": ["train_a", "train_b", "train_c"], "test": ["test_a"]},
            validation_fraction=0.34,
            seed=7,
        )
        for role in ("paratope", "epitope"):
            self.assertEqual(assignments[role]["train_a"], assignments[role]["train_b"])
            self.assertEqual(assignments[role]["test_a"], "test")

    def test_component_splits_isolate_canonical_groups_and_queries(self) -> None:
        """Canonical clusters and repeated focal queries must remain indivisible."""
        frame = self.instances.set_index("instance_id")
        frame.loc["train_d"] = {"ab_seq": "CEF", "ag_seq": "GHI"}
        frame.loc["train_c", "ab_seq"] = frame.loc["train_b", "ab_seq"]
        assignments = build_role_split_maps(
            frame,
            {
                "train": ["train_a", "train_b", "train_c", "train_d"],
                "test": ["test_a"],
            },
            validation_fraction=0.34,
            seed=7,
            validation_group_by_instance={
                "train_a": "cluster_1",
                "train_b": "cluster_1",
                "train_c": "cluster_2",
                "train_d": "cluster_3",
            },
        )
        for role in ("paratope", "epitope"):
            self.assertEqual(assignments[role]["train_a"], assignments[role]["train_b"])
        self.assertEqual(assignments["paratope"]["train_b"], assignments["paratope"]["train_c"])

    def test_prepare_projects_any_positive_contact(self) -> None:
        """Candidate contact labels must project to complete binary residue axes."""
        prepared = self.prepare()
        self.assertEqual(set(prepared.index), set(self.instances["instance_id"]))
        with np.load(self.prepare_dir / "labels/test_a.npz", allow_pickle=False) as artifact:
            np.testing.assert_array_equal(artifact["ab_labels"], [1, 1, 0])
            np.testing.assert_array_equal(artifact["ag_labels"], [1, 1])
        self.assertEqual(int(prepared.at["test_a", "positive_pair_count"]), 2)
        self.assertEqual(prepared.at["test_a", "paratope_split"], "test")
        self.assertEqual(prepared.at["test_a", "epitope_split"], "test")

    def test_assemble_emits_exact_epitope_projection_keys(self) -> None:
        """Assembly must preserve candidate keys and index antigen scores exactly."""
        self.prepare()
        artifact_dir = self.inference_dir / "instances"
        scores = np.asarray([0.2, 0.8], dtype=np.float32)
        labels = np.asarray([1, 1], dtype=np.uint8)
        atomic_write_npz(
            artifact_dir / "test_a.npz",
            format_version=np.asarray(FORMAT_VERSION),
            instance_id=np.asarray("test_a"),
            role=np.asarray("epitope"),
            checkpoint_sha256=np.asarray("a" * 64),
            scores=scores,
            labels=labels,
        )
        command_assemble(
            argparse.Namespace(
                benchmark_root=self.root,
                prepare_dir=self.prepare_dir,
                inference_dir=self.inference_dir,
                output_dir=self.output_dir,
                role="epitope",
                threshold=0.5,
                skip_manifest_hash=True,
                allow_nonstandard_test_count=True,
            )
        )
        predictions = pq.read_table(self.output_dir / "predictions.parquet")
        self.assertEqual(predictions.schema, PREDICTION_SCHEMA)
        self.assertEqual(predictions.num_rows, 4)
        np.testing.assert_array_equal(
            predictions["score"].to_numpy(),
            np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float32),
        )
        verification = json.loads(
            (self.output_dir / "verification.json").read_text(encoding="utf-8")
        )
        self.assertTrue(verification["key_equality"])
        self.assertEqual(verification["row_count"], 4)

    def test_inference_ensemble_preserves_keys_labels_and_mean_scores(self) -> None:
        """Fixed inference ensembles must average only aligned probability axes."""
        self.prepare()
        input_dirs = [Path(self.temporary.name) / "model_a", Path(self.temporary.name) / "model_b"]
        for input_dir, checkpoint, scores in zip(
            input_dirs,
            ("a" * 64, "b" * 64),
            ([0.2, 0.8], [0.4, 0.6]),
            strict=True,
        ):
            atomic_write_npz(
                input_dir / "instances/test_a.npz",
                format_version=np.asarray(FORMAT_VERSION),
                instance_id=np.asarray("test_a"),
                role=np.asarray("epitope"),
                checkpoint_sha256=np.asarray(checkpoint),
                scores=np.asarray(scores, dtype=np.float32),
                labels=np.asarray([1, 1], dtype=np.uint8),
            )
        ensemble_dir = Path(self.temporary.name) / "ensemble"
        command_ensemble_inference(
            argparse.Namespace(
                prepare_dir=self.prepare_dir,
                input_dirs=input_dirs,
                weights=None,
                output_dir=ensemble_dir,
                role="epitope",
            )
        )
        with np.load(ensemble_dir / "instances/test_a.npz", allow_pickle=False) as artifact:
            np.testing.assert_allclose(artifact["scores"], [0.3, 0.7])
            np.testing.assert_array_equal(artifact["labels"], [1, 1])

    def test_binary_metrics_are_json_safe_for_single_class(self) -> None:
        """Undefined discrimination metrics must use null rather than NaN."""
        metrics = binary_metrics(
            np.asarray([1, 1], dtype=np.uint8),
            np.asarray([0.2, 0.8], dtype=np.float32),
            threshold=0.5,
        )
        self.assertIsNone(metrics["auroc"])
        self.assertIsNone(metrics["auprc"])
        json.dumps(metrics, allow_nan=False)

    def test_exact_threshold_selection_recovers_perfect_separator(self) -> None:
        """Validation threshold selection must test observed score boundaries exactly."""
        labels = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
        for objective in ("mcc", "f1"):
            threshold, value = exact_threshold_optimum(labels, scores, objective)
            self.assertAlmostEqual(threshold, 0.8)
            self.assertAlmostEqual(value, 1.0)

    def test_partner_permutation_is_deterministic_and_sequence_deranged(self) -> None:
        """A partner control must change every context sequence reproducibly."""
        frame = pd.DataFrame(
            {
                "ab_sequence_sha256": ["ab1", "ab2", "ab3", "ab4"],
                "ag_sequence_sha256": ["ag1", "ag2", "ag3", "ag4"],
            },
            index=["i1", "i2", "i3", "i4"],
        )
        first = build_partner_context_mapping(frame, "paratope", "partner_permuted", 17)
        second = build_partner_context_mapping(frame, "paratope", "partner_permuted", 17)
        self.assertEqual(first, second)
        for instance_id, row in frame.iterrows():
            self.assertNotEqual(first[instance_id], row["ag_sequence_sha256"])


class ReleasedModelCompatibilityTest(unittest.TestCase):
    """Verify the released architecture, gradient path, and optional checkpoints."""

    def test_checkpoint_round_trip_accepts_torch_version_metadata(self) -> None:
        """Resumable checkpoints must retain safe loading across PyTorch 2.6+."""
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            atomic_torch_save(
                {
                    "torch_version": torch.__version__,
                    "tensor": torch.asarray([1.0, 2.0]),
                },
                path,
            )
            loaded = safe_torch_load(path)
        self.assertEqual(str(loaded["torch_version"]), str(torch.__version__))
        torch.testing.assert_close(loaded["tensor"], torch.asarray([1.0, 2.0]))

    def test_released_parameter_count_and_overlap_gradients(self) -> None:
        """The model must retain 8.2M parameters and differentiable max merging."""
        import torch

        model = ChunkwiseInteractionModel()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 8_202_241)
        first = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        second = torch.tensor([4.0, 1.0, 5.0], requires_grad=True)
        merged = merge_chunk_logits_to_full(
            [first, second],
            [0, 2],
            full_length=5,
            device=torch.device("cpu"),
        )
        merged.sum().backward()
        self.assertIsNotNone(first.grad)
        self.assertIsNotNone(second.grad)
        self.assertGreater(float(first.grad.abs().sum() + second.grad.abs().sum()), 0.0)

    def test_official_checkpoints_strict_load_when_available(self) -> None:
        """Both released bare state dicts must load without missing or extra keys."""
        model_dir = os.environ.get("SPPIDER_OFFICIAL_MODEL_DIR")
        if not model_dir:
            self.skipTest("SPPIDER_OFFICIAL_MODEL_DIR is not set")
        paths = sorted(Path(model_dir).glob("crossattn_*_best.pt"))
        self.assertEqual(len(paths), 2)
        model = ChunkwiseInteractionModel()
        for path in paths:
            model.load_state_dict(safe_torch_load(path), strict=True)


if __name__ == "__main__":
    unittest.main()
