#!/usr/bin/env python3
"""Unit tests for the SAbDab2 AntiBERTy embedding pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sabdab2_antiberty_embed as module


def write_json(path: Path, value: object) -> None:
    """Write a compact JSON fixture."""
    path.write_text(json.dumps(value), encoding="utf-8")


class SplitSafetyTests(unittest.TestCase):
    """Verify split selection and the frozen-test interlock."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split_path = self.root / "split.json"
        write_json(
            self.split_path,
            {
                "intended_use": "development_only",
                "splits": {
                    "train": ["train_a", "train_b"],
                    "validation": ["validation_a"],
                    "test": ["test_a"],
                },
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_default_development_selection_excludes_test(self) -> None:
        selected, summary = module.load_split_selection(
            self.split_path,
            module.DEFAULT_SPLITS,
            allow_frozen_test=False,
        )
        self.assertEqual(
            selected,
            {
                "train_a": "train",
                "train_b": "train",
                "validation_a": "validation",
            },
        )
        self.assertFalse(summary["frozen_test_unlocked"])
        self.assertEqual(summary["all_split_counts"]["test"], 1)

    def test_test_selection_requires_explicit_unlock(self) -> None:
        with self.assertRaises(PermissionError):
            module.load_split_selection(
                self.split_path,
                ["test"],
                allow_frozen_test=False,
            )
        selected, summary = module.load_split_selection(
            self.split_path,
            ["test"],
            allow_frozen_test=True,
        )
        self.assertEqual(selected, {"test_a": "test"})
        self.assertTrue(summary["frozen_test_unlocked"])

    def test_overlap_is_rejected(self) -> None:
        write_json(
            self.split_path,
            {
                "splits": {
                    "train": ["shared"],
                    "validation": ["shared"],
                    "test": [],
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            module.load_split_selection(
                self.split_path,
                module.DEFAULT_SPLITS,
                allow_frozen_test=False,
            )


class CanonicalSelectionTests(unittest.TestCase):
    """Verify sequence filtering, validation, and exact deduplication."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "canonical.csv"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_unselected_sequences_are_not_parsed(self) -> None:
        self.manifest.write_text(
            "INSTANCE,vh_seq,vl_seq,ag_seq\n"
            "train_a,ACD,EFG,NOT_USED\n"
            "test_a,INVALID!,ALSO_INVALID,NOT_USED\n",
            encoding="utf-8",
        )
        records = module.load_selected_antibodies(
            self.manifest,
            {"train_a": "train"},
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].vh_sequence, "ACD")
        self.assertEqual(records[0].vl_sequence, "EFG")

    def test_empty_light_chain_is_supported(self) -> None:
        self.manifest.write_text(
            "INSTANCE,vh_seq,vl_seq\nsolo,ACDE,\n",
            encoding="utf-8",
        )
        records = module.load_selected_antibodies(
            self.manifest,
            {"solo": "validation"},
        )
        self.assertEqual(records[0].vl_length, 0)
        self.assertEqual(records[0].total_length, 4)

    def test_invalid_selected_sequence_is_rejected(self) -> None:
        self.manifest.write_text(
            "INSTANCE,vh_seq,vl_seq\nbad,ACX,EFG\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "invalid residue"):
            module.load_selected_antibodies(self.manifest, {"bad": "train"})

    def test_inconsistent_chain_boundary_deduplication_is_rejected(self) -> None:
        first = module.AntibodyRecord(
            "one", "train", "AC", "DE", module.sequence_sha256("AC", "DE")
        )
        second = module.AntibodyRecord(
            "two", "validation", "ACD", "E", module.sequence_sha256("ACD", "E")
        )
        self.assertEqual(first.sequence_key, second.sequence_key)
        with self.assertRaisesRegex(ValueError, "inconsistent VH/VL"):
            module.deduplicate_antibodies([first, second])

    def test_prepared_metadata_must_match_digest_and_length(self) -> None:
        record = module.AntibodyRecord(
            "one", "train", "AC", "DE", module.sequence_sha256("AC", "DE")
        )
        module.validate_instance_metadata_rows(
            [record],
            {"one": (record.sequence_key, 4)},
        )
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            module.validate_instance_metadata_rows(
                [record],
                {"one": ("0" * 64, 4)},
            )
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            module.validate_instance_metadata_rows(
                [record],
                {"one": (record.sequence_key, 3)},
            )


class ContractTests(unittest.TestCase):
    """Verify immutable output-root and artifact-path behavior."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_contract_requires_validated_resume(self) -> None:
        contract = {"format_version": module.CONTRACT_FORMAT_VERSION, "value": 1}
        digest = module.initialize_output_root(self.root, contract, resume=False)
        self.assertEqual(digest, module.sha256_bytes(module.canonical_json_bytes(contract)))
        with self.assertRaises(FileExistsError):
            module.initialize_output_root(self.root, contract, resume=False)
        self.assertEqual(
            module.initialize_output_root(self.root, contract, resume=True),
            digest,
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            module.initialize_output_root(
                self.root,
                {"format_version": module.CONTRACT_FORMAT_VERSION, "value": 2},
                resume=True,
            )

    def test_uncontracted_output_is_rejected(self) -> None:
        self.root.mkdir(parents=True)
        (self.root / "orphan.pt").write_bytes(b"orphan")
        with self.assertRaisesRegex(FileExistsError, "without an immutable"):
            module.initialize_output_root(self.root, {"value": 1}, resume=True)

    def test_artifact_path_is_sharded_by_digest(self) -> None:
        digest = "ab" + "0" * 62
        self.assertEqual(
            module.artifact_path(self.root, digest),
            self.root / "ab" / f"{digest}.pt",
        )

    def test_atomic_metadata_creation_never_overwrites(self) -> None:
        target = self.root / "immutable.json"
        module.atomic_write_bytes(target, b"first")
        with self.assertRaises(FileExistsError):
            module.atomic_write_bytes(target, b"second")
        self.assertEqual(target.read_bytes(), b"first")


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TensorContractTests(unittest.TestCase):
    """Verify tensor artifact validation without loading AntiBERTy weights."""

    def test_valid_payload_and_nonfinite_rejection(self) -> None:
        import torch

        record = module.AntibodyRecord(
            "one", "train", "AC", "DE", module.sequence_sha256("AC", "DE")
        )
        model = module.ModelInfo(
            model_name=module.MODEL_NAME,
            model_weights_path="weights.bin",
            model_weights_sha256="1" * 64,
            model_config_path=None,
            model_config_sha256=None,
            tokenizer_vocab_path="vocab.txt",
            tokenizer_vocab_sha256="2" * 64,
            antiberty_package_path="antiberty",
            package_versions={},
        )
        payload = {
            "format_version": module.ARTIFACT_FORMAT_VERSION,
            "sequence_key": record.sequence_key,
            "source_sequence_sha256": record.sequence_key,
            "embedding": torch.zeros((4, module.EXPECTED_EMBEDDING_DIM), dtype=torch.float32),
            "mask": torch.ones(4, dtype=torch.bool),
            "vh_length": 2,
            "vl_length": 2,
            "embedding_dim": module.EXPECTED_EMBEDDING_DIM,
            "model_name": module.MODEL_NAME,
            "model_weights_sha256": model.model_weights_sha256,
        }
        module.validate_embedding_artifact(payload, record, model, torch)
        payload["embedding"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.validate_embedding_artifact(payload, record, model, torch)

    def test_tensor_only_artifact_round_trip(self) -> None:
        import torch

        record = module.AntibodyRecord(
            "one", "train", "AC", "DE", module.sequence_sha256("AC", "DE")
        )
        model = module.ModelInfo(
            model_name=module.MODEL_NAME,
            model_weights_path="weights.bin",
            model_weights_sha256="1" * 64,
            model_config_path=None,
            model_config_sha256=None,
            tokenizer_vocab_path="vocab.txt",
            tokenizer_vocab_sha256="2" * 64,
            antiberty_package_path="antiberty",
            package_versions={},
        )
        payload = {
            "format_version": module.ARTIFACT_FORMAT_VERSION,
            "sequence_key": record.sequence_key,
            "source_sequence_sha256": record.sequence_key,
            "embedding": torch.zeros((4, module.EXPECTED_EMBEDDING_DIM), dtype=torch.float32),
            "mask": torch.ones(4, dtype=torch.bool),
            "vh_length": 2,
            "vl_length": 2,
            "embedding_dim": module.EXPECTED_EMBEDDING_DIM,
            "model_name": module.MODEL_NAME,
            "model_weights_sha256": model.model_weights_sha256,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.pt"
            torch.save(payload, path)
            module.load_and_validate_artifact(path, record, model, torch)


class ArgumentTests(unittest.TestCase):
    """Verify CLI defaults and digest validation."""

    def test_default_splits_are_development_only(self) -> None:
        args = module.parse_args(
            [
                "--canonical-manifest",
                "canonical.csv",
                "--instances-parquet",
                "instances.parquet",
                "--split-manifest",
                "split.json",
                "--output-root",
                "output",
            ]
        )
        self.assertEqual(args.splits, ["train", "validation"])
        self.assertFalse(args.allow_frozen_test)


if __name__ == "__main__":
    unittest.main()
