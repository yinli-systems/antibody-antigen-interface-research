"""Check that mismatched handoff inputs fail before creating bundle files."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import finalize_test455_loader_bundle_v1 as finalize


class FinalizeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.key = "a" * 64
        (self.bundle / "aa").mkdir()
        (self.bundle / "aa" / f"{self.key}.pt").write_bytes(b"path-only fixture")
        self.ids = ["instance-a", "instance-b"]
        id_bytes = ("\n".join(self.ids) + "\n").encode()
        (self.bundle / "REQUESTED_INSTANCE_IDS.txt").write_bytes(id_bytes)
        self.mapping = [{
            "instance_id": item,
            "antibody_sequence_sha256": self.key, "antibody_length": 10,
            "antibody_artifact_path": f"aa/{self.key}.pt",
            "antigen_sequence_sha256": self.key, "antigen_length": 10,
            "antigen_artifact_path": f"aa/{self.key}.pt",
        } for item in self.ids]
        self.write_mapping()
        manifest = self.bundle / "SABDAB2_ESM_EMBEDDING_MANIFEST.json"
        manifest.write_text("{}\n")
        self.export = {
            "status": "PASS_MODEL_READY_ESM2_SUBSET_EXPORT",
            "requested_instance_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
            "counts": {"unique_sequences": 1},
            "instance_embedding_map_sha256": finalize.sha256_file(self.bundle / "INSTANCE_EMBEDDING_MAP.csv"),
            "subset_embedding_manifest_sha256": finalize.sha256_file(manifest),
        }
        self.write_receipt()
        self.split = self.root / "split.json"
        self.split.write_text(json.dumps({"splits": {"train": [], "validation": [], "test": self.ids}}))
        self.prepared = self.root / "prepared.parquet"
        pd.DataFrame([{
            "instance_id": item, "ab_sequence_sha256": self.key, "ag_sequence_sha256": self.key,
            "ab_length": 10, "ag_length": 10, "frozen_split": "test",
            "unrequested_label_column": "not to be copied",
        } for item in self.ids]).to_parquet(self.prepared)
        self.args = SimpleNamespace(handoff_root=self.bundle, split_json=self.split,
                                    prepared_instances=self.prepared, expected_instance_count=2,
                                    expected_id_set_sha256=self.export["requested_instance_ids_sha256"])

    def write_mapping(self):
        with (self.bundle / "INSTANCE_EMBEDDING_MAP.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.mapping[0]))
            writer.writeheader()
            writer.writerows(self.mapping)

    def write_receipt(self):
        (self.bundle / "EXPORT_RECEIPT.json").write_text(json.dumps(self.export))

    def run_finalizer(self):
        with patch.object(finalize, "parse_args", return_value=self.args), patch("builtins.print"):
            finalize.main()

    def assert_no_outputs(self):
        for name in ("SABDAB2_455_SPLIT.json", "SABDAB2_TEST455_INSTANCE_INDEX.parquet",
                     "LOADER_PATHS.json", "README.md", "CACHE_BUNDLE_RECEIPT.json"):
            self.assertFalse((self.bundle / name).exists(), name)

    def test_valid_bundle_omits_unrequested_columns_and_refuses_overwrite(self):
        self.run_finalizer()
        frame = pd.read_parquet(self.bundle / "SABDAB2_TEST455_INSTANCE_INDEX.parquet")
        self.assertEqual(frame.instance_id.tolist(), self.ids)
        self.assertNotIn("unrequested_label_column", frame.columns)
        with self.assertRaises(FileExistsError):
            self.run_finalizer()

    def test_mismatched_sequence_axis_fails_before_any_output(self):
        self.mapping[0]["antibody_length"] = 11
        self.write_mapping()
        with self.assertRaisesRegex(ValueError, "index and embedding map differ"):
            self.run_finalizer()
        self.assert_no_outputs()

    def test_stale_export_receipt_fails_before_any_output(self):
        self.export["requested_instance_ids_sha256"] = "0" * 64
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "Original export receipt"):
            self.run_finalizer()
        self.assert_no_outputs()


if __name__ == "__main__":
    unittest.main()
