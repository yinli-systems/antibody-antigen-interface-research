"""Small regression checks for cleanup verification; standard library only."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import verify_cleanup as v


class CleanupTests(unittest.TestCase):
    def test_exact_bytes_required(self):
        v.validate_bytes(b"source", hashlib.sha256(b"source").hexdigest(), "sample")
        with self.assertRaises(ValueError):
            v.validate_bytes(b"changed", hashlib.sha256(b"source").hexdigest(), "sample")

    def test_local_only_evidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not in Git"):
            v.verify_declared([{"path": "x", "sha256": "0" * 64}], "path", set())

    def test_index_bytes_checked_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x").write_bytes(b"good")
            row = {"path": "x", "sha256": hashlib.sha256(b"good").hexdigest()}
            with patch.object(v, "ROOT", root), patch.object(v, "git", return_value=b"bad"):
                with self.assertRaisesRegex(ValueError, "Git index"):
                    v.verify_declared([row], "path", {"x"})

    def test_index_and_worktree_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "x").write_bytes(b"good")
            row = {"path": "x", "sha256": hashlib.sha256(b"good").hexdigest()}
            with patch.object(v, "ROOT", root), patch.object(v, "git", return_value=b"good"):
                self.assertEqual(v.verify_declared([row], "path", {"x"}), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
