"""Tests for immutable exact-resume checkpoint generations."""

from __future__ import annotations

import hashlib
import importlib.util
import random
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from interaction_first.frontier_checkpoint import (
    CHECKPOINT_FILENAME,
    RECEIPT_FILENAME,
    capture_rng_state,
    create_checkpoint_generation,
    load_checkpoint_generation,
    restore_checkpoint_generation,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _digest(label: str) -> str:
    """Return a deterministic synthetic SHA-256 digest."""
    return hashlib.sha256(label.encode("ascii")).hexdigest()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ImmutableCheckpointTests(unittest.TestCase):
    """Verify exact restoration, provenance checks, and tamper rejection."""

    def _components(self) -> tuple[object, object, object]:
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.GELU(),
            torch.nn.Linear(4, 2),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.5,
        )
        sample = torch.tensor([[0.1, 0.2, 0.3]])
        model(sample).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        return model, optimizer, scheduler

    def _write(
        self,
        root: Path,
        generation: str,
        model: object,
        optimizer: object,
        scheduler: object,
    ) -> dict[str, object]:
        return create_checkpoint_generation(
            root,
            generation,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            epoch=7,
            global_step=123,
            provenance_hashes={
                "data_manifest": _digest("data"),
                "source_tree": _digest("source"),
            },
            config={
                "model": {"width": 4, "dropout": 0.0},
                "training": {"world_size": 8, "precision": "bf16"},
            },
        )

    def test_roundtrip_restores_components_and_all_rng_states(self) -> None:
        import torch

        random.seed(1729)
        np.random.seed(1729)
        torch.manual_seed(1729)
        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = self._write(
                root,
                "epoch_0007_step_000123",
                model,
                optimizer,
                scheduler,
            )
            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(5)
            expected_parameters = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            expected_scheduler = scheduler.state_dict().copy()

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            optimizer.param_groups[0]["lr"] = 99.0
            scheduler.last_epoch = 999
            for _ in range(10):
                random.random()
                np.random.random()
                torch.rand(1)

            metadata = restore_checkpoint_generation(
                root,
                "epoch_0007_step_000123",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_provenance_hashes={
                    "source_tree": _digest("source"),
                    "data_manifest": _digest("data"),
                },
            )
            self.assertEqual(metadata["epoch"], 7)
            self.assertEqual(metadata["global_step"], 123)
            self.assertEqual(metadata["checkpoint_sha256"], receipt["checkpoint_sha256"])
            for key, expected in expected_parameters.items():
                torch.testing.assert_close(model.state_dict()[key], expected)
            self.assertEqual(scheduler.state_dict(), expected_scheduler)
            self.assertAlmostEqual(random.random(), expected_python, places=15)
            self.assertAlmostEqual(float(np.random.random()), expected_numpy, places=15)
            torch.testing.assert_close(torch.rand(5), expected_torch)

    def test_existing_generation_is_rejected_without_overwrite(self) -> None:
        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._write(root, "generation_1", model, optimizer, scheduler)
            with self.assertRaises(FileExistsError):
                self._write(root, "generation_1", model, optimizer, scheduler)
            loaded = load_checkpoint_generation(root, "generation_1")
            self.assertEqual(
                loaded["checkpoint_sha256"],
                first["checkpoint_sha256"],
            )

    def test_published_artifacts_are_read_only(self) -> None:
        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "generation_read_only", model, optimizer, scheduler)
            generation = root / "generation_read_only"
            for name in ("checkpoint.pt", "receipt.json"):
                mode = stat.S_IMODE((generation / name).stat().st_mode)
                self.assertEqual(mode, 0o444)

    def test_checkpoint_corruption_is_rejected_before_deserialization(self) -> None:
        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "generation_2", model, optimizer, scheduler)
            checkpoint = root / "generation_2" / CHECKPOINT_FILENAME
            checkpoint.chmod(0o644)
            with checkpoint.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "size|SHA-256"):
                load_checkpoint_generation(root, "generation_2")

    def test_provenance_and_component_mismatch_fail_closed(self) -> None:
        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "generation_3", model, optimizer, scheduler)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_checkpoint_generation(
                    root,
                    "generation_3",
                    expected_provenance_hashes={"source_tree": _digest("wrong")},
                )
            with self.assertRaisesRegex(ValueError, "optimizer presence"):
                restore_checkpoint_generation(
                    root,
                    "generation_3",
                    model=model,
                    optimizer=None,
                    scheduler=scheduler,
                    scaler=None,
                    restore_random_generators=False,
                )

    def test_rank_specific_rng_state_is_restored(self) -> None:
        import torch

        model, optimizer, scheduler = self._components()
        rank_states = []
        expected = []
        for seed in (101, 202):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rank_states.append(capture_rng_state())
            expected.append(
                (
                    random.random(),
                    float(np.random.random()),
                    torch.rand(3),
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_checkpoint_generation(
                root,
                "distributed_generation",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                epoch=8,
                global_step=200,
                provenance_hashes={"source": _digest("source")},
                config={"world_size": 2},
                rng_states=rank_states,
            )
            metadata = restore_checkpoint_generation(
                root,
                "distributed_generation",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                rng_rank=1,
            )
            self.assertEqual(metadata["rng_world_size"], 2)
            self.assertAlmostEqual(random.random(), expected[1][0], places=15)
            self.assertAlmostEqual(float(np.random.random()), expected[1][1], places=15)
            torch.testing.assert_close(torch.rand(3), expected[1][2])

    def test_invalid_rng_rank_is_rejected(self) -> None:
        import torch

        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "generation_4", model, optimizer, scheduler)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(17.0)
            changed = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            with self.assertRaisesRegex(ValueError, "rng_rank"):
                restore_checkpoint_generation(
                    root,
                    "generation_4",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    rng_rank=1,
                )
            for key in changed:
                torch.testing.assert_close(model.state_dict()[key], changed[key])

    def test_external_receipt_anchor_rejects_reissued_checkpoint(self) -> None:
        model, optimizer, scheduler = self._components()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write(root, "anchored", model, optimizer, scheduler)
            receipt_path = root / "anchored" / RECEIPT_FILENAME
            expected_receipt = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

            loaded = load_checkpoint_generation(
                root,
                "anchored",
                expected_receipt_sha256=expected_receipt,
            )
            self.assertEqual(loaded["receipt_sha256"], expected_receipt)
            self.assertTrue(loaded["external_receipt_anchor_verified"])
            with self.assertRaisesRegex(ValueError, "external anchor"):
                load_checkpoint_generation(
                    root,
                    "anchored",
                    expected_receipt_sha256=_digest("different-receipt"),
                )

    def test_malformed_rng_state_is_rejected_before_publication(self) -> None:
        model, optimizer, scheduler = self._components()
        state = capture_rng_state()
        state["numpy"]["position"] = 625
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "position"):
                create_checkpoint_generation(
                    Path(directory),
                    "bad-rng",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    epoch=0,
                    global_step=0,
                    provenance_hashes={"source": _digest("source")},
                    config={"world_size": 1},
                    rng_states=[state],
                )
            self.assertFalse((Path(directory) / "bad-rng").exists())
