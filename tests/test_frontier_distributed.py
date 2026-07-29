"""Tests for distributed contrastive-training primitives."""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from interaction_first.frontier_distributed import (
    EXACT_ALL_REDUCE_GRADIENT_BACKEND,
    LEGACY_AUTOGRAD_GRADIENT_BACKEND,
    LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
    exact_distributed_mean,
    gather_equal_with_grad,
    symmetric_score_matrix_info_nce,
    symmetric_multi_positive_info_nce,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _distributed_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
) -> None:
    """Exercise collective forward and backward semantics in one worker."""
    import torch

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
    )
    try:
        gathered_input = torch.tensor(
            [[float(rank + 1)]],
            dtype=torch.float64,
            requires_grad=True,
        )
        gathered = gather_equal_with_grad(
            gathered_input,
            gradient_backend=EXACT_ALL_REDUCE_GRADIENT_BACKEND,
        )
        gathered_loss = float(rank + 1) * gathered.square().mean()
        gathered_loss.backward()

        mean_input = torch.tensor(
            float(rank + 1),
            dtype=torch.float64,
            requires_grad=True,
        )
        mean = exact_distributed_mean(mean_input, rank + 1)
        mean.backward()
        torch.save(
            {
                "gathered": gathered.detach(),
                "gathered_gradient": gathered_input.grad.detach(),
                "mean": mean.detach(),
                "mean_gradient": mean_input.grad.detach(),
            },
            Path(output_directory) / f"rank_{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


def _ddp_gradient_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_directory: str,
) -> None:
    """Compare exact gather gradients through a real DDP parameter reducer."""
    import torch

    class SharedPairProjection(torch.nn.Module):
        """Project both partners through one shared trainable matrix."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(
                    [[0.8, -0.2], [0.1, 0.7]],
                    dtype=torch.float64,
                )
            )
            self.log_temperature = torch.nn.Parameter(
                torch.tensor(math.log(0.4), dtype=torch.float64)
            )

        def forward(
            self,
            antibody_features: object,
            antigen_features: object,
        ) -> tuple[object, object, object]:
            return (
                antibody_features @ self.weight,
                antigen_features @ self.weight,
                self.log_temperature.exp(),
            )

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
    )
    try:
        global_antibody = torch.tensor(
            [
                [1.0, 0.2],
                [0.1, 1.1],
                [0.8, 0.7],
                [40.0, -30.0],
            ],
            dtype=torch.float64,
        )
        global_antigen = torch.tensor(
            [
                [0.9, 0.3],
                [0.2, 1.0],
                [0.6, 0.9],
                [-25.0, 35.0],
            ],
            dtype=torch.float64,
        )
        start = rank * 2
        local_antibody = global_antibody[start : start + 2].clone().requires_grad_()
        local_antigen = global_antigen[start : start + 2].clone().requires_grad_()
        local_valid = torch.tensor(
            [True, True] if rank == 0 else [False, False],
            dtype=torch.bool,
        )
        positive = torch.eye(4, dtype=torch.bool)
        positive[2, 2] = False
        positive[3, 3] = False
        excluded = torch.zeros(4, 4, dtype=torch.bool)
        excluded[2:, :] = True
        excluded[:, 2:] = True

        model = SharedPairProjection()
        distributed_model = torch.nn.parallel.DistributedDataParallel(model)
        antibody_embeddings, antigen_embeddings, temperature = distributed_model(
            local_antibody,
            local_antigen,
        )
        details = symmetric_multi_positive_info_nce(
            antibody_embeddings,
            antigen_embeddings,
            positive,
            excluded,
            temperature=temperature,
            local_valid_mask=local_valid,
            gather_gradient_backend=LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
            return_details=True,
        )
        details["loss"].backward()
        torch.save(
            {
                "loss": details["loss"].detach(),
                "parameter_gradient": model.weight.grad.detach(),
                "temperature_gradient": model.log_temperature.grad.detach(),
                "local_antibody_gradient": local_antibody.grad.detach(),
                "local_antigen_gradient": local_antigen.grad.detach(),
                "gather_gradient_backend": details["gather_gradient_backend"],
                "global_padded_count": details["global_padded_count"],
            },
            Path(output_directory) / f"ddp_rank_{rank}.pt",
        )
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ContrastiveLossTests(unittest.TestCase):
    """Verify multi-positive masking, gradients, and input contracts."""

    def test_single_rank_gather_is_identity_with_gradient(self) -> None:
        import torch

        value = torch.tensor([[1.0, 2.0]], requires_grad=True)
        gathered = gather_equal_with_grad(value)
        self.assertIs(gathered, value)
        gathered.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value))

    def test_gather_backend_selection_is_explicit(self) -> None:
        import torch

        value = torch.tensor([[1.0]], requires_grad=True)
        legacy = gather_equal_with_grad(
            value,
            gradient_backend=LEGACY_AUTOGRAD_GRADIENT_BACKEND,
        )
        self.assertIs(legacy, value)
        with self.assertRaisesRegex(ValueError, "Unsupported gather"):
            gather_equal_with_grad(value, gradient_backend="automatic")
        with self.assertRaisesRegex(TypeError, "must be a string"):
            gather_equal_with_grad(value, gradient_backend=None)

    def test_symmetric_loss_matches_manual_multi_positive_calculation(self) -> None:
        import torch

        antibody = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
            dtype=torch.float64,
            requires_grad=True,
        )
        antigen = torch.tensor(
            [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        positives = torch.tensor(
            [
                [True, True, False],
                [False, True, False],
                [False, False, True],
            ]
        )
        exclusions = torch.tensor(
            [
                [False, False, True],
                [True, False, False],
                [False, True, False],
            ]
        )
        temperature = 0.4
        details = symmetric_multi_positive_info_nce(
            antibody,
            antigen,
            positives,
            exclusions,
            temperature=temperature,
            return_details=True,
        )

        logits = antibody @ antigen.transpose(0, 1) / temperature

        def manual_direction(
            score_matrix: object,
            positive_mask: object,
            exclusion_mask: object,
        ) -> object:
            negative_infinity = torch.tensor(
                float("-inf"),
                dtype=score_matrix.dtype,
            )
            numerator = torch.logsumexp(
                torch.where(positive_mask, score_matrix, negative_infinity),
                dim=1,
            )
            denominator = torch.logsumexp(
                torch.where(~exclusion_mask, score_matrix, negative_infinity),
                dim=1,
            )
            return (denominator - numerator).mean()

        expected_forward = manual_direction(logits, positives, exclusions)
        expected_reverse = manual_direction(
            logits.transpose(0, 1),
            positives.transpose(0, 1),
            exclusions.transpose(0, 1),
        )
        torch.testing.assert_close(
            details["antibody_to_antigen"],
            expected_forward,
        )
        torch.testing.assert_close(
            details["antigen_to_antibody"],
            expected_reverse,
        )
        torch.testing.assert_close(
            details["loss"],
            0.5 * (expected_forward + expected_reverse),
        )
        details["loss"].backward()
        self.assertTrue(torch.isfinite(antibody.grad).all())
        self.assertTrue(torch.isfinite(antigen.grad).all())
        self.assertGreater(float(antibody.grad.abs().sum()), 0.0)
        self.assertEqual(details["global_antibody_count"], 3)
        self.assertEqual(
            details["gather_gradient_backend"],
            LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
        )

    def test_loss_rejects_missing_positives_and_ambiguous_positives(self) -> None:
        import torch

        antibody = torch.eye(2)
        antigen = torch.eye(2)
        missing = torch.tensor([[True, False], [False, False]])
        with self.assertRaisesRegex(ValueError, "observed positive"):
            symmetric_multi_positive_info_nce(antibody, antigen, missing)

        positives = torch.eye(2, dtype=torch.bool)
        exclusions = positives.clone()
        with self.assertRaisesRegex(ValueError, "cannot be excluded"):
            symmetric_multi_positive_info_nce(
                antibody,
                antigen,
                positives,
                exclusions,
            )

    def test_explicit_score_loss_matches_embedding_loss(self) -> None:
        import torch

        antibody = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        antigen = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        positives = torch.eye(2, dtype=torch.bool)
        embedding_loss = symmetric_multi_positive_info_nce(
            antibody,
            antigen,
            positives,
            temperature=0.2,
        )
        score_loss = symmetric_score_matrix_info_nce(
            antibody @ antigen.transpose(0, 1),
            positives,
            temperature=0.2,
        )
        torch.testing.assert_close(score_loss, embedding_loss)

    def test_explicit_score_loss_uses_float32_for_bfloat16(self) -> None:
        import torch

        scores = torch.tensor(
            [[1.0, -1.0], [-0.5, 0.5]],
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        loss = symmetric_score_matrix_info_nce(
            scores,
            torch.eye(2, dtype=torch.bool),
            temperature=0.07,
        )
        self.assertEqual(loss.dtype, torch.float32)
        reference = symmetric_score_matrix_info_nce(
            scores.detach().float(),
            torch.eye(2, dtype=torch.bool),
            temperature=torch.tensor(0.07, dtype=torch.float32),
        )
        torch.testing.assert_close(loss.detach(), reference, rtol=0.0, atol=0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_exact_mean_single_rank_uses_sample_count(self) -> None:
        import torch

        local_sum = torch.tensor(7.5, dtype=torch.float64, requires_grad=True)
        mean = exact_distributed_mean(local_sum, 3)
        torch.testing.assert_close(mean, torch.tensor(2.5, dtype=torch.float64))
        mean.backward()
        torch.testing.assert_close(
            local_sum.grad,
            torch.tensor(1.0 / 3.0, dtype=torch.float64),
        )

    def test_masks_require_exact_boolean_tensors_without_coercion(self) -> None:
        import torch

        scores = torch.eye(2)
        numeric_mask = torch.eye(2, dtype=torch.int64)
        with self.assertRaisesRegex(TypeError, "torch.bool"):
            symmetric_score_matrix_info_nce(scores, numeric_mask)
        with self.assertRaisesRegex(TypeError, "PyTorch tensor"):
            symmetric_score_matrix_info_nce(
                scores,
                [[True, False], [False, True]],
            )

    def test_runtime_details_separate_observed_scale_from_contract(self) -> None:
        import torch

        details = symmetric_multi_positive_info_nce(
            torch.eye(2),
            torch.eye(2),
            torch.eye(2, dtype=torch.bool),
            return_details=True,
        )
        self.assertEqual(details["local_slot_count_observed"], 2)
        self.assertEqual(details["global_slot_count_observed"], 2)
        self.assertEqual(details["distributed_world_size_observed"], 1)
        self.assertEqual(details["distributed_rank_observed"], 0)
        self.assertEqual(
            details["ddp_averaging_contract"],
            "fixed_world_size_standard_ddp_gradient_average",
        )

    def test_exact_mean_rejects_nonfinite_and_nonfloating_numerators(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "finite"):
            exact_distributed_mean(torch.tensor(float("nan")), 1)
        with self.assertRaisesRegex(TypeError, "floating-point"):
            exact_distributed_mean(torch.tensor(3, dtype=torch.int64), 1)
        with self.assertRaisesRegex(TypeError, "float32 or float64"):
            exact_distributed_mean(torch.tensor(3.0, dtype=torch.bfloat16), 1)
        with self.assertRaisesRegex(TypeError, "boolean"):
            exact_distributed_mean(torch.tensor(3.0), True)


@unittest.skipUnless(
    TORCH_AVAILABLE
    and importlib.util.find_spec("torch.distributed") is not None,
    "PyTorch distributed support is required",
)
class MultiProcessCollectiveTests(unittest.TestCase):
    """Verify real two-rank CPU collective semantics."""

    def test_two_rank_gather_and_exact_mean(self) -> None:
        import torch

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_gloo_available()
        ):
            self.skipTest("The Gloo backend is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendezvous_path = root / "rendezvous"
            rendezvous = f"file://{rendezvous_path}"
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(
                    target=_distributed_worker,
                    args=(rank, 2, rendezvous, str(root)),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("Distributed worker timed out")
                self.assertEqual(process.exitcode, 0)

            for rank in range(2):
                result = torch.load(
                    root / f"rank_{rank}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                torch.testing.assert_close(
                    result["gathered"],
                    torch.tensor([[1.0], [2.0]], dtype=torch.float64),
                )
                torch.testing.assert_close(
                    result["gathered_gradient"],
                    torch.tensor([[3.0 * (rank + 1)]], dtype=torch.float64),
                )
                torch.testing.assert_close(
                    result["mean"],
                    torch.tensor(1.0, dtype=torch.float64),
                )
                torch.testing.assert_close(
                    result["mean_gradient"],
                    torch.tensor(2.0 / 3.0, dtype=torch.float64),
                )

    def test_replicated_global_loss_matches_single_process_parameter_gradient(
        self,
    ) -> None:
        import torch

        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_gloo_available()
        ):
            self.skipTest("The Gloo backend is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendezvous_path = root / "ddp-rendezvous"
            rendezvous = f"file://{rendezvous_path}"
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(
                    target=_ddp_gradient_worker,
                    args=(rank, 2, rendezvous, str(root)),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("Distributed DDP worker timed out")
                self.assertEqual(process.exitcode, 0)

            antibody = torch.tensor(
                [
                    [1.0, 0.2],
                    [0.1, 1.1],
                    [0.8, 0.7],
                    [40.0, -30.0],
                ],
                dtype=torch.float64,
                requires_grad=True,
            )
            antigen = torch.tensor(
                [
                    [0.9, 0.3],
                    [0.2, 1.0],
                    [0.6, 0.9],
                    [-25.0, 35.0],
                ],
                dtype=torch.float64,
                requires_grad=True,
            )
            weight = torch.tensor(
                [[0.8, -0.2], [0.1, 0.7]],
                dtype=torch.float64,
                requires_grad=True,
            )
            positive = torch.eye(4, dtype=torch.bool)
            positive[2, 2] = False
            positive[3, 3] = False
            excluded = torch.zeros(4, 4, dtype=torch.bool)
            excluded[2:, :] = True
            excluded[:, 2:] = True
            log_temperature = torch.tensor(
                math.log(0.4),
                dtype=torch.float64,
                requires_grad=True,
            )
            reference = symmetric_multi_positive_info_nce(
                antibody @ weight,
                antigen @ weight,
                positive,
                excluded,
                temperature=log_temperature.exp(),
                local_valid_mask=torch.tensor([True, True, False, False]),
                return_details=True,
            )
            reference["loss"].backward()

            for rank in range(2):
                result = torch.load(
                    root / f"ddp_rank_{rank}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                torch.testing.assert_close(result["loss"], reference["loss"])
                torch.testing.assert_close(
                    result["parameter_gradient"],
                    weight.grad,
                    rtol=1e-12,
                    atol=1e-12,
                )
                torch.testing.assert_close(
                    result["temperature_gradient"],
                    log_temperature.grad,
                    rtol=1e-12,
                    atol=1e-12,
                )
                self.assertEqual(
                    result["gather_gradient_backend"],
                    LOCAL_SLICE_DDP_EXACT_GRADIENT_BACKEND,
                )
                self.assertEqual(result["global_padded_count"], 2)

            rank_one = torch.load(
                root / "ddp_rank_1.pt",
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(
                rank_one["local_antibody_gradient"],
                torch.zeros(2, 2, dtype=torch.float64),
            )
            torch.testing.assert_close(
                rank_one["local_antigen_gradient"],
                torch.zeros(2, 2, dtype=torch.float64),
            )
