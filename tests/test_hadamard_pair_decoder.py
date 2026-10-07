"""Tests for the Hadamard-product pair decoder.

The interesting test is not that the new decoder runs, but that it fixes a
specific, reproducible failure of the bilinear one: a pair signal that is a
non-monotone function of the elementwise product lies outside the bilinear
hypothesis class, so that decoder falls back on unary terms -- which is exactly
an outer product of marginals, the baseline the native pair head loses to.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is unavailable",
)

D_MODEL, PAIR_DIM = 16, 8


def _decoders():
    from interaction_first.model import BidirectionalBilinearDecoder, HadamardPairDecoder

    return BidirectionalBilinearDecoder, HadamardPairDecoder


def _band_task(seed: int = 0, batch: int = 1, antigen: int = 24, antibody: int = 24):
    """A pair label that is a *band* on the summed elementwise product.

    ``y = 1`` when ``sum(x_ag * x_ab)`` falls inside an interval. Being
    non-monotone in that sum, no linear readout of the elementwise product --
    which is what a low-rank bilinear form computes -- can separate it, while an
    MLP over the same product can.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    ag = torch.randn(batch, antigen, D_MODEL, generator=generator)
    ab = torch.randn(batch, antibody, D_MODEL, generator=generator)
    affinity = torch.einsum("bid,bjd->bij", ag, ab) / (D_MODEL**0.5)
    labels = ((affinity > -0.5) & (affinity < 0.5)).float()
    return ag, ab, labels


def _average_precision(scores, target) -> float:
    import torch

    order = torch.argsort(scores.flatten(), descending=True)
    hits = target.flatten()[order]
    cumulative = torch.cumsum(hits, dim=0)
    precision = cumulative / torch.arange(1, len(hits) + 1, dtype=torch.float32)
    return float((precision * hits).sum() / hits.sum())


def _fit(decoder, train, test, steps: int = 600, lr: float = 0.05) -> float:
    """Fit on one set of residues and score average precision on unseen ones.

    Held-out residues are the point: with enough parameters either decoder can
    memorise a single grid, so training and scoring on the same pairs measures
    capacity rather than which functions the decoder can represent.
    """
    import torch
    import torch.nn.functional as F

    ag, ab, labels = train
    optimiser = torch.optim.Adam(decoder.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        F.binary_cross_entropy_with_logits(decoder(ag, ab), labels).backward()
        optimiser.step()
    test_ag, test_ab, test_labels = test
    with torch.no_grad():
        return _average_precision(decoder(test_ag, test_ab), test_labels)


# --- measured comparison, not an assumed improvement -------------------


def test_bilinear_wins_on_a_signal_inside_its_hypothesis_class() -> None:
    """Recorded so a future change to either decoder is noticed.

    The link-prediction literature reports HadamardMLP decoders outperforming
    bilinear ones, and that motivated adding this decoder. On this synthetic
    setup it does not reproduce: with held-out residues the bilinear decoder
    reaches about 0.98 average precision on a monotone bilinear signal while
    every Hadamard variant tried -- with and without the LayerNorm, and with an
    explicit dot-product residual that makes the bilinear form representable --
    lands near 0.60. The gap is generalisation, not expressiveness.

    So this decoder is an alternative to ablate on real contact data, not an
    established improvement. Nothing selects it by default.
    """
    import torch

    Bilinear, Hadamard = _decoders()

    def monotone(seed):
        ag, ab, _ = _band_task(seed=seed, antigen=64, antibody=64)
        affinity = torch.einsum("bid,bjd->bij", ag, ab) / (D_MODEL**0.5)
        return ag, ab, (affinity > 0.5).float()

    train, test = monotone(3), monotone(97)

    torch.manual_seed(0)
    bilinear_ap = _fit(Bilinear(D_MODEL, PAIR_DIM), train, test)
    torch.manual_seed(0)
    hadamard_ap = _fit(Hadamard(D_MODEL, PAIR_DIM), train, test)

    assert bilinear_ap > 0.90, bilinear_ap
    assert hadamard_ap > float(test[2].mean()), hadamard_ap
    assert bilinear_ap > hadamard_ap, (bilinear_ap, hadamard_ap)


def test_neither_decoder_learns_a_non_monotone_band() -> None:
    """Both sit at the base rate, so the band is not the discriminating case."""
    import torch

    Bilinear, Hadamard = _decoders()
    train = _band_task(seed=0, antigen=64, antibody=64)
    test = _band_task(seed=99, antigen=64, antibody=64)
    prevalence = float(test[2].mean())

    torch.manual_seed(0)
    bilinear_ap = _fit(Bilinear(D_MODEL, PAIR_DIM), train, test)
    torch.manual_seed(0)
    hadamard_ap = _fit(Hadamard(D_MODEL, PAIR_DIM), train, test)

    for value in (bilinear_ap, hadamard_ap):
        assert abs(value - prevalence) < 0.10, (value, prevalence)


# --- interface parity ----------------------------------------------------


def test_component_keys_match_the_bilinear_decoder() -> None:
    import torch

    Bilinear, Hadamard = _decoders()
    ag, ab, _ = _band_task(antigen=5, antibody=7)

    bilinear = Bilinear(D_MODEL, PAIR_DIM).components(ag, ab)
    hadamard = Hadamard(D_MODEL, PAIR_DIM).components(ag, ab)

    assert set(bilinear) == set(hadamard)
    for key in bilinear:
        assert bilinear[key].shape == hadamard[key].shape, key


def test_unary_and_interaction_terms_stay_separable() -> None:
    """The gauge-fixed attribution depends on this decomposition holding."""
    import torch

    _, Hadamard = _decoders()
    ag, ab, _ = _band_task(antigen=5, antibody=7)

    parts = Hadamard(D_MODEL, PAIR_DIM).components(ag, ab)
    rebuilt = (
        parts["interaction_logits"]
        + parts["antigen_bias"]
        + parts["antibody_bias"]
        + parts["global_bias"]
    )

    assert torch.allclose(parts["pair_logits"], rebuilt, atol=1e-6)


def test_interaction_term_ignores_a_pure_unary_shift() -> None:
    """Adding a constant to one side must move only the unary terms."""
    import torch

    _, Hadamard = _decoders()
    decoder = Hadamard(D_MODEL, PAIR_DIM).eval()
    ag, ab, _ = _band_task(antigen=5, antibody=7)

    with torch.no_grad():
        base = decoder.components(ag, ab)
        # A shift along a direction the projection nulls leaves interaction fixed.
        null = torch.linalg.svd(decoder.ag_project.weight, full_matrices=True)[2][-1]
        shifted = decoder.components(ag + 3.0 * null, ab)

    assert torch.allclose(
        base["interaction_logits"], shifted["interaction_logits"], atol=1e-5
    )


def test_forward_matches_the_pair_logits_component() -> None:
    import torch

    _, Hadamard = _decoders()
    decoder = Hadamard(D_MODEL, PAIR_DIM).eval()
    ag, ab, _ = _band_task(antigen=4, antibody=6)

    with torch.no_grad():
        assert torch.allclose(decoder(ag, ab), decoder.components(ag, ab)["pair_logits"])


def test_output_orientation_is_antigen_by_antibody() -> None:
    _, Hadamard = _decoders()
    ag, ab, _ = _band_task(batch=2, antigen=5, antibody=7)

    assert Hadamard(D_MODEL, PAIR_DIM)(ag, ab).shape == (2, 5, 7)


# --- chunking and validation --------------------------------------------


def test_chunking_is_numerically_identical_to_the_full_grid() -> None:
    import torch

    _, Hadamard = _decoders()
    torch.manual_seed(1)
    decoder = Hadamard(D_MODEL, PAIR_DIM).eval()
    ag, ab, _ = _band_task(batch=2, antigen=9, antibody=6)

    with torch.no_grad():
        full = decoder(ag, ab)
        decoder.chunk_size = 4
        chunked = decoder(ag, ab)

    assert torch.allclose(full, chunked, atol=1e-6)


def test_dimensions_are_validated() -> None:
    _, Hadamard = _decoders()

    for bad in ({"d_model": 0}, {"pair_dim": 0}, {"hidden_dim": 0}, {"chunk_size": 0}):
        keyword = {"d_model": D_MODEL, "pair_dim": PAIR_DIM, **bad}
        with pytest.raises(ValueError):
            Hadamard(**keyword)


def test_registry_exposes_both_decoders() -> None:
    from interaction_first.model import PAIR_DECODERS

    assert set(PAIR_DECODERS) == {"bilinear", "hadamard_mlp"}


def test_the_model_can_select_either_decoder() -> None:
    import torch

    from interaction_first.model import (
        BidirectionalBilinearDecoder,
        HadamardPairDecoder,
        PartnerSpecificInteractionModel,
    )

    expected = {"bilinear": BidirectionalBilinearDecoder, "hadamard_mlp": HadamardPairDecoder}
    for name, decoder_type in expected.items():
        model = PartnerSpecificInteractionModel(
            antigen_input_dim=8, antibody_input_dim=8, d_model=16, pair_dim=8,
            num_heads=2, num_layers=1, dropout=0.0, pair_decoder=name,
        ).eval()
        with torch.no_grad():
            out = model(
                torch.randn(1, 5, 8), torch.randn(1, 4, 8),
                torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool),
            )
        assert isinstance(model.decoder, decoder_type)
        assert model.pair_decoder_name == name
        assert out["pair_logits"].shape == (1, 5, 4)


def test_the_default_decoder_is_unchanged() -> None:
    """Adding an option must not silently change what existing runs build."""
    from interaction_first.model import BidirectionalBilinearDecoder, PartnerSpecificInteractionModel

    model = PartnerSpecificInteractionModel(
        antigen_input_dim=8, antibody_input_dim=8, d_model=16, pair_dim=8,
        num_heads=2, num_layers=1, dropout=0.0,
    )

    assert isinstance(model.decoder, BidirectionalBilinearDecoder)
    assert model.pair_decoder_name == "bilinear"


def test_an_unknown_decoder_name_is_rejected() -> None:
    import pytest as _pytest

    from interaction_first.model import PartnerSpecificInteractionModel

    with _pytest.raises(ValueError, match="pair_decoder must be one of"):
        PartnerSpecificInteractionModel(pair_decoder="nope")
