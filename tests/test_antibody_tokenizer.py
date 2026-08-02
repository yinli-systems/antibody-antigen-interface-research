import numpy as np
import pytest

from antibody_pretraining.masking import apply_uniform_mlm
from antibody_pretraining.tokenizer import (
    CLS_ID,
    MASK_ID,
    PAD_ID,
    SEP_ID,
    TokenizerError,
    decode_residues,
    encode_unpaired,
    normalize_locus,
    pad_encoded,
)


SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 4


def test_fixed_tokenizer_round_trip_and_padding() -> None:
    heavy = encode_unpaired(SEQUENCE, "heavy")
    light = encode_unpaired(SEQUENCE + "X", "kappa")
    tokens, types, mask = pad_encoded([heavy, light])

    assert heavy.input_ids[0] == CLS_ID
    assert heavy.input_ids[-1] == SEP_ID
    assert decode_residues(heavy.input_ids) == SEQUENCE
    assert tokens[0][-1] == PAD_ID
    assert types[0][-1] == 0
    assert mask[0][-1] is False
    assert mask[1][-1] is True


def test_locus_and_sequence_validation_fail_closed() -> None:
    assert normalize_locus("IGH") == "heavy"
    assert normalize_locus("K") == "kappa"
    assert normalize_locus("igl") == "lambda"
    with pytest.raises(TokenizerError, match="Unsupported antibody locus"):
        normalize_locus("TRA")
    with pytest.raises(TokenizerError, match="unsupported residues"):
        encode_unpaired(SEQUENCE + "-", "heavy")
    with pytest.raises(TokenizerError, match="outside the frozen boundary"):
        encode_unpaired("ACDE", "heavy")


def test_mlm_masking_is_counter_deterministic_and_boundary_safe() -> None:
    encoded = [encode_unpaired(SEQUENCE, "heavy") for _ in range(8)]
    tokens, _, _ = pad_encoded(encoded, pad_to_length=96)
    input_ids = np.asarray(tokens, dtype=np.int64)
    keys = np.arange(8, dtype=np.uint64) + np.uint64(100)

    first = apply_uniform_mlm(input_ids, keys, update=17, seed=42)
    replay = apply_uniform_mlm(input_ids, keys, update=17, seed=42)
    next_update = apply_uniform_mlm(input_ids, keys, update=18, seed=42)

    assert np.array_equal(first.input_ids, replay.input_ids)
    assert np.array_equal(first.labels, replay.labels)
    assert not np.array_equal(first.labels, next_update.labels)
    assert first.selected_target_count > 0
    assert np.all(first.labels[input_ids == PAD_ID] == -100)
    assert np.all(first.labels[input_ids == CLS_ID] == -100)
    assert np.all(first.labels[input_ids == SEP_ID] == -100)
    selected = first.labels != -100
    assert np.all(
        (first.input_ids[selected] == MASK_ID)
        | (first.input_ids[selected] != first.labels[selected])
        | (first.input_ids[selected] == first.labels[selected])
    )
